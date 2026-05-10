from __future__ import annotations

import contextlib
import io
import math
import queue
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from src.audio.config import AudioConfig
from src.audio.executor import evaluate_pipeline as evaluate_audio_pipeline
from src.audio.memory_manager import AudioMemoryManager
from src.audio.meta_learner import AudioMetaLearner
from src.audio.output_writer import save_processed_dataset as save_audio_processed_dataset
from src.audio.pipeline_generator import generate_pipelines as generate_audio_pipelines
from src.audio.profiler import profile_audio_dataset
from src.audio.reporter import generate_report as generate_audio_report, print_profile_summary as print_audio_profile_summary, save_report as save_audio_report
from src.audio.validator import validate_internal_audio_dataset
from src.utils.shared.selector import select_best
from src.utils.ingestion.audio import (
    get_audio_adapter,
    materialize_for_pipeline,
    has_class_labels as ds_has_class_labels,
    has_speaker_labels as ds_has_speaker_labels,
    has_speaker_pairs as ds_has_speaker_pairs,
    has_transcripts as ds_has_transcripts,
    has_temporal_segments as ds_has_temporal_segments,
    has_event_labels as ds_has_event_labels,
    has_anomaly_labels as ds_has_anomaly_labels,
    has_noisy_clean_pairs as ds_has_noisy_clean_pairs,
)


class AudioAgentWorker:
    def __init__(
        self,
        q: queue.Queue,
        zip_path: Path,
        metric: str,
        task_type: str = "classification",
        domain: str = "",
        constraints: str = "",
        notes: str = "",
        audio_format: str = "",
        channel_layout: str = "",
        sample_rate: str = "",
        input_format: str = "",
        input_format_key: str = "",
        metadata_path: str = "",
        record_path: str = "",
        field_overrides: Dict[str, str] = None,
    ) -> None:
        self.q = q
        self.zip_path = zip_path
        self.metric = metric
        self.task_type = task_type
        self.domain = domain
        self.constraints = constraints
        self.notes = notes
        self.audio_format = audio_format
        self.channel_layout = channel_layout
        self.sample_rate = sample_rate
        self.input_format = input_format
        self.input_format_key = input_format_key or "zip_folder"
        self.metadata_path = metadata_path
        self.record_path = record_path
        self.field_overrides = dict(field_overrides or {})

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            import traceback
            self._log("ERROR", f"Unexpected error: {exc}")
            self._log("ERROR", traceback.format_exc())
            self.q.put({"kind": "fail", "text": str(exc)})

    def _execute(self) -> None:
        config = AudioConfig(
            data_path=self.zip_path,
            metric=self.metric,
            task_type=self.task_type,
            domain=self.domain,
            constraints=self.constraints,
            notes=self.notes,
            modality="Audio",
            input_format=self.input_format,
            input_format_key=self.input_format_key,
            metadata_path=self.metadata_path,
            record_path=self.record_path,
            audio_format=self.audio_format,
            channel_layout=self.channel_layout,
            sample_rate=self.sample_rate,
            field_overrides=self.field_overrides,
        )
        tc = config.task_context()
        self._sep()
        self._log("INFO", f"Dataset : {self.zip_path.name}")
        self._log("INFO", "Modality: Audio")
        self._log("INFO", f"Metric  : {self.metric}")
        self._log("INFO", f"Task    : {tc.get('task_type', '')}")
        if self.input_format:
            self._log("INFO", f"Input fmt: {self.input_format}")
        self._sep()

        adapter = get_audio_adapter(self.input_format_key)
        if adapter is None:
            self._log("ERROR", f"Unsupported audio input format: '{self.input_format_key}'.")
            self.q.put({"kind": "fail", "text": f"Unsupported audio input format: '{self.input_format_key}'."})
            return

        self._step("[1/9] Validating input package ...")
        validation = adapter.validate_input(self.zip_path)
        if not validation.ok:
            for err in (validation.errors or [validation.message]):
                self._log("ERROR", f"  {err}")
            self.q.put({"kind": "fail", "text": validation.message or "Validation failed."})
            return
        self._log("OK", "  Package is valid.")

        tmp_dir = tempfile.mkdtemp(prefix="morphai_audio_")
        try:
            extract_dir = Path(tmp_dir) / "extracted"
            self._step("[1b/9] Parsing input format and building internal audio dataset ...")
            parse_kwargs: Dict[str, Any] = {"task_type": self.task_type}
            if self.metadata_path:
                parse_kwargs["metadata_path"] = self.metadata_path
            if self.record_path:
                parse_kwargs["record_path"] = self.record_path
            for name in (
                "audio_path_field", "label_field", "transcript_field", "speaker_field",
                "event_field", "anomaly_field", "noisy_field", "clean_field",
                "pair_a_field", "pair_b_field", "same_speaker_field",
            ):
                value = self.field_overrides.get(name, "")
                if value:
                    parse_kwargs[name] = value
            adapter_result = adapter.to_internal_dataset(self.zip_path, work_dir=extract_dir, **parse_kwargs)
            if not adapter_result.ok:
                for err in (adapter_result.errors or [adapter_result.message]):
                    self._log("ERROR", f"  {err}")
                self.q.put({"kind": "fail", "text": adapter_result.message or "Failed to parse audio input format."})
                return
            internal_dataset = adapter_result.data["internal_dataset"]
            n_parsed = len(internal_dataset.samples)
            self._log("OK", f"  Parsed {n_parsed} audio sample(s) via {self.input_format or self.input_format_key} adapter.")
            for w in (internal_dataset.warnings or [])[:5]:
                self._log("WARN", f"  parser: {w}")

            self._step("[2/9] Validating internal audio dataset against task ...")
            val_errors = validate_internal_audio_dataset(config, internal_dataset)
            if val_errors:
                for err in val_errors:
                    self._log("ERROR", f"  {err}")
                self.q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(val_errors)})
                return
            self._log("OK", "  Validation passed.")

            self._step("[2b/9] Materializing internal dataset for pipeline ...")
            materialized = Path(tmp_dir) / "materialized"
            dataset_root = materialize_for_pipeline(internal_dataset, materialized, self.task_type)
            self._log("OK", "  Internal dataset prepared for pipeline.")

            self._step("[3/9] Profiling audio dataset ...")
            profile = profile_audio_dataset(dataset_root)
            profile.input_format = self.input_format_key
            profile.parsing_summary = dict(internal_dataset.parsing_summary or {})
            profile.metadata_profile = dict(internal_dataset.metadata_profile or {})
            profile.structure_profile = dict(internal_dataset.structure_profile or {})
            profile.annotation_or_reference_profile = dict(internal_dataset.annotation_or_reference_profile or {})
            profile.parser_warnings = list(internal_dataset.warnings or [])
            profile.has_class_labels_flag = ds_has_class_labels(internal_dataset.samples)
            profile.has_speaker_labels_flag = ds_has_speaker_labels(internal_dataset.samples)
            profile.has_speaker_pairs_flag = ds_has_speaker_pairs(internal_dataset.samples)
            profile.has_transcripts_flag = ds_has_transcripts(internal_dataset.samples)
            profile.has_temporal_segments_flag = ds_has_temporal_segments(internal_dataset.samples)
            profile.has_event_labels_flag = ds_has_event_labels(internal_dataset.samples)
            profile.has_anomaly_labels_flag = ds_has_anomaly_labels(internal_dataset.samples)
            profile.has_noisy_clean_pairs_flag = ds_has_noisy_clean_pairs(internal_dataset.samples)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_audio_profile_summary(profile)
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self._log("INFO", line)
            self._log("OK", f"  {profile.n_audio_files:,} audio files across {profile.n_classes} label(s).")

            self._step("[4/9] Loading meta-learner ...")
            meta = AudioMetaLearner()
            meta.load()
            ms = meta.status_summary()
            self._log("INFO", f"  Meta-learner learning: {ms['n_train']}/{ms['min_to_use']} samples before activation." if not meta.is_mature else f"  Meta-learner active: {ms['n_train']} samples, weight={ms['weight']:.2f}")

            self._step("[5/9] Checking memory ...")
            memory = AudioMemoryManager()
            memory.load()
            good_cases, bad_cases = memory.find_good_and_bad(profile, self.metric, task_type=tc.get("task_type", ""))
            self._log("OK" if good_cases or bad_cases else "INFO", f"  {len(good_cases)} good + {len(bad_cases)} poor similar run(s) found." if good_cases or bad_cases else "  No similar past audio runs. Using heuristics only.")

            profile_summary = {
                "n_audio_files": profile.n_audio_files,
                "n_classes": profile.n_classes,
                "imbalance_ratio": profile.imbalance_ratio,
                "avg_duration_sec": profile.avg_duration_sec,
                "duration_std_sec": profile.duration_std_sec,
                "silence_ratio": profile.silence_ratio,
                "clipping_ratio": profile.clipping_ratio,
                "corruption_ratio": profile.corruption_ratio,
                "estimated_noise_ratio": profile.estimated_noise_ratio,
                "sample_rate_distribution": profile.sample_rate_distribution,
                "channel_count_distribution": profile.channel_count_distribution,
                "input_format": self.input_format_key,
                "has_class_labels": profile.has_class_labels_flag,
                "has_transcripts": profile.has_transcripts_flag,
                "has_speaker_labels": profile.has_speaker_labels_flag,
                "has_speaker_pairs": profile.has_speaker_pairs_flag,
                "has_temporal_segments": profile.has_temporal_segments_flag,
                "has_anomaly_labels": profile.has_anomaly_labels_flag,
                "has_noisy_clean_pairs": profile.has_noisy_clean_pairs_flag,
            }

            self._step("[6/9] Generating candidate pipelines ...")
            pipelines, mem_msgs = generate_audio_pipelines(profile, good_cases, bad_cases, meta_learner=meta, task_context=tc, profile_summary=profile_summary)
            self._log("OK", f"  {len(pipelines)} candidate(s) generated.")
            for msg in mem_msgs:
                self._log("INFO", f"  {msg}")
            mem_influence = {"good_injections": len(good_cases), "bad_avoidances": len(bad_cases), "meta_learner_weight": ms["weight"]}

            self._step("[7/9] Evaluating pipelines ...")
            results: List[Dict[str, Any]] = []
            successful: List[Dict[str, Any]] = []
            for idx, spec in enumerate(pipelines, 1):
                short = spec.name() if len(spec.name()) <= 52 else spec.name()[:49] + "..."
                self._log("INFO", f"  [{idx:2d}/{len(pipelines)}]  {short}")
                result = evaluate_audio_pipeline(spec, profile, config.task_type, config.metric)
                results.append(result)
                if result.get("success", True):
                    successful.append(result)
                    selected_metric = result.get("selected_metric", self.metric)
                    raw = result.get("raw_metrics", result["metrics"]).get(selected_metric, 0.0)
                    score = result.get("normalized_score", result.get("final_score", 0.0))
                    self._log("METRIC", f"           {selected_metric}={raw:.4f}  | normalized={score:.4f}  [{result['n_splits']}folds x {result.get('n_models', 1)}models x {result['elapsed_sec']:.2f}s]")
                else:
                    self._log("WARN", f"           [FAILED - score=0.0000] {result.get('reason', 'invalid evaluation')}")

            if not successful:
                self._log("ERROR", "All pipelines failed.")
                self.q.put({"kind": "fail", "text": "All candidate audio pipelines failed to produce a valid evaluation."})
                return

            self._step("[8/9] Selecting best pipeline ...")
            best = select_best(successful, self.metric)
            bs = best.get("normalized_score", best.get("final_score", 0.0))
            selected_metric = best.get("selected_metric", self.metric)
            sd = best.get("final_score_std", 0.0)
            selected_raw = best.get("raw_metrics", best["metrics"]).get(selected_metric, 0.0)
            self._log("BEST", f"  > {best['spec'].name()}")
            self._log("BEST", f"  > {selected_metric} = {selected_raw:.4f}")
            self._log("BEST", f"  > normalized score = {bs:.4f}  (+/- {sd:.4f})")

            self._step("[9/9] Saving processed audio zip ...")
            cleaned_path, cleaned_shape = save_audio_processed_dataset(best["spec"], profile, config, internal_dataset=internal_dataset)
            self._log("OK", f"  {cleaned_path}")

            self._step("[9/9] Saving report, updating memory and meta-learner ...")
            ms_final = meta.status_summary()
            report = generate_audio_report(profile, results, best, config, meta_status=ms_final, mem_influence=mem_influence)
            report_path = save_audio_report(report)
            outcome = memory.add_run(profile, config, results, best, meta_status=ms_final, mem_influence=mem_influence)
            memory.save()
            n_samples = meta.train_from_memory(memory.all_runs())
            if n_samples > 0:
                meta.save()
                ms_final = meta.status_summary()
            self._log("OK", f"  Report : {report_path}")
            self._log("OK", f"  Memory : {memory.n_runs} total run(s)  [{outcome}]")
            self._sep()
            self._log("OK", "Agent run complete.")
            self._sep()
            ir = profile.imbalance_ratio
            self.q.put({
                "kind": "done",
                "modality": "Audio",
                "report": report,
                "report_path": report_path,
                "cleaned_path": cleaned_path,
                "cleaned_shape": cleaned_shape,
                "best_name": best["spec"].name(),
                "best_score": bs,
                "best_score_std": sd,
                "metric": selected_metric,
                "metrics": best["metrics"],
                "raw_metrics": best.get("raw_metrics", best["metrics"]),
                "metrics_std": best.get("metrics_std", {}),
                "normalized_metrics": best.get("normalized_metrics", {}),
                "normalized_metrics_std": best.get("normalized_metrics_std", {}),
                "per_model": best.get("per_model_metrics", {}),
                "evaluation_mode": best.get("evaluation_mode", ""),
                "evaluation_summary": best.get("evaluation_summary", ""),
                "n_splits": best.get("n_splits", "?"),
                "n_models": best.get("n_models", 1),
                "n_pipelines": len(results),
                "n_audio_files": profile.n_audio_files,
                "n_classes": profile.n_classes,
                "avg_duration_sec": profile.avg_duration_sec,
                "total_duration_sec": profile.total_duration_sec,
                "sample_rates": ", ".join(profile.sample_rate_distribution.keys()) or "unknown",
                "quality_info": f"corrupt={profile.n_corrupt}, silent={profile.n_silent}, clipped={profile.n_clipped}",
                "imbalance_ratio": round(ir, 2) if math.isfinite(ir) else 999.9,
                "task_context": tc,
                "meta_status": ms_final,
                "mem_influence": mem_influence,
                "mem_update": outcome,
                "input_format": self.input_format,
                "input_format_key": self.input_format_key,
            })
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _log(self, level: str, text: str) -> None:
        self.q.put({"kind": "log", "level": level, "text": text})

    def _step(self, text: str) -> None:
        self.q.put({"kind": "log", "level": "STEP", "text": text})

    def _sep(self) -> None:
        self.q.put({"kind": "log", "level": "SEP", "text": "-" * 52})
