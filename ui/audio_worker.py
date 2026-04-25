from __future__ import annotations

import contextlib
import io
import math
import queue
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.audio.config import AudioConfig
from src.audio.executor import evaluate_pipeline as evaluate_audio_pipeline
from src.audio.memory_manager import AudioMemoryManager
from src.audio.meta_learner import AudioMetaLearner
from src.audio.output_writer import save_processed_dataset as save_audio_processed_dataset
from src.audio.pipeline_generator import generate_pipelines as generate_audio_pipelines
from src.audio.profiler import profile_audio_dataset
from src.audio.reporter import generate_report as generate_audio_report, print_profile_summary as print_audio_profile_summary, save_report as save_audio_report
from src.audio.validator import validate_audio_run, validate_audio_zip
from src.shared.selector import select_best

_AUD_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def _find_audio_dataset_root(extracted_dir: Path) -> Path:
    try:
        valid_subdirs = [d for d in extracted_dir.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "__MACOSX"]
    except Exception:
        return extracted_dir
    if len(valid_subdirs) == 1:
        try:
            if any(f.is_file() and f.suffix.lower() in _AUD_EXTENSIONS for f in valid_subdirs[0].rglob("*")):
                return valid_subdirs[0]
        except Exception:
            pass
    return extracted_dir


class AudioAgentWorker:
    def __init__(self, q: queue.Queue, zip_path: Path, metric: str, task_type: str = "classification", domain: str = "", constraints: str = "", notes: str = "", audio_format: str = "", channel_layout: str = "", sample_rate: str = "") -> None:
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

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            self._log("ERROR", f"Unexpected error: {exc}")
            self.q.put({"kind": "fail", "text": str(exc)})

    def _execute(self) -> None:
        config = AudioConfig(self.zip_path, self.metric, self.task_type, self.domain, self.constraints, self.notes, "Audio", self.audio_format, self.channel_layout, self.sample_rate)
        tc = config.task_context()
        self._sep()
        self._log("INFO", f"Dataset : {self.zip_path.name}")
        self._log("INFO", "Modality: Audio")
        self._log("INFO", f"Metric  : {self.metric}")
        self._log("INFO", f"Task    : {tc.get('task_type', '')}")
        self._sep()
        self._step("[1/9] Validating zip archive ...")
        zip_errors = validate_audio_zip(self.zip_path)
        if zip_errors:
            for err in zip_errors:
                self._log("ERROR", f"  {err}")
            self.q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(zip_errors)})
            return
        self._log("OK", "  Archive is valid.")
        tmp_dir = tempfile.mkdtemp(prefix="morphai_audio_")
        try:
            self._step("[1b/9] Extracting zip archive ...")
            with zipfile.ZipFile(str(self.zip_path), "r") as zf:
                zf.extractall(tmp_dir)
            dataset_root = _find_audio_dataset_root(Path(tmp_dir))
            self._log("OK", "  Extracted to temporary working directory.")
            self._step("[2/9] Validating extracted dataset structure ...")
            val_errors = validate_audio_run(config, dataset_root)
            if val_errors:
                for err in val_errors:
                    self._log("ERROR", f"  {err}")
                self.q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(val_errors)})
                return
            self._log("OK", "  Validation passed.")
            self._step("[3/9] Profiling audio dataset ...")
            profile = profile_audio_dataset(dataset_root)
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
            profile_summary = {"n_audio_files": profile.n_audio_files, "n_classes": profile.n_classes, "imbalance_ratio": profile.imbalance_ratio, "avg_duration_sec": profile.avg_duration_sec, "duration_std_sec": profile.duration_std_sec, "silence_ratio": profile.silence_ratio, "clipping_ratio": profile.clipping_ratio, "corruption_ratio": profile.corruption_ratio, "estimated_noise_ratio": profile.estimated_noise_ratio, "sample_rate_distribution": profile.sample_rate_distribution, "channel_count_distribution": profile.channel_count_distribution}
            self._step("[6/9] Generating candidate pipelines ...")
            pipelines, mem_msgs = generate_audio_pipelines(profile, good_cases, bad_cases, meta_learner=meta, task_context=tc, profile_summary=profile_summary)
            self._log("OK", f"  {len(pipelines)} candidate(s) generated.")
            for msg in mem_msgs:
                self._log("INFO", f"  {msg}")
            mem_influence = {"good_injections": len(good_cases), "bad_avoidances": len(bad_cases), "meta_learner_weight": ms["weight"]}
            self._step("[7/9] Evaluating pipelines ...")
            results = []
            successful = []
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
            cleaned_path, cleaned_shape = save_audio_processed_dataset(best["spec"], profile, config)
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
            self.q.put({"kind": "done", "modality": "Audio", "report": report, "report_path": report_path, "cleaned_path": cleaned_path, "cleaned_shape": cleaned_shape, "best_name": best["spec"].name(), "best_score": bs, "best_score_std": sd, "metric": selected_metric, "metrics": best["metrics"], "raw_metrics": best.get("raw_metrics", best["metrics"]), "metrics_std": best.get("metrics_std", {}), "normalized_metrics": best.get("normalized_metrics", {}), "normalized_metrics_std": best.get("normalized_metrics_std", {}), "per_model": best.get("per_model_metrics", {}), "evaluation_mode": best.get("evaluation_mode", ""), "evaluation_summary": best.get("evaluation_summary", ""), "n_splits": best.get("n_splits", "?"), "n_models": best.get("n_models", 1), "n_pipelines": len(results), "n_audio_files": profile.n_audio_files, "n_classes": profile.n_classes, "avg_duration_sec": profile.avg_duration_sec, "total_duration_sec": profile.total_duration_sec, "sample_rates": ", ".join(profile.sample_rate_distribution.keys()) or "unknown", "quality_info": f"corrupt={profile.n_corrupt}, silent={profile.n_silent}, clipped={profile.n_clipped}", "imbalance_ratio": round(ir, 2) if math.isfinite(ir) else 999.9, "task_context": tc, "meta_status": ms_final, "mem_influence": mem_influence, "mem_update": outcome})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _log(self, level: str, text: str) -> None:
        self.q.put({"kind": "log", "level": level, "text": text})

    def _step(self, text: str) -> None:
        self.q.put({"kind": "log", "level": "STEP", "text": text})

    def _sep(self) -> None:
        self.q.put({"kind": "log", "level": "SEP", "text": "-" * 52})
