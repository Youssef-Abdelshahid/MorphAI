from __future__ import annotations

import contextlib
import io
import math
import queue
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.shared.selector import select_best
from src.text.columns import resolve_columns
from src.text.config import TextConfig
from src.text.executor import evaluate_pipeline as evaluate_text_pipeline
from src.text.memory_manager import TextMemoryManager
from src.text.meta_learner import TextMetaLearner
from src.text.output_writer import save_processed_dataset as save_text_processed_dataset
from src.text.pipeline_generator import generate_pipelines as generate_text_pipelines
from src.text.profiler import LANGUAGE_FILTER_METHOD, compute_emoji_stats, profile_text_dataset, remove_empty_and_nonenglish_rows
from src.text.reporter import generate_report as generate_text_report, print_profile_summary as print_text_profile_summary, save_report as save_text_report
from src.text.validator import validate_text_run
from src.utils.ingestion.text import get_text_adapter


_FIELD_FLAGS = {
    "has_text": ("text",),
    "has_labels": ("label",),
    "has_multilabels": ("labels", "binary_label_columns"),
    "has_entity_annotations": ("entities",),
    "has_pos_tags": ("pos_tags",),
    "has_relation_labels": ("relation",),
    "has_similarity_pairs": ("text_a", "text_b"),
    "has_summaries": ("summary",),
    "has_qa_fields": ("context", "question", "answer"),
    "has_generation_references": ("completion",),
}


def _field_availability(col_overrides: Dict[str, Any], df_columns: List[str]) -> Dict[str, bool]:
    norm_columns = {str(c).strip().lower() for c in df_columns}
    flags: Dict[str, bool] = {}
    for flag, keys in _FIELD_FLAGS.items():
        present = False
        for key in keys:
            if key == "binary_label_columns":
                vals = col_overrides.get("binary_label_columns") or []
                if vals:
                    present = True
                    break
                continue
            value = col_overrides.get(key)
            if isinstance(value, str):
                v = value.strip()
                if v and v.lower() in norm_columns:
                    present = True
                    break
        flags[flag] = present
    text_cols = ("text", "tokens", "source_text", "context", "question", "prompt", "text_a", "text_b", "query", "document")
    has_any_text = any(
        isinstance(col_overrides.get(k), str)
        and col_overrides[k].strip()
        and col_overrides[k].strip().lower() in norm_columns
        for k in text_cols
    )
    flags["has_auxiliary_features"] = bool(col_overrides.get("auxiliary_feature_columns"))
    flags["is_plain_corpus"] = has_any_text and not any(flags[f] for f in ("has_labels", "has_multilabels", "has_entity_annotations", "has_pos_tags", "has_relation_labels", "has_similarity_pairs", "has_summaries", "has_qa_fields", "has_generation_references"))
    return flags


class TextAgentWorker:
    def __init__(
        self,
        q: queue.Queue,
        data_path: Path,
        metric: str,
        task_type: str = "classification_single",
        label_mode: str = "",
        domain: str = "",
        constraints: str = "",
        notes: str = "",
        language: str = "",
        text_source: str = "",
        text_length: str = "",
        col_overrides: Optional[Dict[str, str]] = None,
        auxiliary_feature_columns: Optional[List[str]] = None,
        multilabel_format: str = "single_column",
        binary_label_columns: Optional[List[str]] = None,
        input_format: str = "",
        input_format_key: str = "",
        record_path: str = "",
        metadata_path: str = "",
    ) -> None:
        self.q = q
        self.data_path = data_path
        self.metric = metric
        self.task_type = task_type
        self.label_mode = label_mode
        self.domain = domain
        self.constraints = constraints
        self.notes = notes
        self.language = language
        self.text_source = text_source
        self.text_length = text_length
        self.col_overrides = col_overrides or {}
        self.auxiliary_feature_columns = list(auxiliary_feature_columns or [])
        self.multilabel_format = multilabel_format or "single_column"
        self.binary_label_columns = list(binary_label_columns or [])
        self.input_format = input_format
        self.input_format_key = input_format_key or "csv_excel"
        self.record_path = record_path or ""
        self.metadata_path = metadata_path or ""

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            import traceback
            self._log("ERROR", f"Unexpected text processing error: {exc}")
            self._log("ERROR", traceback.format_exc())
            self.q.put({"kind": "fail", "text": str(exc)})

    def _execute(self) -> None:
        config = TextConfig(
            data_path=self.data_path,
            metric=self.metric,
            task_type=self.task_type,
            label_mode=self.label_mode,
            domain=self.domain,
            constraints=self.constraints,
            notes=self.notes,
            modality="Text",
            input_format=self.input_format,
            input_format_key=self.input_format_key,
            record_path=self.record_path,
            metadata_path=self.metadata_path,
            language=self.language,
            text_source=self.text_source,
            text_length=self.text_length,
            col_overrides=self.col_overrides or None,
            auxiliary_feature_columns=list(self.auxiliary_feature_columns),
            multilabel_format=self.multilabel_format,
            binary_label_columns=list(self.binary_label_columns),
        )
        tc = config.task_context()
        self._sep()
        self._log("INFO", f"Dataset : {self.data_path.name}")
        self._log("INFO", "Modality: Text  [English-only]")
        self._log("INFO", f"Metric  : {self.metric}")
        self._log("INFO", f"Task    : {tc.get('task_type', '')}")
        if self.input_format:
            self._log("INFO", f"Input fmt: {self.input_format}")
        self._sep()

        adapter = get_text_adapter(self.input_format_key)
        if adapter is None:
            msg = f"Unsupported text input format: '{self.input_format_key}'."
            self._log("ERROR", msg)
            self.q.put({"kind": "fail", "text": msg})
            return

        self._step("[1/9] Validating input package ...")
        validation = adapter.validate_input(self.data_path)
        if not validation.ok:
            for err in (validation.errors or [validation.message]):
                self._log("ERROR", f"  {err}")
            self.q.put({"kind": "fail", "text": validation.message or "Validation failed."})
            return
        self._log("OK", "  Input is valid.")

        tmp_dir: Optional[str] = None
        cleanup_dir: Optional[str] = None
        try:
            self._step("[1b/9] Parsing input format and building internal text dataset ...")
            parse_kwargs: Dict[str, Any] = {"task_type": self.task_type}
            if self.record_path:
                parse_kwargs["record_path"] = self.record_path
            if self.metadata_path:
                parse_kwargs["metadata_path"] = self.metadata_path
            if self.input_format_key == "txt_zip":
                tmp_dir = tempfile.mkdtemp(prefix="morphai_text_")
                cleanup_dir = tmp_dir
                parse_kwargs["work_dir"] = Path(tmp_dir) / "extracted"

            adapter_result = adapter.to_internal_dataset(self.data_path, **parse_kwargs)
            if not adapter_result.ok:
                for err in (adapter_result.errors or [adapter_result.message]):
                    self._log("ERROR", f"  {err}")
                self.q.put({"kind": "fail", "text": adapter_result.message or "Failed to parse text input format."})
                return
            internal_dataset = adapter_result.data["internal_dataset"]
            df = internal_dataset.dataframe
            if df is None or df.empty:
                self._log("ERROR", "  No usable text records were parsed from the input.")
                self.q.put({"kind": "fail", "text": "No usable text records were parsed from the input."})
                return
            original_count = int(len(df))
            self._log("OK", f"  Parsed {original_count:,} text record(s) via {self.input_format or self.input_format_key} adapter ({df.shape[1]} field(s)).")
            for w in (internal_dataset.warnings or [])[:5]:
                self._log("WARN", f"  parser: {w}")

            self._step("[2/9] Validating task-specific columns ...")
            val_errors = validate_text_run(config, df)
            if val_errors:
                for err in val_errors:
                    self._log("ERROR", f"  {err}")
                self.q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(val_errors)})
                return
            self._log("OK", "  Validation passed.")

            effective_overrides: Dict[str, object] = dict(config.col_overrides or {})
            if config.task_type == "classification_multi" and config.binary_label_columns:
                effective_overrides["binary_label_columns"] = list(config.binary_label_columns)

            self._step("[2c/9] Filtering empty and non-English rows ...")
            cols = resolve_columns(df, config.task_type, col_overrides=effective_overrides)
            df, filter_counts = remove_empty_and_nonenglish_rows(df, cols, config.task_type)
            n_removed_invalid = filter_counts.get("removed_empty_or_invalid", 0)
            n_removed_nonenglish = filter_counts.get("removed_non_english", 0)
            n_removed_uncertain = filter_counts.get("removed_language_uncertain", 0)
            n_removed_noisy = filter_counts.get("removed_too_noisy", 0)
            self._log("INFO", f"  Language filter: {LANGUAGE_FILTER_METHOD}")
            if n_removed_invalid:
                self._log("INFO", f"  Removed {n_removed_invalid:,} empty or invalid row(s).")
            if n_removed_nonenglish:
                self._log("INFO", f"  Removed {n_removed_nonenglish:,} non-English row(s).")
            if n_removed_uncertain:
                self._log("INFO", f"  Removed {n_removed_uncertain:,} language-uncertain row(s).")
            if n_removed_noisy:
                self._log("INFO", f"  Removed {n_removed_noisy:,} noise-dominated row(s).")
            if df.empty:
                msg = "All rows were removed during filtering. The dataset does not contain valid English text for this task."
                self._log("ERROR", f"  {msg}")
                self.q.put({"kind": "fail", "text": msg})
                return
            self._log("OK", f"  {len(df):,} usable English row(s) remain.")
            self._step("[3/9] Profiling text dataset ...")
            profile = profile_text_dataset(
                df,
                config.task_type,
                col_overrides=effective_overrides,
                auxiliary_feature_columns=list(config.auxiliary_feature_columns or []),
                original_row_count=original_count,
                removed_empty_or_invalid_count=n_removed_invalid,
                removed_non_english_count=n_removed_nonenglish,
                removed_language_uncertain_count=n_removed_uncertain,
                removed_too_noisy_count=n_removed_noisy,
                language_filter_method=LANGUAGE_FILTER_METHOD,
            )
            profile.input_format = self.input_format_key
            profile.structure_profile = dict(internal_dataset.structure_profile or {})
            profile.parsing_summary = dict(internal_dataset.parsing_summary or {})
            profile.parser_warnings = list(internal_dataset.warnings or [])
            profile.field_availability = _field_availability(effective_overrides, list(df.columns))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_text_profile_summary(profile)
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self._log("INFO", line)
            self._log("OK", f"  {profile.n_samples:,} structured text sample(s).")
            if profile.has_tabular_features:
                self._log("INFO", f"  Detected {len(profile.auxiliary_numeric_columns)} numeric and {len(profile.auxiliary_categorical_columns)} categorical auxiliary feature(s).")
            self._step("[4/9] Loading meta-learner ...")
            meta = TextMetaLearner()
            meta.load()
            ms = meta.status_summary()
            self._log("INFO", f"  Meta-learner learning: {ms['n_train']}/{ms['min_to_use']} samples before activation." if not meta.is_mature else f"  Meta-learner active: {ms['n_train']} samples, weight={ms['weight']:.2f}")
            self._step("[5/9] Checking memory ...")
            memory = TextMemoryManager()
            memory.load()
            good_cases, bad_cases = memory.find_good_and_bad(profile, config.metric, task_type=tc.get("task_type", ""))
            self._log("OK" if good_cases or bad_cases else "INFO", f"  {len(good_cases)} good + {len(bad_cases)} poor similar run(s) found." if good_cases or bad_cases else "  No similar past text runs. Using heuristics only.")
            profile_summary = {
                "n_samples": profile.n_samples,
                "n_classes": profile.n_classes,
                "avg_token_length": profile.avg_token_length,
                "token_length_std": profile.token_length_std,
                "vocabulary_size_estimate": profile.vocabulary_size_estimate,
                "duplicate_ratio": profile.duplicate_text_count / max(profile.n_samples, 1),
                "empty_text_ratio": profile.n_empty_texts / max(profile.n_samples, 1),
                "noise_ratio": profile.noise_ratio,
                "imbalance_ratio": profile.imbalance_ratio,
                "annotation_invalid_ratio": profile.annotation_validity.get("invalid_count", 0) / max(profile.n_samples, 1),
                "source_target_length_ratio": profile.source_target_length_ratio,
                "has_tabular_features": profile.has_tabular_features,
                "num_extra_numeric_cols": len(profile.auxiliary_numeric_columns),
                "num_extra_categorical_cols": len(profile.auxiliary_categorical_columns),
                "extra_feature_missing_ratio": profile.extra_feature_missing_ratio,
                "original_row_count": profile.original_row_count,
                "removed_empty_or_invalid_count": profile.removed_empty_or_invalid_count,
                "removed_non_english_count": profile.removed_non_english_count,
                "final_row_count": profile.n_samples,
                "input_format": self.input_format_key,
                **{k: bool(v) for k, v in (profile.field_availability or {}).items()},
            }
            self._step("[6/9] Generating candidate pipelines ...")
            pipelines, mem_msgs = generate_text_pipelines(profile, good_cases, bad_cases, meta_learner=meta, task_context=tc, profile_summary=profile_summary)
            self._log("OK", f"  {len(pipelines)} candidate(s) generated.")
            for msg in mem_msgs:
                self._log("INFO", f"  {msg}")
            mem_influence = {"good_injections": len(good_cases), "bad_avoidances": len(bad_cases), "meta_learner_weight": ms["weight"]}
            self._step("[7/9] Evaluating pipelines ...")
            results = []
            successful = []
            for idx, spec in enumerate(pipelines, 1):
                short = spec.name() if len(spec.name()) <= 70 else spec.name()[:67] + "..."
                self._log("INFO", f"  [{idx:2d}/{len(pipelines)}]  {short}")
                result = evaluate_text_pipeline(spec, df.copy(), profile, config.task_type, config.metric)
                results.append(result)
                if result.get("success", True):
                    successful.append(result)
                    selected_metric = result.get("selected_metric", self.metric)
                    raw = result.get("raw_metrics", result["metrics"]).get(selected_metric, 0.0)
                    score = result.get("normalized_score", result.get("final_score", 0.0))
                    self._log("METRIC", f"           {selected_metric}={raw:.4f}  | normalized={score:.4f}  [{result.get('n_models', 1)} evaluator(s) x {result['elapsed_sec']:.2f}s]")
                else:
                    self._log("WARN", f"           [FAILED - score=0.0000] {result.get('reason', 'invalid evaluation')}")
            if not successful:
                self._log("ERROR", "All pipelines failed.")
                self.q.put({"kind": "fail", "text": "All candidate text pipelines failed to produce a valid evaluation."})
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
            self._step("[9/9] Saving cleaned text dataset ...")
            cleaned_path, cleaned_shape = save_text_processed_dataset(best["spec"], df, profile, config, internal_dataset=internal_dataset)
            if profile.primary_text_columns and profile.primary_text_columns[0] in df.columns:
                emoji_stats = compute_emoji_stats(df[profile.primary_text_columns[0]].fillna(""), best["spec"].emoji_handling)
                profile.emoji_strategy = best["spec"].emoji_handling
                profile.emoji_translated_count = int(emoji_stats["emoji_translated_count"])
                profile.emoji_removed_count = int(emoji_stats["emoji_removed_count"])
                profile.removed_excessive_emoji_count = int(emoji_stats["removed_excessive_emoji_count"])
            self._log("OK", f"  {cleaned_path}")
            self._step("[9/9] Saving report, updating memory and meta-learner ...")
            ms_final = meta.status_summary()
            report = generate_text_report(profile, results, best, config, meta_status=ms_final, mem_influence=mem_influence)
            report_path = save_text_report(report)
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
                "kind": "done", "modality": "Text", "report": report, "report_path": report_path,
                "cleaned_path": cleaned_path, "cleaned_shape": cleaned_shape,
                "best_name": best["spec"].name(), "best_score": bs, "best_score_std": sd,
                "metric": selected_metric, "metrics": best["metrics"],
                "raw_metrics": best.get("raw_metrics", best["metrics"]),
                "metrics_std": best.get("metrics_std", {}),
                "normalized_metrics": best.get("normalized_metrics", {}),
                "per_model": best.get("per_model_metrics", {}),
                "evaluation_mode": best.get("evaluation_mode", ""),
                "evaluation_summary": best.get("evaluation_summary", ""),
                "n_splits": best.get("n_splits", "?"),
                "n_models": best.get("n_models", 1),
                "n_pipelines": len(results),
                "n_samples": profile.n_samples,
                "n_classes": profile.n_classes,
                "avg_token_length": profile.avg_token_length,
                "vocabulary_size_estimate": profile.vocabulary_size_estimate,
                "quality_info": (
                    f"empty={profile.n_empty_texts}, duplicate={profile.duplicate_text_count}, "
                    f"noise={profile.noise_ratio:.2%}, removed_invalid={n_removed_invalid}, "
                    f"removed_non_english={n_removed_nonenglish}, "
                    f"removed_uncertain={n_removed_uncertain}, removed_noisy={n_removed_noisy}"
                ),
                "imbalance_ratio": round(ir, 2) if math.isfinite(ir) else 999.9,
                "task_context": tc,
                "meta_status": ms_final,
                "mem_influence": mem_influence,
                "mem_update": outcome,
                "original_row_count": original_count,
                "removed_empty_or_invalid_count": n_removed_invalid,
                "removed_non_english_count": n_removed_nonenglish,
                "fusion_used": bool(best.get("evaluator_details", {}).get("fusion_used", False)),
                "input_format": self.input_format,
                "input_format_key": self.input_format_key,
            })
        finally:
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _log(self, level: str, text: str) -> None:
        self.q.put({"kind": "log", "level": level, "text": text})

    def _step(self, text: str) -> None:
        self.q.put({"kind": "log", "level": "STEP", "text": text})

    def _sep(self) -> None:
        self.q.put({"kind": "log", "level": "SEP", "text": "-" * 52})
