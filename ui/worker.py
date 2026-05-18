from __future__ import annotations

import contextlib
import io
import math
import queue
import shutil
import tempfile
import zipfile
from pathlib import Path

from src.tabular.config import Config
from src.tabular.executor import evaluate_pipeline
from src.tabular.memory_manager import MemoryManager
from src.tabular.meta_learner import MetaLearner
from src.tabular.output_writer import save_cleaned_dataset
from src.tabular.pipeline_generator import generate_pipelines
from src.tabular.profiler import profile_dataset
from src.tabular.reporter import generate_report, print_profile_summary, save_report
from src.tabular.validator import validate_csv_run, validate_input_file
from src.utils.shared.selector import select_best
from src.utils.ingestion.tabular import get_tabular_adapter

from src.image.config import ImageConfig
from src.image.executor import evaluate_pipeline as evaluate_image_pipeline
from src.image.memory_manager import ImageMemoryManager
from src.image.meta_learner import ImageMetaLearner
from src.image.output_writer import save_processed_dataset
from src.image.pipeline_generator import generate_pipelines as generate_image_pipelines
from src.image.profiler import profile_image_dataset
from src.image.reporter import (
    generate_report as generate_image_report,
    print_profile_summary as print_image_profile_summary,
    save_report as save_image_report,
)
from src.image.validator import validate_image_run, validate_image_zip, validate_internal_dataset
from src.utils.ingestion.image import (
    get_image_adapter,
    materialize_for_pipeline,
    has_bboxes as ds_has_bboxes,
    has_masks as ds_has_masks,
    has_keypoints as ds_has_keypoints,
    has_text_labels as ds_has_text_labels,
    has_depth_targets as ds_has_depth_targets,
    has_class_label as ds_has_class_labels,
)

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _find_dataset_root(extracted_dir: Path) -> Path:
    def _has_class_structure(d: Path) -> bool:
        count = 0
        try:
            for sub in d.iterdir():
                if not sub.is_dir() or sub.name.startswith(".") or sub.name == "__MACOSX":
                    continue
                imgs = [
                    f for f in sub.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMG_EXTENSIONS
                ]
                if imgs:
                    count += 1
                    if count >= 2:
                        return True
        except Exception:
            pass
        return False

    if _has_class_structure(extracted_dir):
        return extracted_dir

    try:
        valid_subdirs = [
            d for d in extracted_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != "__MACOSX"
        ]
    except Exception:
        return extracted_dir

    if len(valid_subdirs) == 1 and _has_class_structure(valid_subdirs[0]):
        return valid_subdirs[0]

    return extracted_dir


class AgentWorker:

    def __init__(
        self,
        q: queue.Queue,
        csv_path: Path,
        target: str,
        metric: str,
        task_type:         str,
        domain:            str = "",
        constraints:       str = "",
        notes:             str = "",
        modality:          str = "Tabular",
        input_format:      str = "",
        input_format_key:  str = "",
        record_path:       str = "",
        fe_budget:         str = "",
        data_quality:      str = "",
    ) -> None:
        self.q        = q
        self.csv_path = csv_path
        self.target   = target
        self.metric   = metric
        self.task_type         = task_type
        self.domain            = domain
        self.constraints       = constraints
        self.notes             = notes
        self.modality          = modality
        self.input_format      = input_format
        self.input_format_key  = input_format_key
        self.record_path       = record_path
        self.fe_budget         = fe_budget
        self.data_quality      = data_quality

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            import traceback
            self._log("ERROR", f"Unexpected error: {exc}")
            self._log("ERROR", traceback.format_exc())
            self.q.put({"kind": "fail", "text": str(exc)})

    def _execute(self) -> None:
        q        = self.q
        target   = self.target
        metric   = self.metric
        csv_path = self.csv_path

        config = Config(
            data_path=csv_path,
            target=target,
            metric=metric,
            task_type=self.task_type,
            domain=self.domain,
            constraints=self.constraints,
            notes=self.notes,
            modality=self.modality,
            input_format=self.input_format,
            input_format_key=self.input_format_key,
            record_path=self.record_path,
            fe_budget=self.fe_budget,
            data_quality=self.data_quality,
        )
        tc = config.task_context()

        self._sep()
        self._log("INFO", f"Dataset : {csv_path.name}")
        self._log("INFO", f"Target  : {target}")
        self._log("INFO", f"Metric  : {metric}")
        if tc.get("task_type"):
            self._log("INFO", f"Task    : {tc['task_type']}  ({tc.get('supervision', '')})")
        if tc.get("domain"):
            self._log("INFO", f"Domain  : {tc['domain']}")
        if tc.get("fe_budget"):
            self._log("INFO", f"FE budget   : {tc['fe_budget']}")
        if tc.get("data_quality"):
            self._log("INFO", f"Data qual.  : {tc['data_quality']}")
        active = tc.get("active_constraints", [])
        if active:
            self._log("INFO", f"Constraints : {', '.join(active)}")
        self._sep()

        self._step("[1/9] Loading dataset ...")
        format_key = self.input_format_key or "csv_excel"

        ext_errors = validate_input_file(config)
        if ext_errors:
            for err in ext_errors:
                self._log("ERROR", f"  {err}")
            q.put({"kind": "fail", "text": "; ".join(ext_errors)})
            return

        adapter = get_tabular_adapter(format_key)
        if adapter is None:
            self._log("ERROR", f"Unsupported tabular input format '{format_key}'.")
            q.put({"kind": "fail", "text": f"Unsupported tabular input format '{format_key}'."})
            return

        self._log("INFO", f"  Format : {self.input_format or format_key}")
        ingest_result = adapter.to_internal_dataset(csv_path, record_path=self.record_path)
        if not ingest_result.ok:
            for err in (ingest_result.errors or [ingest_result.message]):
                self._log("ERROR", f"  {err}")
            q.put({"kind": "fail", "text": ingest_result.message or "Failed to ingest input file."})
            return

        df = ingest_result.data["dataframe"]
        structure_profile = ingest_result.data.get("structure_profile", {})
        parsing_summary = ingest_result.data.get("parsing_summary", {})
        parser_warnings = ingest_result.data.get("warnings", []) or []

        if df is None or df.empty:
            q.put({"kind": "fail", "text": "Parsed dataset is empty."})
            return

        if structure_profile.get("detected_record_path"):
            self._log("INFO", f"  Record path  : {structure_profile['detected_record_path']}")
        if structure_profile.get("flattened_column_count"):
            self._log("INFO", f"  Flat columns : {structure_profile['flattened_column_count']}")
        if parser_warnings:
            for w in parser_warnings[:5]:
                self._log("WARN", f"  parser: {w}")

        if target and target not in df.columns and config.supervision == "supervised":
            cols = list(df.columns)
            similar = [c for c in cols if target.lower() in c.lower() or c.lower() in target.lower()]
            self._log("ERROR", f"Target column '{target}' not found after parsing/flattening.")
            if similar:
                self._log("ERROR", f"  Did you mean: {similar[:5]}")
            self._log("ERROR", f"  Available: {cols[:30]}{'...' if len(cols) > 30 else ''}")
            q.put({"kind": "fail", "text": f"Target column '{target}' not found in parsed data."})
            return
        n_before = len(df)
        if target in df.columns:
            df = df.dropna(subset=[target]).reset_index(drop=True)
        dropped  = n_before - len(df)
        msg = f"  {len(df):,} rows  x  {df.shape[1]} columns"
        if dropped:
            msg += f"  (dropped {dropped} rows with missing target)"
        self._log("OK", msg)

        self._step("[1b/9] Validating inputs ...")
        val_errors = validate_csv_run(config, df)
        if val_errors:
            for err in val_errors:
                self._log("ERROR", f"  {err}")
            q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(val_errors)})
            return
        self._log("OK", "  Validation passed.")

        self._step("[2/9] Profiling dataset ...")
        profile = profile_dataset(df, target)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_profile_summary(profile)
        for line in buf.getvalue().splitlines():
            if line.strip():
                self._log("INFO", line)
        flags = []
        if profile.has_outliers:          flags.append("outliers")
        if profile.has_high_skew:         flags.append("high-skew")
        if profile.has_high_kurtosis:     flags.append("heavy-tails")
        if profile.has_sparse_features:   flags.append("sparse")
        if profile.has_multicollinearity: flags.append("multicollinearity")
        if profile.has_high_missing_cols: flags.append("cols>50%missing")
        if flags:
            self._log("WARN", f"  Active flags: {', '.join(flags)}")

        self._step("[3/9] Loading meta-learner ...")
        meta = MetaLearner()
        meta.load()
        ms = meta.status_summary()
        if meta.is_mature:
            self._log("OK", f"  Meta-learner active: {ms['n_train']} samples, "
                             f"weight={ms['weight']:.2f}")
        else:
            self._log("INFO", f"  Meta-learner learning: {ms['n_train']}/{ms['min_to_use']} "
                              f"samples before activation.")

        self._step("[4/9] Checking memory ...")
        memory = MemoryManager()
        memory.load()
        good_cases, bad_cases = memory.find_good_and_bad(
            profile, metric, task_type=tc.get("task_type", "")
        )
        if good_cases or bad_cases:
            self._log("OK", f"  {len(good_cases)} good + {len(bad_cases)} poor similar run(s) found.")
        else:
            self._log("INFO", "  No similar past runs.  Using heuristics only.")

        profile_summary_dict = {
            "n_rows":               profile.n_rows,
            "n_cols":               profile.n_cols,
            "missing_ratio":        profile.total_missing_ratio,
            "imbalance_ratio":      profile.imbalance_ratio,
            "num_col_ratio":        len(profile.num_cols) / max(profile.n_cols, 1),
            "cat_col_ratio":        len(profile.cat_cols) / max(profile.n_cols, 1),
            "has_outliers":         profile.has_outliers,
            "has_high_skew":        profile.has_high_skew,
            "is_imbalanced":        profile.is_imbalanced,
            "is_highly_imbalanced": profile.is_highly_imbalanced,
        }

        self._step("[5/9] Generating candidate pipelines ...")
        pipelines, mem_msgs = generate_pipelines(
            profile, good_cases, bad_cases,
            meta_learner=meta,
            task_context=tc,
            profile_summary=profile_summary_dict,
        )
        self._log("OK", f"  {len(pipelines)} candidate(s) generated.")
        for m_msg in mem_msgs:
            self._log("INFO", f"  {m_msg}")

        mem_influence = {
            "good_injections":     len(good_cases),
            "bad_avoidances":      len(bad_cases),
            "meta_learner_weight": ms["weight"],
        }

        self._step("[6/9] Evaluating pipelines ...")
        results = []
        successful_results = []
        for idx, spec in enumerate(pipelines, 1):
            label = spec.name()
            short = label if len(label) <= 52 else label[:49] + "..."
            self._log("INFO", f"  [{idx:2d}/{len(pipelines)}]  {short}")
            result = evaluate_pipeline(spec, df.copy(), profile, config.task_type, config.metric)
            if result:
                results.append(result)
                if result.get("success", True):
                    successful_results.append(result)
                    selected_metric = result.get("selected_metric", metric)
                    m_val = result.get("raw_metrics", result["metrics"]).get(selected_metric, 0.0)
                    s_val = result.get("normalized_score", result.get("final_score", 0.0))
                    n_m   = result.get("n_models", 1)
                    self._log(
                        "METRIC",
                        f"           {selected_metric}={m_val:.4f}  "
                        f"| normalized={s_val:.4f}  "
                        f"[{result['n_splits']}folds x {n_m}models "
                        f"x {result['elapsed_sec']:.2f}s]"
                    )
                    pmt = result.get("per_model_metrics", {})
                    if pmt:
                        row = "  ".join(f"{mn}={mv.get(selected_metric, 0):.4f}" for mn, mv in pmt.items())
                        self._log("MUTED", f"           {row}")
                else:
                    self._log("WARN", f"           [FAILED - score=0.0000] {result.get('reason', 'invalid evaluation')}")

        if not successful_results:
            self._log("ERROR", "All pipelines failed.")
            q.put({"kind": "fail", "text": "All candidate pipelines failed to produce a valid evaluation."})
            return

        self._step("[7/9] Selecting best pipeline ...")
        best = select_best(successful_results, metric)
        bs   = best.get("normalized_score", best.get("final_score", 0.0))
        sd   = best.get("final_score_std", 0.0)
        selected_metric = best.get("selected_metric", metric)
        selected_raw = best.get("raw_metrics", best["metrics"]).get(selected_metric, 0.0)
        self._log("BEST", f"  > {best['spec'].name()}")
        self._log("BEST", f"  > {selected_metric} = {selected_raw:.4f}")
        self._log("BEST", f"  > normalized score = {bs:.4f}  (+/- {sd:.4f})")

        self._step("[8/9] Saving cleaned dataset ...")
        cleaned_path, cleaned_shape = save_cleaned_dataset(
            best["spec"], df.copy(), profile, config,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            parser_warnings=parser_warnings,
        )
        self._log("OK", f"  {cleaned_path}")
        self._log("INFO",
                  f"  Shape: {cleaned_shape[0]:,} rows x {cleaned_shape[1]} cols")

        self._step("[9/9] Saving report, updating memory and meta-learner ...")
        ms_final    = meta.status_summary()
        report      = generate_report(
            profile, results, best, config,
            meta_status=ms_final,
            mem_influence=mem_influence,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            parser_warnings=parser_warnings,
        )
        report_path  = save_report(report)

        outcome = memory.add_run(
            profile, config, results, best,
            meta_status=ms_final,
            mem_influence=mem_influence,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
        )
        memory.save()
        self._log("OK", f"  Report : {report_path}")
        self._log("OK", f"  Memory : {memory.n_runs} total run(s)  [{outcome}]")

        n_samples = meta.train_from_memory(memory.all_runs())
        if n_samples > 0:
            meta.save()
            ms_final = meta.status_summary()
            self._log("OK",
                      f"  Meta-learner retrained: {n_samples} pipeline samples  "
                      f"(weight={ms_final['weight']:.2f})")
        else:
            self._log("INFO",
                      f"  Meta-learner: not enough data yet "
                      f"({memory.n_runs} run(s) in memory)")

        self._sep()
        self._log("OK", "Agent run complete.")
        self._sep()

        ir = profile.imbalance_ratio
        ir_display = round(ir, 2) if math.isfinite(ir) else 999.9

        q.put({
            "kind":           "done",
            "report":         report,
            "report_path":    report_path,
            "cleaned_path":   cleaned_path,
            "cleaned_shape":  cleaned_shape,
            "best_name":      best["spec"].name(),
            "best_score":     bs,
            "best_score_std": sd,
            "metric":         selected_metric,
            "metrics":        best["metrics"],
            "raw_metrics":    best.get("raw_metrics", best["metrics"]),
            "metrics_std":    best.get("metrics_std", {}),
            "normalized_metrics": best.get("normalized_metrics", {}),
            "normalized_metrics_std": best.get("normalized_metrics_std", {}),
            "per_model":      best.get("per_model_metrics", {}),
            "evaluation_mode": best.get("evaluation_mode", ""),
            "evaluation_summary": best.get("evaluation_summary", ""),
            "n_splits":       best.get("n_splits", "?"),
            "n_models":       best.get("n_models", 1),
            "n_pipelines":    len(results),
            "profile_rows":   profile.n_rows,
            "profile_cols":   profile.n_cols,
            "num_cols":       len(profile.num_cols),
            "cat_cols":       len(profile.cat_cols),
            "n_classes":      profile.n_classes,
            "imbalance_ratio": ir_display,
            "task_context":   tc,
            "meta_status":    ms_final,
            "mem_influence":  mem_influence,
            "mem_update":     outcome,
        })

    def _log(self, level: str, text: str) -> None:
        self.q.put({"kind": "log", "level": level, "text": text})

    def _step(self, text: str) -> None:
        self.q.put({"kind": "log", "level": "STEP", "text": text})

    def _sep(self) -> None:
        self.q.put({"kind": "log", "level": "SEP", "text": "─" * 52})


class ImageAgentWorker:

    def __init__(
        self,
        q: queue.Queue,
        zip_path: Path,
        metric: str,
        task_type:    str = "classification",
        label_mode:   str = "",
        domain:       str = "",
        constraints:  str = "",
        notes:        str = "",
        image_format: str = "",
        color_space:  str = "",
        input_format: str = "",
        input_format_key: str = "",
        annotation_path: str = "",
        image_dir: str = "",
        annotation_dir: str = "",
        label_dir: str = "",
        class_config: str = "",
        split_selection: str = "",
    ) -> None:
        self.q            = q
        self.zip_path     = zip_path
        self.metric       = metric
        self.task_type    = task_type
        self.label_mode   = label_mode
        self.domain       = domain
        self.constraints  = constraints
        self.notes        = notes
        self.image_format = image_format
        self.color_space  = color_space
        self.input_format = input_format
        self.input_format_key = input_format_key or "zip_folder"
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.label_dir = label_dir
        self.class_config = class_config
        self.split_selection = split_selection

    def run(self) -> None:
        try:
            self._execute()
        except Exception as exc:
            import traceback
            self._log("ERROR", f"Unexpected error: {exc}")
            self._log("ERROR", traceback.format_exc())
            self.q.put({"kind": "fail", "text": str(exc)})

    def _execute(self) -> None:
        q        = self.q
        metric   = self.metric
        zip_path = self.zip_path

        config = ImageConfig(
            data_path=zip_path,
            metric=metric,
            task_type=self.task_type,
            label_mode=self.label_mode,
            domain=self.domain,
            constraints=self.constraints,
            notes=self.notes,
            modality="Image",
            input_format=self.input_format,
            image_format=self.image_format,
            color_space=self.color_space,
        )
        tc = config.task_context()

        self._sep()
        self._log("INFO", f"Dataset : {zip_path.name}")
        self._log("INFO", f"Modality: Image")
        self._log("INFO", f"Metric  : {metric}")
        if self.input_format:
            self._log("INFO", f"Input fmt: {self.input_format}")
        if tc.get("task_type"):
            self._log("INFO", f"Task    : {tc['task_type']}")
        if tc.get("domain"):
            self._log("INFO", f"Domain  : {tc['domain']}")
        if tc.get("image_format"):
            self._log("INFO", f"Format  : {tc['image_format']}")
        if tc.get("color_space"):
            self._log("INFO", f"Color   : {tc['color_space']}")
        active = tc.get("active_constraints", [])
        if active:
            self._log("INFO", f"Constraints : {', '.join(active)}")
        self._sep()

        self._step("[1/9] Validating zip archive ...")
        zip_errors = validate_image_zip(zip_path)
        if zip_errors:
            for err in zip_errors:
                self._log("ERROR", f"  {err}")
            q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(zip_errors)})
            return
        self._log("OK", "  Archive is valid.")

        adapter = get_image_adapter(self.input_format_key)
        if adapter is None:
            self._log("ERROR", f"Unsupported image input format: '{self.input_format_key}'.")
            q.put({"kind": "fail", "text": f"Unsupported image input format: '{self.input_format_key}'."})
            return

        tmp_dir = tempfile.mkdtemp(prefix="morphai_img_")
        try:
            self._step("[1b/9] Parsing input format and building internal dataset ...")
            parse_kwargs = {}
            if self.input_format_key == "coco":
                if self.annotation_path:
                    parse_kwargs["annotation_path"] = self.annotation_path
                if self.image_dir:
                    parse_kwargs["image_dir"] = self.image_dir
            elif self.input_format_key == "pascal_voc":
                if self.image_dir:
                    parse_kwargs["image_dir"] = self.image_dir
                if self.annotation_dir:
                    parse_kwargs["annotation_dir"] = self.annotation_dir
            elif self.input_format_key == "yolo":
                if self.image_dir:
                    parse_kwargs["image_dir"] = self.image_dir
                if self.label_dir:
                    parse_kwargs["label_dir"] = self.label_dir
                if self.class_config:
                    parse_kwargs["class_config"] = self.class_config
            if self.split_selection:
                parse_kwargs["split"] = self.split_selection

            extract_dir = Path(tmp_dir) / "extracted"
            adapter_result = adapter.to_internal_dataset(zip_path, work_dir=extract_dir, **parse_kwargs)
            if not adapter_result.ok:
                for err in (adapter_result.errors or [adapter_result.message]):
                    self._log("ERROR", f"  {err}")
                q.put({"kind": "fail", "text": adapter_result.message or "Failed to parse image input format."})
                return
            internal_dataset = adapter_result.data["internal_dataset"]
            n_parsed = len(internal_dataset.samples)
            self._log("OK", f"  Parsed {n_parsed} image(s) via {self.input_format or self.input_format_key} adapter.")
            for w in (internal_dataset.warnings or [])[:5]:
                self._log("WARN", f"  parser: {w}")

            self._step("[2/9] Validating internal image dataset against task ...")
            val_errors = validate_internal_dataset(config, internal_dataset)
            if val_errors:
                for err in val_errors:
                    self._log("ERROR", f"  {err}")
                q.put({"kind": "fail", "text": "Validation failed: " + "; ".join(val_errors)})
                return
            self._log("OK", "  Validation passed.")

            self._step("[2b/9] Materializing internal dataset for pipeline ...")
            materialized = Path(tmp_dir) / "materialized"
            dataset_root = materialize_for_pipeline(internal_dataset, materialized, self.task_type)
            self._log("OK", "  Internal dataset prepared for pipeline.")

            self._step("[3/9] Profiling image dataset ...")
            profile = profile_image_dataset(dataset_root)
            profile.input_format = self.input_format_key
            profile.parsing_summary = dict(internal_dataset.parsing_summary or {})
            profile.annotation_profile = dict(internal_dataset.annotation_profile or {})
            profile.structure_profile = dict(internal_dataset.structure_profile or {})
            profile.parser_warnings = list(internal_dataset.warnings or [])
            profile.class_mapping = dict(internal_dataset.class_mapping or {})
            profile.has_bboxes = ds_has_bboxes(internal_dataset.samples)
            profile.has_masks = ds_has_masks(internal_dataset.samples)
            profile.has_keypoints = ds_has_keypoints(internal_dataset.samples)
            profile.has_text_labels = ds_has_text_labels(internal_dataset.samples)
            profile.has_depth_targets = ds_has_depth_targets(internal_dataset.samples)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print_image_profile_summary(profile)
            for line in buf.getvalue().splitlines():
                if line.strip():
                    self._log("INFO", line)
            flags = []
            if profile.has_varied_sizes:           flags.append("varied-sizes")
            if profile.has_low_contrast:           flags.append("low-contrast")
            if profile.has_high_contrast_variance: flags.append("contrast-variance")
            if profile.has_varied_brightness:      flags.append("varied-brightness")
            if profile.has_mostly_grayscale:       flags.append("mostly-grayscale")
            if profile.has_small_images:           flags.append("small-images")
            if profile.has_large_images:           flags.append("large-images")
            if profile.has_corrupt_images:         flags.append(f"corrupt({profile.n_corrupt})")
            if flags:
                self._log("WARN", f"  Active flags: {', '.join(flags)}")
            self._log("OK", f"  {profile.n_images:,} images across {profile.n_classes} classes.")

            self._step("[4/9] Loading meta-learner ...")
            meta = ImageMetaLearner()
            meta.load()
            ms = meta.status_summary()
            if meta.is_mature:
                self._log("OK", f"  Meta-learner active: {ms['n_train']} samples, "
                                 f"weight={ms['weight']:.2f}")
            else:
                self._log("INFO", f"  Meta-learner learning: {ms['n_train']}/{ms['min_to_use']} "
                                  f"samples before activation.")

            self._step("[5/9] Checking memory ...")
            memory = ImageMemoryManager()
            memory.load()
            good_cases, bad_cases = memory.find_good_and_bad(
                profile, metric, task_type=tc.get("task_type", "")
            )
            if good_cases or bad_cases:
                self._log("OK", f"  {len(good_cases)} good + {len(bad_cases)} poor similar run(s) found.")
            else:
                self._log("INFO", "  No similar past runs.  Using heuristics only.")

            profile_summary_dict = {
                "n_images":              profile.n_images,
                "n_classes":             profile.n_classes,
                "imbalance_ratio":       profile.imbalance_ratio,
                "avg_brightness":        profile.avg_brightness,
                "brightness_std":        profile.brightness_std,
                "avg_contrast":          profile.avg_contrast,
                "contrast_std":          profile.contrast_std,
                "grayscale_ratio":       profile.grayscale_ratio,
                "has_varied_sizes":      profile.has_varied_sizes,
                "has_low_contrast":      profile.has_low_contrast,
                "is_imbalanced":         profile.is_imbalanced,
                "is_highly_imbalanced":  profile.is_highly_imbalanced,
            }

            self._step("[6/9] Generating candidate pipelines ...")
            pipelines, mem_msgs = generate_image_pipelines(
                profile, good_cases, bad_cases,
                meta_learner=meta,
                task_context=tc,
                profile_summary=profile_summary_dict,
            )
            self._log("OK", f"  {len(pipelines)} candidate(s) generated.")
            for m_msg in mem_msgs:
                self._log("INFO", f"  {m_msg}")

            mem_influence = {
                "good_injections":     len(good_cases),
                "bad_avoidances":      len(bad_cases),
                "meta_learner_weight": ms["weight"],
            }

            self._step("[7/9] Evaluating pipelines ...")
            results = []
            successful_results = []
            for idx, spec in enumerate(pipelines, 1):
                label = spec.name()
                short = label if len(label) <= 52 else label[:49] + "..."
                self._log("INFO", f"  [{idx:2d}/{len(pipelines)}]  {short}")
                result = evaluate_image_pipeline(spec, profile, config.task_type, config.metric)
                if result:
                    results.append(result)
                    if result.get("success", True):
                        successful_results.append(result)
                        selected_metric = result.get("selected_metric", metric)
                        m_val = result.get("raw_metrics", result["metrics"]).get(selected_metric, 0.0)
                        s_val = result.get("normalized_score", result.get("final_score", 0.0))
                        n_m = result.get("n_models", 1)
                        self._log(
                            "METRIC",
                            f"           {selected_metric}={m_val:.4f}  "
                            f"| normalized={s_val:.4f}  "
                            f"[{result['n_splits']}folds x {n_m}models x {result['elapsed_sec']:.2f}s]"
                        )
                        pmt = result.get("per_model_metrics", {})
                        if pmt:
                            row = "  ".join(f"{mn}={mv.get(selected_metric, 0):.4f}" for mn, mv in pmt.items())
                            self._log("MUTED", f"           {row}")
                    else:
                        self._log("WARN", f"           [FAILED - score=0.0000] {result.get('reason', 'invalid evaluation')}")

            if not successful_results:
                self._log("ERROR", "All pipelines failed.")
                q.put({"kind": "fail", "text": "All candidate image pipelines failed to produce a valid evaluation."})
                return

            self._step("[8/9] Selecting best pipeline ...")
            best = select_best(successful_results, metric)
            bs = best.get("normalized_score", best.get("final_score", 0.0))
            selected_metric = best.get("selected_metric", metric)
            sd = best.get("final_score_std", best.get("metrics_std", {}).get(f"{selected_metric}_std", 0.0))
            self._log("BEST", f"  > {best['spec'].name()}")
            self._log("BEST", f"  > {selected_metric} = {best.get('raw_metrics', best['metrics']).get(selected_metric, 0.0):.4f}")
            self._log("BEST", f"  > normalized score = {bs:.4f}  (+/- {sd:.4f})")

            self._step("[9/9] Saving processed zip ...")
            cleaned_path, cleaned_shape = save_processed_dataset(
                best["spec"], profile, config,
                internal_dataset=internal_dataset,
            )
            self._log("OK", f"  {cleaned_path}")
            self._log("INFO",
                      f"  Saved: {cleaned_shape[0]:,} images across {cleaned_shape[1]} class(es)")

            self._step("[9/9] Saving report, updating memory and meta-learner ...")
            ms_final = meta.status_summary()
            report = generate_image_report(
                profile, results, best, config,
                meta_status=ms_final,
                mem_influence=mem_influence,
            )
            report_path = save_image_report(report)

            outcome = memory.add_run(
                profile, config, results, best,
                meta_status=ms_final,
                mem_influence=mem_influence,
            )
            memory.save()
            self._log("OK", f"  Report : {report_path}")
            self._log("OK", f"  Memory : {memory.n_runs} total run(s)  [{outcome}]")

            n_samples = meta.train_from_memory(memory.all_runs())
            if n_samples > 0:
                meta.save()
                ms_final = meta.status_summary()
                self._log("OK",
                          f"  Meta-learner retrained: {n_samples} pipeline samples  "
                          f"(weight={ms_final['weight']:.2f})")
            else:
                self._log("INFO",
                          f"  Meta-learner: not enough data yet "
                          f"({memory.n_runs} run(s) in memory)")

            self._sep()
            self._log("OK", "Agent run complete.")
            self._sep()

            ir = profile.imbalance_ratio
            ir_display = round(ir, 2) if math.isfinite(ir) else 999.9

            q.put({
                "kind":           "done",
                "modality":       "Image",
                "report":         report,
                "report_path":    report_path,
                "cleaned_path":   cleaned_path,
                "cleaned_shape":  cleaned_shape,
                "best_name":      best["spec"].name(),
                "best_score":     bs,
                "best_score_std": sd,
                "metric":         selected_metric,
                "metrics":        best["metrics"],
                "raw_metrics":    best.get("raw_metrics", best["metrics"]),
                "metrics_std":    best.get("metrics_std", {}),
                "normalized_metrics": best.get("normalized_metrics", {}),
                "normalized_metrics_std": best.get("normalized_metrics_std", {}),
                "per_model":      best.get("per_model_metrics", {}),
                "evaluation_mode": best.get("evaluation_mode", ""),
                "evaluation_summary": best.get("evaluation_summary", ""),
                "n_splits":       best.get("n_splits", "?"),
                "n_models":       best.get("n_models", 1),
                "n_pipelines":    len(results),
                "n_images":       profile.n_images,
                "n_classes":      profile.n_classes,
                "avg_height":     round(profile.avg_height),
                "avg_width":      round(profile.avg_width),
                "color_info":     f"{'Grayscale' if profile.has_mostly_grayscale else 'RGB'} "
                                  f"(gray={profile.grayscale_ratio:.0%})",
                "imbalance_ratio": ir_display,
                "task_context":   tc,
                "input_format":   self.input_format,
                "input_format_key": self.input_format_key,
                "annotation_profile": dict(profile.annotation_profile or {}),
                "parsing_summary": dict(profile.parsing_summary or {}),
                "meta_status":    ms_final,
                "mem_influence":  mem_influence,
                "mem_update":     outcome,
            })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _log(self, level: str, text: str) -> None:
        self.q.put({"kind": "log", "level": level, "text": text})

    def _step(self, text: str) -> None:
        self.q.put({"kind": "log", "level": "STEP", "text": text})

    def _sep(self) -> None:
        self.q.put({"kind": "log", "level": "SEP", "text": "─" * 52})
