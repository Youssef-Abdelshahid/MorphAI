import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import TextConfig, metric_label, task_family
from .profiler import TextProfile

MEMORY_DIR = Path("memory") / "text"
MEMORY_FILE = MEMORY_DIR / "memory.json"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"

_SIMILARITY_THRESHOLD = 0.58
GOOD_SCORE_THRESHOLD = 0.60
_MEMORY_SCHEMA_VERSION = 1
_SCORE_SYSTEM = "normalized_text_v1"


def text_meta_features(profile: TextProfile, task_type: str = "", selected_metric: str = "", pipeline: Optional[dict] = None) -> Dict[str, Any]:
    length_var = profile.token_length_std / max(profile.avg_token_length, 1e-6)
    return {
        "sample_count_bucket": "small" if profile.n_samples < 500 else "medium" if profile.n_samples < 5000 else "large",
        "sample_count_ratio": min(math.log10(max(profile.n_samples, 1)) / 6.0, 1.0),
        "average_length_bucket": "short" if profile.avg_token_length < 30 else "medium" if profile.avg_token_length < 300 else "long",
        "length_variance_bucket": "low" if length_var < 0.25 else "medium" if length_var < 1.0 else "high",
        "vocabulary_size_bucket": "small" if profile.vocabulary_size_estimate < 2000 else "medium" if profile.vocabulary_size_estimate < 50000 else "large",
        "duplicate_ratio": profile.duplicate_text_count / max(profile.n_samples, 1),
        "empty_text_ratio": profile.n_empty_texts / max(profile.n_samples, 1),
        "noise_ratio": profile.noise_ratio,
        "label_count": profile.n_classes,
        "class_imbalance_ratio": profile.imbalance_ratio if math.isfinite(profile.imbalance_ratio) else 999.9,
        "annotation_missing_ratio": profile.annotation_validity.get("invalid_count", 0) / max(profile.n_samples, 1),
        "source_target_length_ratio": profile.source_target_length_ratio,
        "task_type": task_type,
        "supervision_type": "unsupervised" if task_type == "topic_modeling" else "supervised",
        "selected_metric": selected_metric,
        "preprocessing_pipeline_components": pipeline or {},
    }


def _features(summary: dict) -> Dict[str, float]:
    return {
        "samples": min(math.log10(max(float(summary.get("n_samples", 1)), 1.0)) / 6.0, 1.0),
        "avg_len": min(float(summary.get("avg_token_length", 0.0)) / 1000.0, 1.0),
        "len_std": min(float(summary.get("token_length_std", 0.0)) / 1000.0, 1.0),
        "vocab": min(math.log10(max(float(summary.get("vocabulary_size_estimate", 1)), 1.0)) / 6.0, 1.0),
        "duplicates": float(summary.get("duplicate_ratio", 0.0)),
        "empty": float(summary.get("empty_text_ratio", 0.0)),
        "noise": float(summary.get("noise_ratio", 0.0)),
        "classes": min(float(summary.get("n_classes", 0)), 200.0) / 200.0,
        "imbalance": min(float(summary.get("imbalance_ratio", 1.0)), 50.0) / 50.0,
        "annotation_invalid": float(summary.get("annotation_invalid_ratio", 0.0)),
        "source_target": min(float(summary.get("source_target_length_ratio", 0.0)), 5.0) / 5.0,
    }


def _similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = list(a.keys())
    dist = sum(abs(a[k] - b.get(k, 0.0)) for k in keys) / max(len(keys), 1)
    return max(0.0, 1.0 - dist)


def _profile_summary(profile: TextProfile) -> dict:
    ir = profile.imbalance_ratio
    ir_val = round(ir, 2) if math.isfinite(ir) else 999.9
    return {
        "n_samples": profile.n_samples,
        "columns": profile.columns,
        "task_type": profile.task_type,
        "resolved_columns": profile.resolved_columns,
        "primary_text_columns": profile.primary_text_columns,
        "target_columns": profile.target_columns,
        "n_empty_texts": profile.n_empty_texts,
        "duplicate_text_count": profile.duplicate_text_count,
        "duplicate_ratio": round(profile.duplicate_text_count / max(profile.n_samples, 1), 6),
        "empty_text_ratio": round(profile.n_empty_texts / max(profile.n_samples, 1), 6),
        "avg_char_length": round(profile.avg_char_length, 3),
        "avg_token_length": round(profile.avg_token_length, 3),
        "min_char_length": profile.min_char_length,
        "max_char_length": profile.max_char_length,
        "char_length_std": round(profile.char_length_std, 3),
        "token_length_std": round(profile.token_length_std, 3),
        "text_length_distribution": profile.text_length_distribution,
        "vocabulary_size_estimate": profile.vocabulary_size_estimate,
        "unique_token_ratio": round(profile.unique_token_ratio, 6),
        "language_distribution": profile.language_distribution,
        "label_distribution": profile.label_distribution,
        "n_classes": profile.n_classes,
        "imbalance_ratio": ir_val,
        "min_class_size": profile.min_class_size,
        "missing_target_count": profile.missing_target_count,
        "noise_counts": profile.noise_counts,
        "noise_ratios": profile.noise_ratios,
        "noise_ratio": round(profile.noise_ratio, 6),
        "annotation_validity": profile.annotation_validity,
        "annotation_invalid_ratio": profile.annotation_validity.get("invalid_count", 0) / max(profile.n_samples, 1),
        "source_target_length_ratio": round(profile.source_target_length_ratio, 6),
    }


class TextMemoryManager:
    def __init__(self) -> None:
        self._runs: List[Dict[str, Any]] = []

    def load(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("schema_version") == _MEMORY_SCHEMA_VERSION and data.get("score_system") == _SCORE_SYSTEM:
                self._runs = data.get("runs", [])
            else:
                self._runs = []
                self.save()
                if META_LEARNER_FILE.exists():
                    META_LEARNER_FILE.unlink()
        else:
            self._runs = []

    def save(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
            json.dump({"schema_version": _MEMORY_SCHEMA_VERSION, "score_system": _SCORE_SYSTEM, "runs": self._runs}, fh, indent=2)

    def find_good_and_bad(self, profile: TextProfile, metric: str, top_k: int = 3, task_type: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self._runs:
            return [], []
        cf = _features(_profile_summary(profile))
        family = task_family(task_type)
        good = []
        bad = []
        for run in self._runs:
            if run.get("modality") != "text" or run.get("metric_priority") != metric:
                continue
            if task_type and task_family(run.get("task_type", "")) != family:
                continue
            sim = _similarity(cf, _features(run.get("profile_summary", {})))
            if sim < _SIMILARITY_THRESHOLD:
                continue
            score = run.get("best_score", 0.0)
            if score >= GOOD_SCORE_THRESHOLD:
                good.append((sim, score, run))
            else:
                bad.append((sim, score, run))
        good.sort(key=lambda x: (-x[0], -x[1]))
        bad.sort(key=lambda x: (-x[0], -x[1]))
        return [r for _, _, r in good[:top_k]], [r for _, _, r in bad[:top_k]]

    def add_run(self, profile: TextProfile, config: TextConfig, results: List[Dict[str, Any]], best: Dict[str, Any], meta_status: Optional[dict] = None, mem_influence: Optional[dict] = None) -> str:
        selected_metric = best.get("selected_metric", config.metric)
        normalized_score = best.get("normalized_score", best.get("final_score", 0.0))
        raw_metrics = best.get("raw_metrics", best["metrics"])
        summary = _profile_summary(profile)
        best_pipeline = best["spec"].to_dict()
        sorted_results = sorted(results, key=lambda r: -r.get("normalized_score", r.get("final_score", 0.0)))
        all_pipelines = [
            {
                "rank": i + 1,
                "pipeline_name": r["spec"].name(),
                "pipeline_config": r["spec"].to_dict(),
                "selected_metric": r.get("selected_metric", config.metric),
                "raw_metrics": r.get("raw_metrics", r["metrics"]),
                "normalized_metrics": r.get("normalized_metrics", {}),
                "normalized_score": round(r.get("normalized_score", r.get("final_score", 0.0)), 6),
                "evaluation_mode": r.get("evaluation_mode", ""),
                "evaluator_details": r.get("evaluator_details", {}),
                "evaluation_summary": r.get("evaluation_summary", ""),
                "success": bool(r.get("success", True)),
                "reason": r.get("reason", ""),
                "elapsed_sec": round(r.get("elapsed_sec", 0.0), 3),
            }
            for i, r in enumerate(sorted_results)
        ]
        now = datetime.now()
        record = {
            "id": now.strftime("%Y%m%d_%H%M%S_%f"),
            "timestamp": now.isoformat(),
            "schema_version": _MEMORY_SCHEMA_VERSION,
            "score_system": _SCORE_SYSTEM,
            "modality": "text",
            "dataset": config.data_path.name,
            "task_type": config.task_type,
            "metric_priority": config.metric,
            "selected_metric": selected_metric,
            "normalized_score": round(normalized_score, 6),
            "best_score": round(normalized_score, 6),
            "raw_metrics": raw_metrics,
            "evaluation_mode": best.get("evaluation_mode", ""),
            "evaluator_details": best.get("evaluator_details", {}),
            "profile_summary": summary,
            "text_meta_features": text_meta_features(profile, config.task_type, selected_metric, best_pipeline),
            "selected_pipeline": best_pipeline,
            "best_pipeline": best_pipeline,
            "all_pipelines_tested": all_pipelines,
            "task_context": config.task_context(),
            "constraints_options": {"constraints": config.constraints, "language": config.language, "text_source": config.text_source, "text_length": config.text_length},
            "run_metadata": {"pipelines_tested": len(results), "meta_learner_status": meta_status or {}, "memory_influence": mem_influence or {}},
            "selection_reasoning": f"Selected by highest normalized score = {normalized_score:.4f} from {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}; tie-broken by complexity then evaluation time.",
        }
        fingerprint = json.dumps({"dataset": record["dataset"], "task_type": record["task_type"], "metric": record["metric_priority"], "best_pipeline": best_pipeline, "n_samples": summary["n_samples"]}, sort_keys=True)
        for existing in self._runs:
            existing_fp = json.dumps({"dataset": existing.get("dataset"), "task_type": existing.get("task_type"), "metric": existing.get("metric_priority"), "best_pipeline": existing.get("best_pipeline"), "n_samples": existing.get("profile_summary", {}).get("n_samples")}, sort_keys=True)
            if existing_fp == fingerprint:
                return "skipped"
        self._runs.append(record)
        return "added"

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def all_runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)
