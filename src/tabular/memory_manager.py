import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config, metric_label, task_family
from .profiler import DataProfile

MEMORY_DIR = Path("memory") / "tabular"
MEMORY_FILE = MEMORY_DIR / "memory.json"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"

_SIMILARITY_THRESHOLD = 0.60
GOOD_SCORE_THRESHOLD = 0.60
_MEMORY_SCHEMA_VERSION = 4
_SCORE_SYSTEM = "normalized_v2"


def _profile_features(profile_summary: dict) -> Dict[str, float]:
    return {
        "missing_ratio": float(profile_summary.get("missing_ratio", 0.0)),
        "imbalance_ratio": float(profile_summary.get("imbalance_ratio", 1.0)),
        "num_col_ratio": float(profile_summary.get("num_col_ratio", 1.0)),
        "cat_col_ratio": float(profile_summary.get("cat_col_ratio", 0.0)),
    }


def _normalise_imbalance(r: float) -> float:
    return min(r, 20.0) / 20.0


def _similarity(fa: Dict[str, float], fb: Dict[str, float]) -> float:
    d_missing = abs(fa["missing_ratio"] - fb["missing_ratio"])
    d_imbalance = abs(_normalise_imbalance(fa["imbalance_ratio"]) - _normalise_imbalance(fb["imbalance_ratio"]))
    d_num = abs(fa["num_col_ratio"] - fb["num_col_ratio"]) * 0.5
    d_cat = abs(fa["cat_col_ratio"] - fb["cat_col_ratio"]) * 0.5
    return max(0.0, 1.0 - (d_missing + d_imbalance + d_num + d_cat) / 3.0)


def _exact_fingerprint(record: dict) -> str:
    ps = record.get("profile_summary", {})
    tc = record.get("task_context", {})
    parts = {
        "dataset": record.get("dataset", ""),
        "target": record.get("target", ""),
        "metric": record.get("metric_priority", ""),
        "task_type": tc.get("task_type", ""),
        "input_format": record.get("input_format", "csv_excel"),
        "fe_budget_norm": tc.get("fe_budget_norm", ""),
        "data_quality_norm": tc.get("data_quality_norm", ""),
        "constraints": tc.get("constraints", ""),
        "best_pipeline": record.get("best_pipeline", {}),
        "n_rows": int(ps.get("n_rows", 0)),
        "n_cols": int(ps.get("n_cols", 0)),
        "missing_ratio": round(float(ps.get("missing_ratio", 0.0)), 3),
    }
    return json.dumps(parts, sort_keys=True)


class MemoryManager:
    def __init__(self) -> None:
        self._runs: List[Dict[str, Any]] = []

    def load(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("schema_version") != _MEMORY_SCHEMA_VERSION or data.get("score_system") != _SCORE_SYSTEM:
                self._runs = []
                self.save()
                if META_LEARNER_FILE.exists():
                    META_LEARNER_FILE.unlink()
            else:
                self._runs = data.get("runs", [])
        else:
            self._runs = []

    def save(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "schema_version": _MEMORY_SCHEMA_VERSION,
                    "score_system": _SCORE_SYSTEM,
                    "runs": self._runs,
                },
                fh,
                indent=2,
            )

    def _current_features(self, profile: DataProfile) -> Dict[str, float]:
        return _profile_features({
            "missing_ratio": profile.total_missing_ratio,
            "imbalance_ratio": profile.imbalance_ratio,
            "num_col_ratio": len(profile.num_cols) / max(profile.n_cols, 1),
            "cat_col_ratio": len(profile.cat_cols) / max(profile.n_cols, 1),
        })

    def find_similar(
        self,
        profile: DataProfile,
        metric: str,
        top_k: int = 3,
        task_type: str = "",
    ) -> List[Dict[str, Any]]:
        if not self._runs:
            return []
        cf = self._current_features(profile)
        current_family = task_family(task_type)
        scored: List[Tuple[float, float, dict]] = []
        for run in self._runs:
            if run.get("metric_priority") != metric:
                continue
            run_family = task_family(run.get("task_context", {}).get("task_type", ""))
            if task_type and run_family != current_family:
                continue
            sim = _similarity(cf, _profile_features(run.get("profile_summary", {})))
            if sim >= _SIMILARITY_THRESHOLD:
                scored.append((sim, run.get("best_score", 0.0), run))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [r for _, _, r in scored[:top_k]]

    def find_good_and_bad(
        self,
        profile: DataProfile,
        metric: str,
        top_k: int = 3,
        task_type: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self._runs:
            return [], []

        cf = self._current_features(profile)
        current_family = task_family(task_type)
        good: List[Tuple[float, float, dict]] = []
        bad: List[Tuple[float, float, dict]] = []

        for run in self._runs:
            if run.get("metric_priority") != metric:
                continue
            run_family = task_family(run.get("task_context", {}).get("task_type", ""))
            if task_type and run_family != current_family:
                continue
            sim = _similarity(cf, _profile_features(run.get("profile_summary", {})))
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

    def add_run(
        self,
        profile: DataProfile,
        config: Config,
        results: List[Dict[str, Any]],
        best: Dict[str, Any],
        meta_status: Optional[dict] = None,
        mem_influence: Optional[dict] = None,
        structure_profile: Optional[dict] = None,
        parsing_summary: Optional[dict] = None,
    ) -> str:
        bp_dict = best["spec"].to_dict()
        ds_name = config.data_path.name

        profile_summary = {
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "missing_ratio": round(profile.total_missing_ratio, 4),
            "imbalance_ratio": round(profile.imbalance_ratio, 2),
            "num_col_ratio": round(len(profile.num_cols) / max(profile.n_cols, 1), 4),
            "cat_col_ratio": round(len(profile.cat_cols) / max(profile.n_cols, 1), 4),
            "num_cols_count": len(profile.num_cols),
            "cat_cols_count": len(profile.cat_cols),
            "high_missing_cols_count": len(profile.high_missing_cols),
            "n_duplicates": profile.n_duplicates,
            "n_classes": profile.n_classes,
            "min_class_size": profile.min_class_size,
            "has_outliers": profile.has_outliers,
            "has_high_skew": profile.has_high_skew,
            "has_high_kurtosis": profile.has_high_kurtosis,
            "has_sparse_features": profile.has_sparse_features,
            "has_multicollinearity": profile.has_multicollinearity,
            "has_high_cardinality": profile.has_high_cardinality,
            "is_imbalanced": profile.is_imbalanced,
            "is_highly_imbalanced": profile.is_highly_imbalanced,
            "high_outlier_cols": profile.high_outlier_cols,
            "high_skew_cols": profile.high_skew_cols,
            "constant_cols_count": len(profile.constant_cols) + len(profile.near_constant_cols),
            "n_high_corr_pairs": profile.n_high_corr_pairs,
        }

        sorted_results = sorted(results, key=lambda r: -r.get("normalized_score", r.get("final_score", 0.0)))
        all_pipelines = [
            {
                "rank": rank + 1,
                "pipeline_name": r["spec"].name(),
                "pipeline_config": r["spec"].to_dict(),
                "selected_metric": r.get("selected_metric", config.metric),
                "metrics": r["metrics"],
                "raw_metrics": r.get("raw_metrics", r["metrics"]),
                "metrics_std": r.get("metrics_std", {}),
                "normalized_metrics": r.get("normalized_metrics", {}),
                "normalized_metrics_std": r.get("normalized_metrics_std", {}),
                "evaluation_mode": r.get("evaluation_mode", ""),
                "evaluator_details": r.get("evaluator_details", {}),
                "evaluation_summary": r.get("evaluation_summary", ""),
                "success": bool(r.get("success", True)),
                "reason": r.get("reason", ""),
                "normalized_score": round(r.get("normalized_score", r.get("final_score", 0.0)), 6),
                "final_score": round(r.get("final_score", 0.0), 6),
                "final_score_std": round(r.get("final_score_std", 0.0), 6),
                "elapsed_sec": round(r["elapsed_sec"], 3),
            }
            for rank, r in enumerate(sorted_results)
        ]

        selected_metric = best.get("selected_metric", config.metric)
        normalized_score = best.get("normalized_score", best.get("final_score", 0.0))
        raw_metrics = best.get("raw_metrics", best["metrics"])
        selection_reasoning = (
            f"Selected by highest normalized score = {normalized_score:.4f} "
            f"from {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f} "
            f"across {best.get('n_splits', '?')} folds x {best.get('n_models', 1)} model(s); "
            f"tie-broken by complexity then evaluation time."
        )

        quality = "good" if normalized_score >= GOOD_SCORE_THRESHOLD else "poor"
        outcome_summary = (
            f"{quality.capitalize()} result: normalized score = {normalized_score:.4f}  |  "
            f"{metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}  |  "
            f"{len(results)} pipeline(s) tested  |  "
            f"best: {best['spec'].name()}"
        )

        input_format_key = (getattr(config, "input_format_key", "") or "csv_excel").strip().lower() or "csv_excel"

        now = datetime.now()
        record: Dict[str, Any] = {
            "id": now.strftime("%Y%m%d_%H%M%S_%f"),
            "timestamp": now.isoformat(),
            "schema_version": _MEMORY_SCHEMA_VERSION,
            "score_system": _SCORE_SYSTEM,
            "dataset": ds_name,
            "target": config.target,
            "metric_priority": config.metric,
            "task_type": config.task_type,
            "modality": "tabular",
            "input_format": input_format_key,
            "input_format_label": getattr(config, "input_format", ""),
            "record_path": getattr(config, "record_path", ""),
            "selected_metric": selected_metric,
            "normalized_score": round(normalized_score, 6),
            "raw_metrics": raw_metrics,
            "evaluation_mode": best.get("evaluation_mode", ""),
            "evaluator_details": best.get("evaluator_details", {}),
            "task_context": config.task_context(),
            "profile_summary": profile_summary,
            "structure_profile": dict(structure_profile or {}),
            "parsing_summary": dict(parsing_summary or {}),
            "all_pipelines_tested": all_pipelines,
            "pipelines_tested": len(results),
            "best_pipeline": bp_dict,
            "best_score": round(normalized_score, 6),
            "best_score_std": round(best.get("final_score_std", 0.0), 6),
            "best_metrics": best["metrics"],
            "best_raw_metrics": raw_metrics,
            "best_normalized_metrics": best.get("normalized_metrics", {}),
            "selection_reasoning": selection_reasoning,
            "memory_influence": mem_influence or {},
            "meta_learner_status": meta_status or {},
            "outcome_summary": outcome_summary,
        }

        fp = _exact_fingerprint(record)
        for existing in self._runs:
            if _exact_fingerprint(existing) == fp:
                return "skipped"

        self._runs.append(record)
        return "added"

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def all_runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)
