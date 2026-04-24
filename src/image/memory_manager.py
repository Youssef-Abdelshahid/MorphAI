import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ImageConfig, metric_label, task_family
from .profiler import ImageProfile

MEMORY_DIR = Path("memory") / "image"
MEMORY_FILE = MEMORY_DIR / "memory.json"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"

_SIMILARITY_THRESHOLD = 0.60
GOOD_SCORE_THRESHOLD = 0.60
_MEMORY_SCHEMA_VERSION = 2
_SCORE_SYSTEM = "normalized_v2"


def _profile_features(profile_summary: dict) -> Dict[str, float]:
    return {
        "imbalance_ratio": float(profile_summary.get("imbalance_ratio", 1.0)),
        "brightness_mean": float(profile_summary.get("avg_brightness", 0.5)),
        "contrast_mean": float(profile_summary.get("avg_contrast", 0.15)),
        "grayscale_ratio": float(profile_summary.get("grayscale_ratio", 0.0)),
    }


def _normalise_imbalance(r: float) -> float:
    return min(r, 20.0) / 20.0


def _similarity(fa: Dict[str, float], fb: Dict[str, float]) -> float:
    d_imbalance = abs(_normalise_imbalance(fa["imbalance_ratio"]) - _normalise_imbalance(fb["imbalance_ratio"]))
    d_brightness = abs(fa["brightness_mean"] - fb["brightness_mean"])
    d_contrast = abs(fa["contrast_mean"] - fb["contrast_mean"])
    d_grayscale = abs(fa["grayscale_ratio"] - fb["grayscale_ratio"]) * 0.5
    return max(0.0, 1.0 - (d_imbalance + d_brightness + d_contrast + d_grayscale) / 3.0)


def _exact_fingerprint(record: dict) -> str:
    ps = record.get("profile_summary", {})
    tc = record.get("task_context", {})
    parts = {
        "dataset": record.get("dataset", ""),
        "metric": record.get("metric_priority", ""),
        "task_type": tc.get("task_type", ""),
        "constraints": tc.get("constraints", ""),
        "best_pipeline": record.get("best_pipeline", {}),
        "n_images": int(ps.get("n_images", 0)),
        "n_classes": int(ps.get("n_classes", 0)),
    }
    return json.dumps(parts, sort_keys=True)


class ImageMemoryManager:
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

    def _current_features(self, profile: ImageProfile) -> Dict[str, float]:
        return _profile_features({
            "imbalance_ratio": profile.imbalance_ratio,
            "avg_brightness": profile.avg_brightness,
            "avg_contrast": profile.avg_contrast,
            "grayscale_ratio": profile.grayscale_ratio,
        })

    def find_similar(self, profile: ImageProfile, metric: str, top_k: int = 3, task_type: str = "") -> List[Dict[str, Any]]:
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
        return [run for _, _, run in scored[:top_k]]

    def find_good_and_bad(self, profile: ImageProfile, metric: str, top_k: int = 3, task_type: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        return [run for _, _, run in good[:top_k]], [run for _, _, run in bad[:top_k]]

    def add_run(
        self,
        profile: ImageProfile,
        config: ImageConfig,
        results: List[Dict[str, Any]],
        best: Dict[str, Any],
        meta_status: Optional[dict] = None,
        mem_influence: Optional[dict] = None,
    ) -> str:
        bp_dict = best["spec"].to_dict()
        ds_name = config.data_path.name
        ir = profile.imbalance_ratio
        ir_val = round(ir, 2) if math.isfinite(ir) else 999.9
        profile_summary = {
            "n_images": profile.n_images,
            "n_classes": profile.n_classes,
            "imbalance_ratio": ir_val,
            "min_class_size": profile.min_class_size,
            "avg_height": round(profile.avg_height, 1),
            "avg_width": round(profile.avg_width, 1),
            "min_height": profile.min_height,
            "min_width": profile.min_width,
            "max_height": profile.max_height,
            "max_width": profile.max_width,
            "height_std": round(profile.height_std, 2),
            "width_std": round(profile.width_std, 2),
            "avg_brightness": round(profile.avg_brightness, 4),
            "brightness_std": round(profile.brightness_std, 4),
            "avg_contrast": round(profile.avg_contrast, 4),
            "contrast_std": round(profile.contrast_std, 4),
            "grayscale_ratio": round(profile.grayscale_ratio, 4),
            "rgba_ratio": round(profile.rgba_ratio, 4),
            "dominant_color_channels": profile.dominant_color_channels,
            "avg_file_size_kb": round(profile.avg_file_size_kb, 1),
            "n_corrupt": profile.n_corrupt,
            "has_varied_sizes": profile.has_varied_sizes,
            "has_low_contrast": profile.has_low_contrast,
            "has_high_contrast_variance": profile.has_high_contrast_variance,
            "has_varied_brightness": profile.has_varied_brightness,
            "has_grayscale_images": profile.has_grayscale_images,
            "has_mostly_grayscale": profile.has_mostly_grayscale,
            "has_small_images": profile.has_small_images,
            "has_large_images": profile.has_large_images,
            "is_imbalanced": profile.is_imbalanced,
            "is_highly_imbalanced": profile.is_highly_imbalanced,
            "is_uniform_size": profile.is_uniform_size,
        }

        sorted_results = sorted(results, key=lambda r: -r.get("normalized_score", r.get("final_score", 0.0)))
        all_pipelines = [
            {
                "rank": rank + 1,
                "pipeline_name": result["spec"].name(),
                "pipeline_config": result["spec"].to_dict(),
                "selected_metric": result.get("selected_metric", config.metric),
                "metrics": result["metrics"],
                "raw_metrics": result.get("raw_metrics", result["metrics"]),
                "metrics_std": result.get("metrics_std", {}),
                "normalized_metrics": result.get("normalized_metrics", {}),
                "normalized_metrics_std": result.get("normalized_metrics_std", {}),
                "evaluation_mode": result.get("evaluation_mode", ""),
                "evaluator_details": result.get("evaluator_details", {}),
                "evaluation_summary": result.get("evaluation_summary", ""),
                "success": bool(result.get("success", True)),
                "reason": result.get("reason", ""),
                "normalized_score": round(result.get("normalized_score", result.get("final_score", 0.0)), 6),
                "final_score": round(result.get("final_score", 0.0), 6),
                "final_score_std": round(result.get("final_score_std", 0.0), 6),
                "elapsed_sec": round(result["elapsed_sec"], 3),
            }
            for rank, result in enumerate(sorted_results)
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
            f"{len(results)} pipeline(s) tested  |  best: {best['spec'].name()}"
        )

        now = datetime.now()
        record: Dict[str, Any] = {
            "id": now.strftime("%Y%m%d_%H%M%S_%f"),
            "timestamp": now.isoformat(),
            "schema_version": _MEMORY_SCHEMA_VERSION,
            "score_system": _SCORE_SYSTEM,
            "dataset": ds_name,
            "metric_priority": config.metric,
            "task_type": config.task_type,
            "selected_metric": selected_metric,
            "normalized_score": round(normalized_score, 6),
            "raw_metrics": raw_metrics,
            "evaluation_mode": best.get("evaluation_mode", ""),
            "evaluator_details": best.get("evaluator_details", {}),
            "task_context": config.task_context(),
            "profile_summary": profile_summary,
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

        fingerprint = _exact_fingerprint(record)
        for existing in self._runs:
            if _exact_fingerprint(existing) == fingerprint:
                return "skipped"
        self._runs.append(record)
        return "added"

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def all_runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)
