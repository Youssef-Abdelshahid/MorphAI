import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AudioConfig, metric_label, task_family
from .profiler import AudioProfile

MEMORY_DIR = Path("memory") / "audio"
MEMORY_FILE = MEMORY_DIR / "memory.json"
META_LEARNER_FILE = MEMORY_DIR / "meta_learner.pkl"

_SIMILARITY_THRESHOLD = 0.60
GOOD_SCORE_THRESHOLD = 0.60
_MEMORY_SCHEMA_VERSION = 1
_SCORE_SYSTEM = "normalized_audio_v1"


def audio_meta_features(profile: AudioProfile, task_type: str = "", selected_metric: str = "", pipeline: Optional[dict] = None) -> Dict[str, Any]:
    avg = profile.avg_duration_sec
    dur_var = profile.duration_std_sec / max(avg, 1e-6)
    return {
        "file_count_bucket": "small" if profile.n_audio_files < 100 else "medium" if profile.n_audio_files < 1000 else "large",
        "file_count_ratio": min(math.log10(max(profile.n_audio_files, 1)) / 6.0, 1.0),
        "average_duration_bucket": "short" if avg < 5 else "medium" if avg < 60 else "long",
        "duration_variance_bucket": "low" if dur_var < 0.25 else "medium" if dur_var < 1.0 else "high",
        "sampling_rate_consistency": 1.0 / max(len(profile.sample_rate_distribution), 1),
        "channel_consistency": 1.0 / max(len(profile.channel_count_distribution), 1),
        "silence_ratio": profile.silence_ratio,
        "clipping_ratio": profile.clipping_ratio,
        "corruption_ratio": profile.corruption_ratio,
        "estimated_noise_ratio": profile.estimated_noise_ratio,
        "class_imbalance_ratio": profile.imbalance_ratio if math.isfinite(profile.imbalance_ratio) else 999.9,
        "label_count": profile.n_classes,
        "task_type": task_type,
        "supervision_type": "unsupervised" if task_type == "anomaly" and not profile.has_labels else "supervised",
        "selected_metric": selected_metric,
        "preprocessing_pipeline_components": pipeline or {},
    }


def _features(summary: dict) -> Dict[str, float]:
    return {
        "file_count": min(math.log10(max(float(summary.get("n_audio_files", 1)), 1.0)) / 6.0, 1.0),
        "avg_duration": min(float(summary.get("avg_duration_sec", 0.0)) / 120.0, 1.0),
        "duration_std": min(float(summary.get("duration_std_sec", 0.0)) / 120.0, 1.0),
        "silence": float(summary.get("silence_ratio", 0.0)),
        "clipping": float(summary.get("clipping_ratio", 0.0)),
        "corruption": float(summary.get("corruption_ratio", 0.0)),
        "noise": float(summary.get("estimated_noise_ratio", 0.0)),
        "imbalance": min(float(summary.get("imbalance_ratio", 1.0)), 20.0) / 20.0,
        "classes": min(float(summary.get("n_classes", 0)), 100.0) / 100.0,
    }


def _similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = list(a.keys())
    dist = sum(abs(a[k] - b.get(k, 0.0)) for k in keys) / max(len(keys), 1)
    return max(0.0, 1.0 - dist)


def _profile_summary(profile: AudioProfile) -> dict:
    ir = profile.imbalance_ratio
    ir_val = round(ir, 2) if math.isfinite(ir) else 999.9
    return {
        "n_audio_files": profile.n_audio_files,
        "n_classes": profile.n_classes,
        "class_counts": profile.class_counts,
        "imbalance_ratio": ir_val,
        "min_class_size": profile.min_class_size,
        "total_duration_sec": round(profile.total_duration_sec, 3),
        "avg_duration_sec": round(profile.avg_duration_sec, 3),
        "min_duration_sec": round(profile.min_duration_sec, 3),
        "max_duration_sec": round(profile.max_duration_sec, 3),
        "duration_std_sec": round(profile.duration_std_sec, 3),
        "duration_distribution": profile.duration_distribution,
        "sample_rate_distribution": profile.sample_rate_distribution,
        "channel_count_distribution": profile.channel_count_distribution,
        "bit_depth_distribution": profile.bit_depth_distribution,
        "file_format_distribution": profile.file_format_distribution,
        "n_corrupt": profile.n_corrupt,
        "n_silent": profile.n_silent,
        "n_clipped": profile.n_clipped,
        "avg_rms": round(profile.avg_rms, 6),
        "rms_std": round(profile.rms_std, 6),
        "avg_loudness_db": round(profile.avg_loudness_db, 3),
        "noise_proxy": round(profile.noise_proxy, 6),
        "silence_ratio": round(profile.silence_ratio, 6),
        "clipping_ratio": round(profile.clipping_ratio, 6),
        "corruption_ratio": round(profile.corruption_ratio, 6),
        "estimated_noise_ratio": round(profile.estimated_noise_ratio, 6),
        "label_distribution": profile.label_distribution,
        "missing_invalid_labels": profile.missing_invalid_labels,
        "transcript_count": profile.transcript_count,
        "speaker_label_count": profile.speaker_label_count,
        "annotation_counts": profile.annotation_counts,
    }


class AudioMemoryManager:
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

    def find_good_and_bad(self, profile: AudioProfile, metric: str, top_k: int = 3, task_type: str = "") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self._runs:
            return [], []
        cf = _features(_profile_summary(profile))
        family = task_family(task_type)
        good = []
        bad = []
        for run in self._runs:
            if run.get("modality") != "audio" or run.get("metric_priority") != metric:
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

    def add_run(self, profile: AudioProfile, config: AudioConfig, results: List[Dict[str, Any]], best: Dict[str, Any], meta_status: Optional[dict] = None, mem_influence: Optional[dict] = None) -> str:
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
            "modality": "audio",
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
            "audio_meta_features": audio_meta_features(profile, config.task_type, selected_metric, best_pipeline),
            "selected_pipeline": best_pipeline,
            "best_pipeline": best_pipeline,
            "all_pipelines_tested": all_pipelines,
            "task_context": config.task_context(),
            "constraints_options": {"constraints": config.constraints, "audio_format": config.audio_format, "channel_layout": config.channel_layout, "sample_rate": config.sample_rate},
            "run_metadata": {"pipelines_tested": len(results), "meta_learner_status": meta_status or {}, "memory_influence": mem_influence or {}},
            "selection_reasoning": f"Selected by highest normalized score = {normalized_score:.4f} from {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}; tie-broken by complexity then evaluation time.",
        }
        fingerprint = json.dumps({"dataset": record["dataset"], "task_type": record["task_type"], "metric": record["metric_priority"], "best_pipeline": best_pipeline, "n_audio_files": summary["n_audio_files"]}, sort_keys=True)
        for existing in self._runs:
            existing_fp = json.dumps({"dataset": existing.get("dataset"), "task_type": existing.get("task_type"), "metric": existing.get("metric_priority"), "best_pipeline": existing.get("best_pipeline"), "n_audio_files": existing.get("profile_summary", {}).get("n_audio_files")}, sort_keys=True)
            if existing_fp == fingerprint:
                return "skipped"
        self._runs.append(record)
        return "added"

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def all_runs(self) -> List[Dict[str, Any]]:
        return list(self._runs)
