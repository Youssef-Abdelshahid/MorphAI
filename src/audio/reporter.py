import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import AudioConfig, metric_label, valid_metrics_for_task
from .memory_manager import audio_meta_features
from .preprocessing import AudioPipelineSpec
from .profiler import AudioProfile

REPORTS_DIR = Path("reports")


def _profile_to_dict(profile: AudioProfile) -> dict:
    ir = profile.imbalance_ratio
    return {
        "n_audio_files": profile.n_audio_files,
        "n_classes": profile.n_classes,
        "class_names": profile.class_names,
        "class_counts": profile.class_counts,
        "imbalance_ratio": round(ir, 2) if math.isfinite(ir) else 999.9,
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
        "corrupt_paths": profile.corrupt_paths,
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


def generate_explanation(profile: AudioProfile, best: Dict[str, Any], metric: str, task_context: Optional[Dict[str, Any]] = None, meta_status: Optional[Dict[str, Any]] = None, mem_influence: Optional[Dict[str, Any]] = None) -> str:
    spec: AudioPipelineSpec = best["spec"]
    selected_metric = best.get("selected_metric", metric)
    raw = best.get("raw_metrics", best["metrics"])
    normalized = best.get("normalized_score", best.get("final_score", 0.0))
    lines = [
        f"The best audio pipeline scored {raw.get(selected_metric, 0.0):.4f} {metric_label(selected_metric)} with a normalized score of {normalized:.4f}.",
        f"Evaluation mode: {best.get('evaluation_mode', 'unknown')}.",
    ]
    if best.get("evaluation_summary"):
        lines.append(best["evaluation_summary"])
    lines.append("")
    lines.append("Audio preprocessing decisions and rationale:")
    lines.append(f"- Resampling target: {spec.target_sample_rate if spec.target_sample_rate else 'native'} Hz.")
    lines.append(f"- Channel handling: {'mono conversion' if spec.mono else 'native channels'}.")
    lines.append(f"- Feature representation: {spec.feature_representation}.")
    lines.append(f"- Duration handling: {spec.duration_strategy}.")
    lines.append(f"- Loudness normalization: {spec.loudness_normalization}.")
    if spec.trim_silence:
        lines.append("- Silence trimming enabled.")
    if spec.noise_filter != "none":
        lines.append(f"- Noise filtering: {spec.noise_filter}.")
    if spec.clipping_handling != "none":
        lines.append(f"- Clipping handling: {spec.clipping_handling}.")
    if spec.augmentation != "none":
        lines.append(f"- Augmentation strategy: {spec.augmentation}.")
    if spec.imbalance != "none":
        lines.append(f"- Class imbalance handling: {spec.imbalance}.")
    if best.get("evaluation_mode") == "proxy":
        lines.append("")
        lines.append("This run used proxy/internal metrics because required ground-truth references were not available.")
    if mem_influence and (mem_influence.get("good_injections") or mem_influence.get("bad_avoidances")):
        lines.append("")
        lines.append(f"Memory influence: {mem_influence.get('good_injections', 0)} positive injection(s), {mem_influence.get('bad_avoidances', 0)} poor pattern avoidance(s).")
    if meta_status:
        lines.append("")
        if meta_status.get("is_mature") and meta_status.get("weight", 0.0) > 0:
            lines.append(f"Meta-learner advisory: active with {meta_status.get('n_train', 0)} training samples and weight {meta_status.get('weight', 0.0):.2f}.")
        else:
            lines.append(f"Meta-learner advisory: learning with {meta_status.get('n_train', 0)}/{meta_status.get('min_to_use', 5)} samples before activation.")
    return "\n".join(lines)


def generate_report(profile: AudioProfile, results: List[Dict[str, Any]], best: Dict[str, Any], config: AudioConfig, meta_status: Optional[Dict[str, Any]] = None, mem_influence: Optional[Dict[str, Any]] = None, mem_update_outcome: Optional[str] = None) -> dict:
    tc = config.task_context()
    sorted_results = sorted(results, key=lambda result: -result.get("normalized_score", result.get("final_score", 0.0)))
    selected_metric = best.get("selected_metric", config.metric)
    profile_dict = _profile_to_dict(profile)
    learning_summary = {}
    if meta_status:
        learning_summary["meta_learner"] = meta_status
    if mem_influence:
        learning_summary["memory_influence"] = mem_influence
    if mem_update_outcome:
        learning_summary["memory_update"] = mem_update_outcome
    return {
        "timestamp": datetime.now().isoformat(),
        "modality": "Audio",
        "config": {"data_path": str(config.data_path), "metric": config.metric, "modality": config.modality, "input_format": getattr(config, "input_format", "")},
        "task_context": tc,
        "profile_summary": profile_dict,
        "audio_meta_features": audio_meta_features(profile, config.task_type, selected_metric, best["spec"].to_dict()),
        "pipelines_tested": len(results),
        "n_models": best.get("n_models", 1),
        "results": [
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
                "normalized_score": result.get("normalized_score", result.get("final_score")),
                "final_score": result.get("final_score"),
                "final_score_std": result.get("final_score_std"),
                "per_model_metrics": result.get("per_model_metrics", {}),
                "evaluation_mode": result.get("evaluation_mode", ""),
                "evaluation_summary": result.get("evaluation_summary", ""),
                "evaluator_details": result.get("evaluator_details", {}),
                "success": result.get("success", True),
                "reason": result.get("reason", ""),
                "n_splits": result.get("n_splits"),
                "elapsed_sec": result.get("elapsed_sec", 0.0),
            }
            for rank, result in enumerate(sorted_results)
        ],
        "best_pipeline": {
            "name": best["spec"].name(),
            "config": best["spec"].to_dict(),
            "selected_metric": selected_metric,
            "metrics": best["metrics"],
            "raw_metrics": best.get("raw_metrics", best["metrics"]),
            "metrics_std": best.get("metrics_std", {}),
            "normalized_metrics": best.get("normalized_metrics", {}),
            "normalized_metrics_std": best.get("normalized_metrics_std", {}),
            "normalized_score": best.get("normalized_score", best.get("final_score")),
            "final_score": best.get("final_score"),
            "final_score_std": best.get("final_score_std"),
            "per_model_metrics": best.get("per_model_metrics", {}),
            "evaluation_mode": best.get("evaluation_mode", ""),
            "evaluation_summary": best.get("evaluation_summary", ""),
            "evaluator_details": best.get("evaluator_details", {}),
            "n_splits": best.get("n_splits"),
            "n_models": best.get("n_models"),
            "elapsed_sec": best.get("elapsed_sec", 0.0),
        },
        "audio_report_sections": {
            "dataset_overview": {"files": profile.n_audio_files, "total_duration_sec": profile.total_duration_sec},
            "duration_statistics": profile.duration_distribution,
            "sampling_rate_distribution": profile.sample_rate_distribution,
            "channel_distribution": profile.channel_count_distribution,
            "file_format_distribution": profile.file_format_distribution,
            "quality_counts": {"corrupted": profile.n_corrupt, "silent": profile.n_silent, "clipped": profile.n_clipped},
            "label_distribution": profile.label_distribution,
            "preprocessing_strategies_tested": [r["spec"].to_dict() for r in sorted_results],
            "best_selected_pipeline": best["spec"].to_dict(),
            "task_specific_raw_metrics": best.get("raw_metrics", best["metrics"]),
            "normalized_score": best.get("normalized_score", best.get("final_score")),
            "evaluation_mode": best.get("evaluation_mode", ""),
        },
        "explanation": generate_explanation(profile, best, selected_metric, task_context=tc, meta_status=meta_status, mem_influence=mem_influence),
        "learning_summary": learning_summary,
    }


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"report_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def print_profile_summary(profile: AudioProfile) -> None:
    ir = profile.imbalance_ratio
    ir_display = f"{ir:.1f}x" if math.isfinite(ir) else ">999x"
    print()
    print("  Audio Dataset Profile")
    print("  " + "-" * 48)
    print(f"  Audio files        : {profile.n_audio_files:,}")
    print(f"  Total duration     : {profile.total_duration_sec:.2f}s")
    print(f"  Avg duration       : {profile.avg_duration_sec:.2f}s  (std={profile.duration_std_sec:.2f}s)")
    print(f"  Duration range     : {profile.min_duration_sec:.2f}s to {profile.max_duration_sec:.2f}s")
    print(f"  Classes            : {profile.n_classes}  (imbalance ratio = {ir_display}, min class = {profile.min_class_size})")
    print(f"  Sample rates       : {profile.sample_rate_distribution}")
    print(f"  Channels           : {profile.channel_count_distribution}")
    print(f"  Formats            : {profile.file_format_distribution}")
    print(f"  Corrupt files      : {profile.n_corrupt}")
    print(f"  Silent / clipped   : {profile.n_silent} / {profile.n_clipped}")
    print(f"  Avg RMS / noise    : {profile.avg_rms:.6f} / {profile.estimated_noise_ratio:.4f}")
