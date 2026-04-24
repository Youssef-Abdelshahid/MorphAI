import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ImageConfig, metric_label, valid_metrics_for_task
from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

REPORTS_DIR = Path("reports")


def generate_explanation(
    profile: ImageProfile,
    best: Dict[str, Any],
    metric: str,
    task_context: Optional[Dict[str, Any]] = None,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
) -> str:
    spec: ImagePipelineSpec = best["spec"]
    selected_metric = best.get("selected_metric", metric)
    raw_metrics = best.get("raw_metrics", best["metrics"])
    score = raw_metrics.get(selected_metric, 0.0)
    normalized_score = best.get("normalized_score", best.get("final_score", score))
    n_models = best.get("n_models", 1)
    lines = [
        f"The best pipeline scored {score:.4f} {metric_label(selected_metric)} with a normalized score of {normalized_score:.4f} "
        f"(averaged across {n_models} model type{'s' if n_models != 1 else ''}).",
    ]
    if best.get("evaluation_mode"):
        lines.append(f"Evaluation mode: {best['evaluation_mode']}.")
    if best.get("evaluation_summary"):
        lines.append(best["evaluation_summary"])
    if task_context:
        task_parts = []
        if task_context.get("task_type"):
            task_parts.append(f"task={task_context['task_type']}")
        if task_context.get("domain"):
            task_parts.append(f"domain={task_context['domain']}")
        if task_context.get("image_format"):
            task_parts.append(f"format={task_context['image_format']}")
        if task_context.get("color_space"):
            task_parts.append(f"color_space={task_context['color_space']}")
        if task_parts:
            lines.append(f"Context: {', '.join(task_parts)}.")
    active_constraints = task_context.get("active_constraints", []) if task_context else []
    if active_constraints:
        lines.append("")
        lines.append("Active constraints and their effects on this run:")
        mapping = {
            "no_augmentation": "all data augmentation disabled per constraint",
            "no_normalization": "pixel normalization disabled per constraint",
            "no_resize": "image resizing disabled per constraint",
            "grayscale_only": "forced grayscale conversion per constraint",
            "no_color_jitter": "color jitter disabled per constraint",
            "preserve_aspect": "aspect ratio preservation requested per constraint",
        }
        for constraint in active_constraints:
            if constraint in mapping:
                lines.append(f"• {mapping[constraint].capitalize()}.")
    lines += ["", "Preprocessing decisions and rationale:"]
    lines.append(f"• Resize: {spec.resize}x{spec.resize} for consistent feature extraction and batch handling.")
    lines.append(f"• Color mode: {spec.color_mode.upper()}.")
    lines.append(f"• Normalization: {spec.normalization.upper()}.")
    if spec.histogram_eq:
        lines.append("• Histogram equalization enabled.")
    if spec.denoise:
        lines.append("• Denoising enabled.")
    if spec.sharpen:
        lines.append("• Sharpening enabled.")
    aug_parts = []
    if spec.augment_h_flip:
        aug_parts.append("horizontal flip")
    if spec.augment_v_flip:
        aug_parts.append("vertical flip")
    if spec.augment_rotation != "none":
        aug_parts.append(f"rotation={spec.augment_rotation}")
    if spec.augment_color_jitter:
        aug_parts.append("color jitter")
    if aug_parts:
        lines.append(f"• Augmentation: {', '.join(aug_parts)}.")
    if spec.imbalance != "none":
        lines.append(f"• Imbalance handling: {spec.imbalance}.")
    if mem_influence:
        good_inj = mem_influence.get("good_injections", 0)
        bad_avoid = mem_influence.get("bad_avoidances", 0)
        if good_inj or bad_avoid:
            lines.append("")
            lines.append("Memory influence on this run:")
            if good_inj:
                lines.append(f"• Similarity (positive): {good_inj} pipeline(s) injected from high-scoring similar past run(s).")
            if bad_avoid:
                lines.append(f"• Similarity (avoidance): {bad_avoid} candidate(s) filtered out because they matched low-scoring past pipeline patterns.")
    if meta_status:
        lines.append("")
        if meta_status.get("is_mature", False) and meta_status.get("weight", 0.0) > 0:
            lines.append(
                f"Meta-learner advisory: ACTIVE — trained on {meta_status.get('n_train', 0)} past pipeline evaluations, "
                f"contributing {meta_status.get('weight', 0.0):.0%} advisory weight to candidate ordering."
            )
        else:
            lines.append(
                f"Meta-learner advisory: NOT YET ACTIVE — requires {meta_status.get('min_to_use', 5)} training samples "
                f"(currently {meta_status.get('n_train', 0)})."
            )
    return "\n".join(lines)


def _profile_to_dict(profile: ImageProfile) -> dict:
    ir = profile.imbalance_ratio
    ir_val = round(ir, 2) if math.isfinite(ir) else 999.9
    return {
        "n_images": profile.n_images,
        "n_classes": profile.n_classes,
        "class_names": profile.class_names,
        "class_counts": profile.class_counts,
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
        "avg_aspect_ratio": round(profile.avg_aspect_ratio, 3),
        "dominant_color_channels": profile.dominant_color_channels,
        "grayscale_ratio": round(profile.grayscale_ratio, 4),
        "rgba_ratio": round(profile.rgba_ratio, 4),
        "avg_brightness": round(profile.avg_brightness, 4),
        "brightness_std": round(profile.brightness_std, 4),
        "avg_contrast": round(profile.avg_contrast, 4),
        "contrast_std": round(profile.contrast_std, 4),
        "avg_file_size_kb": round(profile.avg_file_size_kb, 1),
        "n_corrupt": profile.n_corrupt,
        "has_varied_sizes": profile.has_varied_sizes,
        "has_low_contrast": profile.has_low_contrast,
        "has_high_contrast_variance": profile.has_high_contrast_variance,
        "has_varied_brightness": profile.has_varied_brightness,
        "has_grayscale_images": profile.has_grayscale_images,
        "has_mostly_grayscale": profile.has_mostly_grayscale,
        "has_rgba_images": profile.has_rgba_images,
        "has_small_images": profile.has_small_images,
        "has_large_images": profile.has_large_images,
        "is_imbalanced": profile.is_imbalanced,
        "is_highly_imbalanced": profile.is_highly_imbalanced,
        "is_uniform_size": profile.is_uniform_size,
        "has_corrupt_images": profile.has_corrupt_images,
    }


def generate_report(
    profile: ImageProfile,
    results: List[Dict[str, Any]],
    best: Dict[str, Any],
    config: ImageConfig,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
    mem_update_outcome: Optional[str] = None,
) -> dict:
    tc = config.task_context()
    explanation = generate_explanation(
        profile, best, best.get("selected_metric", config.metric),
        task_context=tc, meta_status=meta_status, mem_influence=mem_influence,
    )
    sorted_results = sorted(results, key=lambda result: -result.get("normalized_score", result.get("final_score", 0.0)))
    learning_summary: Dict[str, Any] = {}
    if meta_status:
        learning_summary["meta_learner"] = meta_status
    if mem_influence:
        learning_summary["memory_influence"] = mem_influence
    if mem_update_outcome:
        learning_summary["memory_update"] = mem_update_outcome
    return {
        "timestamp": datetime.now().isoformat(),
        "modality": "Image",
        "config": {
            "data_path": str(config.data_path),
            "metric": config.metric,
        },
        "task_context": tc,
        "profile_summary": _profile_to_dict(profile),
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
                "elapsed_sec": result["elapsed_sec"],
            }
            for rank, result in enumerate(sorted_results)
        ],
        "best_pipeline": {
            "name": best["spec"].name(),
            "config": best["spec"].to_dict(),
            "selected_metric": best.get("selected_metric", config.metric),
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
            "elapsed_sec": best["elapsed_sec"],
        },
        "explanation": explanation,
        "learning_summary": learning_summary,
    }


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_image_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


_W = 100


def print_profile_summary(profile: ImageProfile) -> None:
    ir = profile.imbalance_ratio
    ir_display = f"{ir:.1f}x" if math.isfinite(ir) else ">999x"
    ch_label = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(profile.dominant_color_channels, f"{profile.dominant_color_channels}ch")
    print()
    print("  Image Dataset Profile")
    print("  " + "-" * 48)
    print(f"  Total images       : {profile.n_images:,}")
    print(f"  Classes            : {profile.n_classes}  (imbalance ratio = {ir_display}, min class = {profile.min_class_size} samples)")
    print(f"  Avg dimensions     : {profile.avg_height:.0f} x {profile.avg_width:.0f}")
    print(f"  Dimension range    : [{profile.min_height}x{profile.min_width}] to [{profile.max_height}x{profile.max_width}]")
    print(f"  Size uniformity    : {'uniform' if profile.is_uniform_size else 'varied'}  (h_std={profile.height_std:.1f}, w_std={profile.width_std:.1f})")
    print(f"  Dominant color     : {ch_label}  (grayscale={profile.grayscale_ratio:.0%}, rgba={profile.rgba_ratio:.0%})")
    print(f"  Avg brightness     : {profile.avg_brightness:.3f}  (std={profile.brightness_std:.3f})")
    print(f"  Avg contrast       : {profile.avg_contrast:.3f}  (std={profile.contrast_std:.3f})")
    print(f"  Avg file size      : {profile.avg_file_size_kb:.1f} KB")
    print(f"  Corrupt images     : {profile.n_corrupt}")


def print_final_summary(
    profile: ImageProfile,
    results: List[Dict[str, Any]],
    best: Dict[str, Any],
    config: ImageConfig,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
    mem_update_outcome: Optional[str] = None,
) -> None:
    metric = best.get("selected_metric", config.metric)
    metric_names = valid_metrics_for_task(config.task_type) or list(best.get("raw_metrics", best["metrics"]).keys())
    sorted_results = sorted(results, key=lambda result: -result.get("normalized_score", result.get("final_score", 0.0)))
    n_splits = best.get("n_splits", "?")
    n_models = best.get("n_models", 1)
    print()
    print("=" * _W)
    print(f"  PIPELINE EVALUATION RESULTS  ({n_splits}-fold CV x {n_models} models)")
    print("=" * _W)
    header = f"  {'#':<3} {'Pipeline':<42} {'Score':>8}  {'inter-model std':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, result in enumerate(sorted_results, 1):
        name = result["spec"].name()
        if len(name) > 40:
            name = name[:37] + "..."
        score = result.get("normalized_score", result.get("final_score", 0.0))
        std = result.get("final_score_std", result.get("metrics_std", {}).get(f"{metric}_std", 0.0))
        marker = " *" if result is best else ""
        print(f"  {rank:<3} {name:<42} {score:>8.4f}  {std:>15.4f}{marker}")
    print("=" * _W)
    raw_metrics = best.get("raw_metrics", best["metrics"])
    print()
    print(
        f"  Best {metric_label(metric)} : {raw_metrics.get(metric, 0.0):.4f} "
        f"(normalized {best.get('normalized_score', best.get('final_score', 0.0)):.4f}  |  "
        f"inter-model std {best.get('metrics_std', {}).get(metric + '_std', 0.0):.4f}  "
        f"over {n_splits} folds x {n_models} models)"
    )
    print("  All metrics : " + "  ".join(f"{metric_label(name)}={raw_metrics.get(name, 0.0):.4f}" for name in metric_names))
    if best.get("per_model_metrics"):
        print("  Per-model   : " + "  ".join(f"{name}={values.get(metric, 0.0):.4f}" for name, values in best["per_model_metrics"].items()))
    if best.get("evaluation_mode"):
        print(f"  Eval mode   : {best['evaluation_mode']}")
    if best.get("evaluation_summary"):
        print(f"  Summary     : {best['evaluation_summary']}")
    tc = config.task_context()
    if any(tc.values()):
        print()
        print("  TASK CONTEXT")
        print("  " + "-" * 40)
        if tc.get("task_type"):
            print(f"  Task type   : {tc['task_type']}")
        if tc.get("domain"):
            print(f"  Domain      : {tc['domain']}")
        if tc.get("image_format"):
            print(f"  Format      : {tc['image_format']}")
        if tc.get("color_space"):
            print(f"  Color space : {tc['color_space']}")
        active = tc.get("active_constraints", [])
        if active:
            print(f"  Constraints : {', '.join(active)}")
        if tc.get("notes"):
            note = tc["notes"]
            if len(note) > 80:
                note = note[:77] + "..."
            print(f"  Notes       : {note}")
    if meta_status:
        print()
        print("  META-LEARNER")
        print("  " + "-" * 40)
        if meta_status.get("is_mature"):
            print(f"  Status      : ACTIVE  (weight={meta_status.get('weight', 0.0):.2f}, samples={meta_status.get('n_train', 0)})")
        else:
            print(f"  Status      : LEARNING  ({meta_status.get('n_train', 0)}/{meta_status.get('min_to_use', 5)} samples needed to activate)")
    if mem_update_outcome:
        print()
        print(f"  Memory      : {mem_update_outcome}")
    print()
    print("  EXPLANATION")
    print("  " + "-" * 40)
    explanation = generate_explanation(
        profile, best, metric, task_context=tc, meta_status=meta_status, mem_influence=mem_influence,
    )
    for line in explanation.split("\n"):
        print(f"  {line}")
    print()
