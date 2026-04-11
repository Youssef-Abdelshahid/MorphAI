import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ImageConfig
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
    score    = best["metrics"][metric]
    n_models = best.get("n_models", 1)

    lines = [
        f"The best pipeline scored {score:.4f} {metric.upper()} "
        f"(averaged across {n_models} model type{'s' if n_models != 1 else ''}).",
    ]

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
        constraint_effects = []
        if "no_augmentation" in active_constraints:
            constraint_effects.append("all data augmentation disabled per constraint")
        if "no_normalization" in active_constraints:
            constraint_effects.append("pixel normalization disabled per constraint")
        if "no_resize" in active_constraints:
            constraint_effects.append("image resizing disabled per constraint")
        if "grayscale_only" in active_constraints:
            constraint_effects.append("forced grayscale conversion per constraint")
        if "no_color_jitter" in active_constraints:
            constraint_effects.append("color jitter disabled per constraint")
        if "preserve_aspect" in active_constraints:
            constraint_effects.append("aspect ratio preservation requested per constraint")
        if constraint_effects:
            lines.append("")
            lines.append("Active constraints and their effects on this run:")
            for eff in constraint_effects:
                lines.append(f"• {eff.capitalize()}.")

    lines += ["", "Preprocessing decisions and rationale:"]

    lines.append(
        f"• Resize: {spec.resize}x{spec.resize} — all images resized to a uniform "
        f"square dimension for consistent feature extraction."
    )

    if spec.color_mode == "grayscale":
        reason = (
            "chosen because the dataset is predominantly grayscale"
            if profile and profile.has_mostly_grayscale
            else "reduces feature dimensionality by removing color channels"
        )
        lines.append(f"• Color Mode: GRAYSCALE — {reason}.")
    else:
        lines.append(
            "• Color Mode: RGB — retains full color information across "
            "three channels for richer feature representation."
        )

    if spec.normalization == "standard":
        lines.append(
            "• Normalization: STANDARD (zero mean, unit variance) — "
            "centres pixel intensities and normalises their spread, "
            "improving convergence for linear classifiers."
        )
    elif spec.normalization == "minmax":
        lines.append(
            "• Normalization: MIN-MAX — maps all pixel values to the [0, 1] range, "
            "useful when brightness varies significantly across samples."
        )
    else:
        lines.append(
            "• Normalization: NONE — raw pixel values are preserved without scaling."
        )

    if spec.histogram_eq:
        lines.append(
            "• Histogram Equalization: ENABLED — enhances contrast by spreading "
            "the intensity distribution more uniformly, helpful for low-contrast images."
        )

    if spec.denoise:
        lines.append(
            "• Denoising: ENABLED — a light Gaussian blur reduces high-frequency "
            "noise while preserving major structural features."
        )

    if spec.sharpen:
        lines.append(
            "• Sharpening: ENABLED — enhances edge details and fine textures "
            "to improve feature discriminability."
        )

    aug_parts = []
    if spec.augment_h_flip:
        aug_parts.append("horizontal flip")
    if spec.augment_v_flip:
        aug_parts.append("vertical flip")
    if spec.augment_rotation != "none":
        angle = "15" if spec.augment_rotation == "light" else "30"
        aug_parts.append(f"random rotation (up to +/-{angle} degrees)")
    if spec.augment_color_jitter:
        aug_parts.append("brightness/contrast jitter")
    if aug_parts:
        lines.append(
            f"• Data Augmentation: {', '.join(aug_parts).upper()} — "
            f"augmented training samples are generated to increase diversity "
            f"and reduce overfitting during cross-validation."
        )

    if spec.imbalance == "oversample":
        lines.append(
            "• Imbalance Handling: RANDOM OVERSAMPLING — minority class images are "
            "duplicated to rebalance the training set before fitting."
        )

    if mem_influence:
        good_inj  = mem_influence.get("good_injections", 0)
        bad_avoid = mem_influence.get("bad_avoidances", 0)
        if good_inj or bad_avoid:
            lines.append("")
            lines.append("Memory influence on this run:")
            if good_inj:
                lines.append(
                    f"• Similarity (positive): {good_inj} pipeline(s) injected from "
                    f"high-scoring similar past run(s)."
                )
            if bad_avoid:
                lines.append(
                    f"• Similarity (avoidance): {bad_avoid} candidate(s) filtered out "
                    f"because they matched low-scoring past pipeline patterns."
                )

    if meta_status:
        is_mature = meta_status.get("is_mature", False)
        n_train   = meta_status.get("n_train", 0)
        weight    = meta_status.get("weight", 0.0)
        lines.append("")
        if is_mature and weight > 0:
            lines.append(
                f"Meta-learner advisory: ACTIVE — trained on {n_train} past "
                f"pipeline evaluations, contributing {weight:.0%} advisory weight "
                f"to candidate ordering (rules + similarity remain primary drivers)."
            )
        else:
            needed = meta_status.get("min_to_use", 5)
            lines.append(
                f"Meta-learner advisory: NOT YET ACTIVE — requires {needed} training "
                f"samples (currently {n_train}).  Ordering based on rules + similarity only."
            )

    return "\n".join(lines)


def _profile_to_dict(profile: ImageProfile) -> dict:
    ir = profile.imbalance_ratio
    ir_val = round(ir, 2) if math.isfinite(ir) else 999.9
    return {
        "n_images":                   profile.n_images,
        "n_classes":                  profile.n_classes,
        "class_names":                profile.class_names,
        "class_counts":               profile.class_counts,
        "imbalance_ratio":            ir_val,
        "min_class_size":             profile.min_class_size,
        "avg_height":                 round(profile.avg_height, 1),
        "avg_width":                  round(profile.avg_width, 1),
        "min_height":                 profile.min_height,
        "min_width":                  profile.min_width,
        "max_height":                 profile.max_height,
        "max_width":                  profile.max_width,
        "height_std":                 round(profile.height_std, 2),
        "width_std":                  round(profile.width_std, 2),
        "avg_aspect_ratio":           round(profile.avg_aspect_ratio, 3),
        "dominant_color_channels":    profile.dominant_color_channels,
        "grayscale_ratio":            round(profile.grayscale_ratio, 4),
        "rgba_ratio":                 round(profile.rgba_ratio, 4),
        "avg_brightness":             round(profile.avg_brightness, 4),
        "brightness_std":             round(profile.brightness_std, 4),
        "avg_contrast":               round(profile.avg_contrast, 4),
        "contrast_std":               round(profile.contrast_std, 4),
        "avg_file_size_kb":           round(profile.avg_file_size_kb, 1),
        "n_corrupt":                  profile.n_corrupt,
        "has_varied_sizes":           profile.has_varied_sizes,
        "has_low_contrast":           profile.has_low_contrast,
        "has_high_contrast_variance": profile.has_high_contrast_variance,
        "has_varied_brightness":      profile.has_varied_brightness,
        "has_grayscale_images":       profile.has_grayscale_images,
        "has_mostly_grayscale":       profile.has_mostly_grayscale,
        "has_rgba_images":            profile.has_rgba_images,
        "has_small_images":           profile.has_small_images,
        "has_large_images":           profile.has_large_images,
        "is_imbalanced":              profile.is_imbalanced,
        "is_highly_imbalanced":       profile.is_highly_imbalanced,
        "is_uniform_size":            profile.is_uniform_size,
        "has_corrupt_images":         profile.has_corrupt_images,
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
        profile, best, config.metric,
        task_context=tc,
        meta_status=meta_status,
        mem_influence=mem_influence,
    )
    sorted_results = sorted(results, key=lambda r: -r["metrics"][config.metric])

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
            "metric":    config.metric,
        },
        "task_context": tc,
        "profile_summary": _profile_to_dict(profile),
        "pipelines_tested": len(results),
        "n_models": best.get("n_models", 1),
        "results": [
            {
                "rank":              rank + 1,
                "pipeline_name":     r["spec"].name(),
                "pipeline_config":   r["spec"].to_dict(),
                "metrics":           r["metrics"],
                "metrics_std":       r.get("metrics_std", {}),
                "per_model_metrics": r.get("per_model_metrics", {}),
                "n_splits":          r.get("n_splits"),
                "elapsed_sec":       r["elapsed_sec"],
            }
            for rank, r in enumerate(sorted_results)
        ],
        "best_pipeline": {
            "name":              best["spec"].name(),
            "config":            best["spec"].to_dict(),
            "metrics":           best["metrics"],
            "metrics_std":       best.get("metrics_std", {}),
            "per_model_metrics": best.get("per_model_metrics", {}),
            "n_splits":          best.get("n_splits"),
            "n_models":          best.get("n_models"),
            "elapsed_sec":       best["elapsed_sec"],
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
    ch_label = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(
        profile.dominant_color_channels, f"{profile.dominant_color_channels}ch")
    print()
    print("  Image Dataset Profile")
    print("  " + "-" * 48)
    print(f"  Total images       : {profile.n_images:,}")
    print(f"  Classes            : {profile.n_classes}  "
          f"(imbalance ratio = {ir_display}, "
          f"min class = {profile.min_class_size} samples)")
    print(f"  Avg dimensions     : {profile.avg_height:.0f} x {profile.avg_width:.0f}")
    print(f"  Dimension range    : [{profile.min_height}x{profile.min_width}] "
          f"to [{profile.max_height}x{profile.max_width}]")
    print(f"  Size uniformity    : {'uniform' if profile.is_uniform_size else 'varied'}"
          f"  (h_std={profile.height_std:.1f}, w_std={profile.width_std:.1f})")
    print(f"  Dominant color     : {ch_label}  "
          f"(grayscale={profile.grayscale_ratio:.0%}, rgba={profile.rgba_ratio:.0%})")
    print(f"  Avg brightness     : {profile.avg_brightness:.3f}  "
          f"(std={profile.brightness_std:.3f})")
    print(f"  Avg contrast       : {profile.avg_contrast:.3f}  "
          f"(std={profile.contrast_std:.3f})")
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
    metric = config.metric
    sorted_results = sorted(results, key=lambda r: -r["metrics"][metric])

    n_splits = best.get("n_splits", "?")
    n_models = best.get("n_models", 1)

    print()
    print("=" * _W)
    print(f"  PIPELINE EVALUATION RESULTS  "
          f"({n_splits}-fold CV x {n_models} models)")
    print("=" * _W)
    col = metric.upper()
    header = f"  {'#':<3} {'Pipeline':<42} {col:>8}  {'inter-model std':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rank, r in enumerate(sorted_results, 1):
        name = r["spec"].name()
        if len(name) > 40:
            name = name[:37] + "..."
        score = r["metrics"][metric]
        std = r.get("metrics_std", {}).get(f"{metric}_std", 0.0)
        marker = " *" if r is best else ""
        print(f"  {rank:<3} {name:<42} {score:>8.4f}  {std:>15.4f}{marker}")

    print("=" * _W)
    print()
    m = best["metrics"]
    s = best.get("metrics_std", {})
    print(f"  Best {metric.upper()} : {m[metric]:.4f} "
          f"(inter-model std {s.get(metric + '_std', 0.0):.4f}  "
          f"over {n_splits} folds x {n_models} models)")
    print(f"  All metrics : "
          f"acc={m['accuracy']:.4f}  "
          f"f1={m['f1']:.4f}  "
          f"prec={m['precision']:.4f}  "
          f"rec={m['recall']:.4f}")

    pmt = best.get("per_model_metrics", {})
    if pmt:
        print(f"  Per-model   : "
              + "  ".join(f"{mn}={mv.get(metric, 0):.4f}" for mn, mv in pmt.items()))

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
        is_mature = meta_status.get("is_mature", False)
        n_train = meta_status.get("n_train", 0)
        weight = meta_status.get("weight", 0.0)
        needed = meta_status.get("min_to_use", 5)
        if is_mature:
            print(f"  Status      : ACTIVE  (weight={weight:.2f}, samples={n_train})")
        else:
            print(f"  Status      : LEARNING  ({n_train}/{needed} samples needed to activate)")

    if mem_update_outcome:
        print()
        print(f"  Memory      : {mem_update_outcome}")

    print()
    print("  EXPLANATION")
    print("  " + "-" * 40)
    explanation = generate_explanation(
        profile, best, metric,
        task_context=tc,
        meta_status=meta_status,
        mem_influence=mem_influence,
    )
    for line in explanation.split("\n"):
        print(f"  {line}")
    print()
