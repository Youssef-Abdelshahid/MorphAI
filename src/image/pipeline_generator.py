from typing import Any, Dict, List, Optional, Tuple

from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

_MAX_PIPELINES = 12


def _deduplicate(pipelines: List[ImagePipelineSpec]) -> List[ImagePipelineSpec]:
    seen: set = set()
    unique: List[ImagePipelineSpec] = []
    for p in pipelines:
        key = str(sorted(p.to_dict().items()))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _matches_bad_pattern(spec: ImagePipelineSpec, bad_specs: List[ImagePipelineSpec]) -> bool:
    for bad in bad_specs:
        if (
            spec.resize        == bad.resize
            and spec.color_mode    == bad.color_mode
            and spec.normalization == bad.normalization
            and spec.imbalance     == bad.imbalance
        ):
            return True
    return False


def _apply_constraints(spec: ImagePipelineSpec, constraints: List[str]) -> ImagePipelineSpec:
    d = spec.to_dict()
    if "no_augmentation" in constraints:
        d["augment_h_flip"] = False
        d["augment_v_flip"] = False
        d["augment_rotation"] = "none"
        d["augment_color_jitter"] = False
    if "no_normalization" in constraints:
        d["normalization"] = "none"
    if "no_resize" in constraints:
        d["resize"] = 0
    if "grayscale_only" in constraints:
        d["color_mode"] = "grayscale"
    if "no_color_jitter" in constraints:
        d["augment_color_jitter"] = False
    if "no_crop" in constraints:
        pass
    return ImagePipelineSpec(**d)


def generate_pipelines(
    profile: ImageProfile,
    good_cases:      Optional[List[Dict[str, Any]]] = None,
    bad_cases:       Optional[List[Dict[str, Any]]] = None,
    meta_learner:    Any  = None,
    task_context:    Optional[Dict[str, Any]] = None,
    profile_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ImagePipelineSpec], List[str]]:
    candidates: List[ImagePipelineSpec] = []
    messages:   List[str] = []

    tc = task_context or {}
    constraints = tc.get("active_constraints") or []
    task_type = tc.get("task_type", "classification")
    classification_like = task_type in {"classification", "multilabel", "anomaly"}

    use_heq      = profile.has_low_contrast or profile.has_high_contrast_variance
    use_denoise  = profile.has_varied_brightness
    use_sharpen  = profile.has_low_contrast
    use_gray     = profile.has_mostly_grayscale
    need_oversample = profile.is_imbalanced if classification_like else False
    heavily_imbalanced = profile.is_highly_imbalanced if classification_like else False

    primary_color = "grayscale" if use_gray else "rgb"
    secondary_color = "rgb" if use_gray else "grayscale"

    primary_norm = "standard"
    if profile.has_varied_brightness:
        primary_norm = "minmax"

    primary_resize = 64
    if profile.has_small_images and profile.min_height >= 32 and profile.min_width >= 32:
        primary_resize = 32
    elif profile.has_large_images and not profile.has_small_images:
        primary_resize = 128

    secondary_resize = 32 if primary_resize >= 64 else 64

    baseline = ImagePipelineSpec(
        resize=64, color_mode="rgb", normalization="standard",
        histogram_eq=False, denoise=False, sharpen=False,
        augment_h_flip=False, augment_v_flip=False,
        augment_rotation="none", augment_color_jitter=False,
        imbalance="none",
    )
    candidates.append(baseline)

    candidates.append(ImagePipelineSpec(
        resize=primary_resize, color_mode=primary_color,
        normalization=primary_norm,
        histogram_eq=use_heq, denoise=use_denoise, sharpen=use_sharpen,
        augment_h_flip=False, augment_v_flip=False,
        augment_rotation="none", augment_color_jitter=False,
        imbalance="none",
    ))

    if need_oversample:
        candidates.append(ImagePipelineSpec(
            resize=primary_resize, color_mode=primary_color,
            normalization=primary_norm,
            histogram_eq=use_heq, denoise=use_denoise, sharpen=use_sharpen,
            augment_h_flip=False, augment_v_flip=False,
            augment_rotation="none", augment_color_jitter=False,
            imbalance="oversample",
        ))

    candidates.append(ImagePipelineSpec(
        resize=primary_resize, color_mode=primary_color,
        normalization=primary_norm,
        histogram_eq=use_heq, denoise=False, sharpen=False,
        augment_h_flip=True, augment_v_flip=False,
        augment_rotation="light", augment_color_jitter=False,
        imbalance="oversample" if heavily_imbalanced else "none",
    ))

    candidates.append(ImagePipelineSpec(
        resize=secondary_resize, color_mode=primary_color,
        normalization="minmax",
        histogram_eq=False, denoise=False, sharpen=False,
        augment_h_flip=False, augment_v_flip=False,
        augment_rotation="none", augment_color_jitter=False,
        imbalance="none",
    ))

    candidates.append(ImagePipelineSpec(
        resize=primary_resize, color_mode=secondary_color,
        normalization=primary_norm,
        histogram_eq=use_heq, denoise=use_denoise, sharpen=False,
        augment_h_flip=False, augment_v_flip=False,
        augment_rotation="none", augment_color_jitter=False,
        imbalance="none",
    ))

    if profile.has_low_contrast:
        candidates.append(ImagePipelineSpec(
            resize=primary_resize, color_mode=primary_color,
            normalization="minmax",
            histogram_eq=True, denoise=False, sharpen=True,
            augment_h_flip=False, augment_v_flip=False,
            augment_rotation="none", augment_color_jitter=False,
            imbalance="oversample" if need_oversample else "none",
        ))

    candidates.append(ImagePipelineSpec(
        resize=primary_resize, color_mode=primary_color,
        normalization=primary_norm,
        histogram_eq=use_heq, denoise=use_denoise, sharpen=use_sharpen,
        augment_h_flip=True, augment_v_flip=False,
        augment_rotation="moderate", augment_color_jitter=True,
        imbalance="oversample" if need_oversample else "none",
    ))

    if primary_resize != 128:
        candidates.append(ImagePipelineSpec(
            resize=128, color_mode=primary_color,
            normalization=primary_norm,
            histogram_eq=False, denoise=False, sharpen=False,
            augment_h_flip=False, augment_v_flip=False,
            augment_rotation="none", augment_color_jitter=False,
            imbalance="none",
        ))

    if profile.has_varied_sizes:
        candidates.append(ImagePipelineSpec(
            resize=primary_resize, color_mode=primary_color,
            normalization="minmax",
            histogram_eq=True, denoise=True, sharpen=False,
            augment_h_flip=False, augment_v_flip=False,
            augment_rotation="none", augment_color_jitter=False,
            imbalance="none",
        ))

    candidates.append(ImagePipelineSpec(
        resize=primary_resize, color_mode="grayscale",
        normalization="standard",
        histogram_eq=True, denoise=False, sharpen=False,
        augment_h_flip=True, augment_v_flip=False,
        augment_rotation="light", augment_color_jitter=False,
        imbalance="oversample" if need_oversample else "none",
    ))

    constraint_msgs = []
    if "no_augmentation" in constraints:
        constraint_msgs.append("no_augmentation -> all augmentation disabled")
    if "no_normalization" in constraints:
        constraint_msgs.append("no_normalization -> pixel normalization disabled")
    if "no_resize" in constraints:
        constraint_msgs.append("no_resize -> images kept at original size")
    if "grayscale_only" in constraints:
        constraint_msgs.append("grayscale_only -> forced grayscale conversion")
    if "no_color_jitter" in constraints:
        constraint_msgs.append("no_color_jitter -> color jitter disabled")
    if constraint_msgs:
        messages.append("Constraints applied: " + "; ".join(constraint_msgs))

    if bad_cases:
        bad_specs: List[ImagePipelineSpec] = []
        for case in bad_cases:
            d = case.get("best_pipeline")
            if d:
                try:
                    bad_specs.append(ImagePipelineSpec.from_dict(d))
                except Exception:
                    pass

        if bad_specs:
            rest = candidates[1:]
            filtered = [c for c in rest if not _matches_bad_pattern(c, bad_specs)]
            n_avoided = len(rest) - len(filtered)
            candidates = [baseline] + filtered
            if n_avoided > 0:
                messages.append(
                    f"Memory (avoidance): skipped {n_avoided} candidate(s) "
                    f"matching {len(bad_specs)} poor past pipeline pattern(s)."
                )

    if good_cases:
        injected = 0
        for case in good_cases[:3]:
            d = case.get("best_pipeline")
            if not d:
                continue
            try:
                mem_spec = ImagePipelineSpec.from_dict(d)
                candidates.append(mem_spec)
                injected += 1
            except Exception:
                pass
        if injected:
            messages.append(
                f"Memory (positive): injected {injected} pipeline(s) from "
                f"good past run(s) (score >= threshold) as extra candidates."
            )

    if constraints:
        candidates = [_apply_constraints(c, constraints) for c in candidates]

    candidates = _deduplicate(candidates)[:_MAX_PIPELINES]

    if meta_learner is not None and task_context is not None and profile_summary is not None:
        try:
            reordered, ml_msgs = meta_learner.rank_candidates(
                candidates, task_context, profile_summary
            )
            if len(reordered) == len(candidates):
                candidates = reordered
                messages.extend(ml_msgs)
        except Exception:
            pass

    return candidates, messages
