from typing import Any, Dict, List, Optional, Tuple

from .preprocessing import PipelineSpec
from .profiler import DataProfile

_SMOTE_MIN    = 6
_MAX_PIPELINES = 12

_FE_MAX_CANDIDATES = {
    "minimal":  4,
    "light":    8,
    "moderate": _MAX_PIPELINES,
    "heavy":    _MAX_PIPELINES,
}


def _deduplicate(pipelines: List[PipelineSpec]) -> List[PipelineSpec]:
    seen: set = set()
    unique: List[PipelineSpec] = []
    for p in pipelines:
        key = str(sorted(p.to_dict().items()))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _matches_bad_pattern(spec: PipelineSpec, bad_specs: List[PipelineSpec]) -> bool:
    for bad in bad_specs:
        if (
            spec.num_imputer == bad.num_imputer
            and spec.scaler    == bad.scaler
            and spec.encoder   == bad.encoder
            and spec.imbalance == bad.imbalance
        ):
            return True
    return False


def _apply_constraints(spec: PipelineSpec, constraints: List[str]) -> PipelineSpec:
    d = spec.to_dict()
    if "no_smote" in constraints:
        if d["imbalance"] == "smote":
            d["imbalance"] = "oversample"
    if "no_scaling" in constraints:
        d["scaler"] = "none"
    if "no_power_transform" in constraints:
        d["power_transform"] = False
    if "no_outlier_clip" in constraints:
        d["outlier_clip"] = False
    if "no_variance_filter" in constraints:
        d["remove_low_variance"] = False
    return PipelineSpec(**d)


def generate_pipelines(
    profile: DataProfile,
    good_cases:      Optional[List[Dict[str, Any]]] = None,
    bad_cases:       Optional[List[Dict[str, Any]]] = None,
    meta_learner:    Any  = None,
    task_context:    Optional[Dict[str, Any]] = None,
    profile_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[List[PipelineSpec], List[str]]:
    candidates: List[PipelineSpec] = []
    messages:   List[str] = []

    tc = task_context or {}
    constraints      = tc.get("active_constraints") or []
    fe_budget_norm   = tc.get("fe_budget_norm", "moderate")
    data_quality_norm = tc.get("data_quality_norm", "unknown")
    task_type        = tc.get("task_type", "classification")

    prefer_simple    = "prefer_simple" in constraints

    supervision_off = task_type in ("clustering", "anomaly", "association_rules")
    regression_like = task_type in ("regression", "time_series")

    prefer_robust     = profile.has_outliers
    use_power         = profile.has_high_skew
    use_clip          = profile.has_outliers
    drop_miss         = profile.has_high_missing_cols
    smote_ok          = profile.is_highly_imbalanced and profile.min_class_size >= _SMOTE_MIN
    need_oversample   = profile.is_imbalanced
    remove_dupes      = profile.has_duplicates
    remove_lv         = profile.has_constant_cols

    if data_quality_norm == "clean":
        use_clip     = False
        use_power    = False
        prefer_robust = False
    elif data_quality_norm == "noisy":
        prefer_robust = True
        use_clip      = True
        prefer_robust = True

    if fe_budget_norm == "minimal":
        use_power  = False
        smote_ok   = False

    if fe_budget_norm in ("minimal", "light"):
        pass

    allow_knn = (
        fe_budget_norm in ("moderate", "heavy")
        and profile.has_missing
        and profile.n_rows <= 5_000
    )

    primary_num_imp   = "median" if (profile.has_outliers or profile.has_high_missing) else "mean"
    if data_quality_norm == "noisy":
        primary_num_imp = "median"
    secondary_num_imp = "mean" if primary_num_imp == "median" else "median"
    primary_scaler    = "robust" if prefer_robust else "standard"
    secondary_scaler  = "standard" if prefer_robust else "minmax"
    primary_cat_imp   = "mode"

    def _imb(pref: str) -> str:
        if supervision_off:
            return "none"
        if regression_like:
            return "none"
        if pref == "smote":
            return "smote" if smote_ok else ("oversample" if need_oversample else "none")
        if pref == "oversample":
            return "oversample" if need_oversample else "none"
        return "none"

    baseline = PipelineSpec(
        num_imputer="mean", cat_imputer="mode", scaler="standard",
        power_transform=False, encoder="onehot",
        remove_duplicates=False, remove_low_variance=False, imbalance="none",
        outlier_clip=False, drop_high_missing_cols=False,
    )
    candidates.append(baseline)

    if not prefer_simple or len(candidates) < 4:
        candidates.append(PipelineSpec(
            num_imputer=primary_num_imp, cat_imputer=primary_cat_imp,
            scaler=primary_scaler, power_transform=use_power, encoder="onehot",
            remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
            imbalance="none",
            outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
        ))

    if profile.is_imbalanced and not supervision_off and not regression_like:
        if not prefer_simple or len(candidates) < 4:
            candidates.append(PipelineSpec(
                num_imputer=primary_num_imp, cat_imputer=primary_cat_imp,
                scaler=primary_scaler, power_transform=use_power, encoder="onehot",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance=_imb("smote"),
                outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
            ))

    if fe_budget_norm not in ("minimal",):
        if profile.is_imbalanced and not supervision_off and not regression_like:
            candidates.append(PipelineSpec(
                num_imputer=primary_num_imp, cat_imputer=primary_cat_imp,
                scaler=primary_scaler, power_transform=use_power, encoder="onehot",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance="oversample" if need_oversample and not supervision_off else "none",
                outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
            ))

        if profile.has_categorical:
            candidates.append(PipelineSpec(
                num_imputer=primary_num_imp, cat_imputer=primary_cat_imp,
                scaler=primary_scaler, power_transform=False, encoder="ordinal",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance=_imb("smote") if profile.is_imbalanced else "none",
                outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
            ))

        candidates.append(PipelineSpec(
            num_imputer=secondary_num_imp, cat_imputer=primary_cat_imp,
            scaler=secondary_scaler, power_transform=False, encoder="onehot",
            remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
            imbalance="none",
            outlier_clip=False, drop_high_missing_cols=drop_miss,
        ))

        if profile.has_high_cardinality:
            candidates.append(PipelineSpec(
                num_imputer=primary_num_imp, cat_imputer="constant",
                scaler="minmax", power_transform=False, encoder="ordinal",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance=_imb("oversample") if profile.is_imbalanced else "none",
                outlier_clip=False, drop_high_missing_cols=drop_miss,
            ))

    if allow_knn:
        candidates.append(PipelineSpec(
            num_imputer="knn", cat_imputer="constant",
            scaler=primary_scaler, power_transform=False, encoder="onehot",
            remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
            imbalance=_imb("smote") if profile.is_imbalanced else "none",
            outlier_clip=False, drop_high_missing_cols=drop_miss,
        ))

    if fe_budget_norm not in ("minimal", "light"):
        if profile.has_high_skew and not use_power:
            candidates.append(PipelineSpec(
                num_imputer=primary_num_imp, cat_imputer=primary_cat_imp,
                scaler="standard", power_transform=True, encoder="onehot",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance=_imb("smote") if profile.is_imbalanced else "none",
                outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
            ))

        candidates.append(PipelineSpec(
            num_imputer=primary_num_imp, cat_imputer="constant",
            scaler=primary_scaler, power_transform=use_power, encoder="onehot",
            remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
            imbalance=_imb("smote") if profile.is_imbalanced else "none",
            outlier_clip=use_clip, drop_high_missing_cols=drop_miss,
        ))

        if profile.has_outliers:
            candidates.append(PipelineSpec(
                num_imputer="median", cat_imputer=primary_cat_imp,
                scaler="robust", power_transform=False, encoder="onehot",
                remove_duplicates=remove_dupes, remove_low_variance=remove_lv,
                imbalance=_imb("smote") if profile.is_imbalanced else "none",
                outlier_clip=True, drop_high_missing_cols=drop_miss,
            ))

    constraint_msgs = []
    if "no_smote" in constraints:
        constraint_msgs.append("no_smote → SMOTE replaced with oversample where applicable")
    if "no_scaling" in constraints:
        constraint_msgs.append("no_scaling → scaler forced to none")
    if "no_power_transform" in constraints:
        constraint_msgs.append("no_power_transform → power transform disabled")
    if "no_outlier_clip" in constraints:
        constraint_msgs.append("no_outlier_clip → outlier clipping disabled")
    if "no_variance_filter" in constraints:
        constraint_msgs.append("no_variance_filter → variance filter disabled")
    if "prefer_simple" in constraints:
        constraint_msgs.append("prefer_simple → candidate set limited to 4")
    if constraint_msgs:
        messages.append("Constraints applied: " + "; ".join(constraint_msgs))

    if fe_budget_norm == "minimal":
        messages.append("FE budget=Minimal: power transform and KNN imputer disabled, candidate set capped at 4.")
    elif fe_budget_norm == "light":
        messages.append("FE budget=Light: KNN imputer and D/pipeline variants disabled.")
    elif fe_budget_norm == "heavy":
        messages.append("FE budget=Heavy: full candidate set including KNN and power transforms.")

    if data_quality_norm == "noisy":
        messages.append("Data quality=Noisy: preferring robust scaler, median imputer, and outlier clipping.")
    elif data_quality_norm == "clean":
        messages.append("Data quality=Clean: skipping outlier clipping and power transform.")

    if supervision_off:
        messages.append("Unsupervised task: imbalance handling disabled.")
    elif regression_like:
        messages.append("This task family does not use imbalance handling.")

    if bad_cases:
        bad_specs: List[PipelineSpec] = []
        for case in bad_cases:
            d = case.get("best_pipeline")
            if d:
                try:
                    bad_specs.append(PipelineSpec.from_dict(d))
                except Exception:
                    pass

        if bad_specs:
            rest     = candidates[1:]
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
                mem_spec = PipelineSpec.from_dict(d)
                if mem_spec.imbalance == "smote" and profile.min_class_size < _SMOTE_MIN:
                    mem_spec = PipelineSpec(**{**mem_spec.to_dict(), "imbalance": "oversample"})
                if mem_spec.imbalance == "smote" and not smote_ok:
                    mem_spec = PipelineSpec(**{**mem_spec.to_dict(), "imbalance": "none"})
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

    cap = _FE_MAX_CANDIDATES.get(fe_budget_norm, _MAX_PIPELINES)
    if prefer_simple:
        cap = min(cap, 4)

    candidates = _deduplicate(candidates)[:cap]

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
