import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, metric_label, valid_metrics_for_task
from .preprocessing import PipelineSpec
from .profiler import DataProfile

REPORTS_DIR = Path("reports")


def generate_explanation(
    profile: DataProfile,
    best: Dict[str, Any],
    metric: str,
    task_context: Optional[Dict[str, Any]] = None,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
) -> str:
    spec: PipelineSpec = best["spec"]
    selected_metric = best.get("selected_metric", metric)
    raw_metrics = best.get("raw_metrics", best["metrics"])
    score = raw_metrics.get(selected_metric, 0.0)
    final_score = best.get("normalized_score", best.get("final_score", score))
    n_models = best.get("n_models", 1)
    evaluation_mode = best.get("evaluation_mode", "")

    lines = [
        f"The best pipeline scored {score:.4f} {metric_label(selected_metric)} "
        f"with a normalized score of {final_score:.4f} "
        f"(averaged across {n_models} model type{'s' if n_models != 1 else ''}).",
    ]
    if evaluation_mode:
        lines.append(f"Evaluation mode: {evaluation_mode}.")
    evaluator_details = best.get("evaluator_details") or {}
    family = evaluator_details.get("model_family")
    used_models = evaluator_details.get("models")
    baselines = evaluator_details.get("baselines")
    if family:
        lines.append(f"Evaluator family: {family}.")
    if used_models:
        lines.append(f"Models used: {', '.join(str(m) for m in used_models)}.")
    if baselines:
        lines.append(f"Baselines: {', '.join(str(b) for b in baselines)}.")
    if best.get("evaluation_summary"):
        lines.append(best["evaluation_summary"])

    if task_context:
        task_parts = []
        if task_context.get("task_type") and task_context["task_type"] != "classification":
            task_parts.append(f"task={task_context['task_type']}")
        if task_context.get("domain"):
            task_parts.append(f"domain={task_context['domain']}")
        fe_val = task_context.get("fe_budget", "")
        if fe_val:
            task_parts.append(f"fe_budget={fe_val}")
        dq_val = task_context.get("data_quality", "")
        if dq_val:
            task_parts.append(f"data_quality={dq_val}")
        if task_parts:
            lines.append(f"Context: {', '.join(task_parts)}.")

    active_constraints = task_context.get("active_constraints", []) if task_context else []
    if active_constraints:
        constraint_effects = []
        if "no_smote" in active_constraints:
            constraint_effects.append("SMOTE replaced with oversample (or none) per constraint")
        if "no_scaling" in active_constraints:
            constraint_effects.append("feature scaling disabled per constraint")
        if "no_power_transform" in active_constraints:
            constraint_effects.append("power transform disabled per constraint")
        if "no_outlier_clip" in active_constraints:
            constraint_effects.append("outlier clipping disabled per constraint")
        if "no_variance_filter" in active_constraints:
            constraint_effects.append("variance filter disabled per constraint")
        if "prefer_simple" in active_constraints:
            constraint_effects.append("candidate set limited to simpler pipelines per constraint")
        if constraint_effects:
            lines.append("")
            lines.append("Active constraints and their effects on this run:")
            for eff in constraint_effects:
                lines.append(f"• {eff.capitalize()}.")

    lines += ["", "Preprocessing decisions and rationale:"]

    if spec.num_imputer == "median":
        reason = (
            "chosen because outliers were detected — median is unaffected by extreme values"
            if profile and profile.has_outliers
            else "more robust than the mean for skewed or asymmetric distributions"
        )
        lines.append(f"• Numerical Imputation: MEDIAN — {reason}.")
    elif spec.num_imputer == "knn":
        lines.append(
            "• Numerical Imputation: KNN — uses the values of nearest neighbours "
            "to estimate missing entries, capturing local data structure."
        )
    else:
        lines.append(
            "• Numerical Imputation: MEAN — replaces missing values with the column "
            "average, suitable for near-symmetric distributions with few gaps."
        )

    if spec.outlier_clip:
        lines.append(
            "• Outlier Clipping: ENABLED — extreme values are capped to the "
            "[Q1 - 1.5×IQR, Q3 + 1.5×IQR] range per feature before any scaling."
        )

    if spec.cat_imputer == "constant":
        lines.append(
            "• Categorical Imputation: CONSTANT — missing categories are replaced "
            "with 'missing', treating absence of data as its own informative signal."
        )
    else:
        lines.append(
            "• Categorical Imputation: MODE — fills missing entries with the most "
            "frequent category, preserving the dominant distribution."
        )

    if spec.drop_high_missing_cols:
        lines.append(
            "• High-Missing Columns: DROPPED — columns with more than 50% missing "
            "values were removed to avoid introducing excessive noise through imputation."
        )

    if spec.scaler == "robust":
        lines.append(
            "• Feature Scaling: ROBUST — centres using the median and scales by the "
            "IQR, making it resilient to outliers during normalisation."
        )
    elif spec.scaler == "standard":
        lines.append(
            "• Feature Scaling: STANDARD (z-score) — shifts each feature to zero mean "
            "and unit variance, normalising magnitudes across columns."
        )
    elif spec.scaler == "minmax":
        lines.append(
            "• Feature Scaling: MIN-MAX — maps all feature values to the [0, 1] range, "
            "useful when a bounded input space is preferred."
        )
    else:
        lines.append(
            "• Feature Scaling: NONE — raw values are preserved without any normalisation."
        )

    if spec.power_transform:
        lines.append(
            "• Power Transform: YEO-JOHNSON — applied to reduce skewness in numeric "
            "features and improve linear separability for the classifier."
        )

    if spec.encoder == "onehot":
        lines.append(
            "• Categorical Encoding: ONE-HOT — each category becomes a binary column, "
            "avoiding any implied numeric ordering between categories."
        )
    else:
        lines.append(
            "• Categorical Encoding: ORDINAL — categories are mapped to integers, "
            "a compact representation well-suited to tree-based models."
        )

    if spec.imbalance == "smote":
        lines.append(
            "• Imbalance Handling: SMOTE — synthetic minority samples are generated "
            "by interpolating between existing examples to balance class sizes."
        )
    elif spec.imbalance == "oversample":
        lines.append(
            "• Imbalance Handling: RANDOM OVERSAMPLING — minority class rows are "
            "duplicated to rebalance the training set before fitting."
        )

    if spec.remove_duplicates:
        lines.append(
            "• Duplicate Removal: ENABLED — identical rows are dropped to prevent "
            "repeated samples from biasing the classifier during training."
        )
    if spec.remove_low_variance:
        lines.append(
            "• Low-Variance Filter: ENABLED — near-constant features with negligible "
            "variation are removed as they carry little predictive information."
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


def _profile_to_dict(profile: DataProfile) -> dict:
    return {
        "n_rows":                    profile.n_rows,
        "n_cols":                    profile.n_cols,
        "num_cols_count":            len(profile.num_cols),
        "cat_cols_count":            len(profile.cat_cols),
        "total_missing_ratio":       round(profile.total_missing_ratio, 4),
        "high_missing_cols_count":   len(profile.high_missing_cols),
        "high_missing_cols":         profile.high_missing_cols,
        "n_duplicates":              profile.n_duplicates,
        "n_classes":                 profile.n_classes,
        "imbalance_ratio":           round(profile.imbalance_ratio, 2),
        "min_class_size":            profile.min_class_size,
        "high_outlier_cols_count":   len(profile.high_outlier_cols),
        "high_outlier_cols":         profile.high_outlier_cols,
        "high_skew_cols_count":      len(profile.high_skew_cols),
        "high_skew_cols":            profile.high_skew_cols,
        "high_kurtosis_cols_count":  len(profile.high_kurtosis_cols),
        "high_cardinality_cols_count": len(profile.high_cardinality_cols),
        "high_cardinality_cols":     profile.high_cardinality_cols,
        "constant_cols_count":       len(profile.constant_cols) + len(profile.near_constant_cols),
        "binary_num_cols_count":     len(profile.binary_num_cols),
        "n_high_corr_pairs":         profile.n_high_corr_pairs,
        "has_sparse_features":       profile.has_sparse_features,
        "has_multicollinearity":     profile.has_multicollinearity,
        "missing_ratio":   round(profile.total_missing_ratio, 4),
        "num_col_ratio":   round(len(profile.num_cols) / max(profile.n_cols, 1), 4),
        "cat_col_ratio":   round(len(profile.cat_cols) / max(profile.n_cols, 1), 4),
        "has_outliers":           profile.has_outliers,
        "has_high_skew":          profile.has_high_skew,
        "has_high_kurtosis":      profile.has_high_kurtosis,
        "is_imbalanced":          profile.is_imbalanced,
        "is_highly_imbalanced":   profile.is_highly_imbalanced,
        "has_high_cardinality":   profile.has_high_cardinality,
        "has_high_missing_cols":  profile.has_high_missing_cols,
        "has_duplicates":         profile.has_duplicates,
        "has_categorical":        profile.has_categorical,
    }


def generate_report(
    profile: DataProfile,
    results: List[Dict[str, Any]],
    best: Dict[str, Any],
    config: Config,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
    mem_update_outcome: Optional[str] = None,
    structure_profile: Optional[Dict[str, Any]] = None,
    parsing_summary: Optional[Dict[str, Any]] = None,
    parser_warnings: Optional[List[str]] = None,
) -> dict:
    tc      = config.task_context()
    explanation = generate_explanation(
        profile, best, best.get("selected_metric", config.metric),
        task_context=tc,
        meta_status=meta_status,
        mem_influence=mem_influence,
    )
    sorted_results = sorted(results, key=lambda r: -r.get("normalized_score", r.get("final_score", 0.0)))

    learning_summary: Dict[str, Any] = {}
    if meta_status:
        learning_summary["meta_learner"] = meta_status
    if mem_influence:
        learning_summary["memory_influence"] = mem_influence
    if mem_update_outcome:
        learning_summary["memory_update"] = mem_update_outcome

    return {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_path":    str(config.data_path),
            "target":       config.target,
            "metric":       config.metric,
            "modality":     config.modality,
            "input_format": getattr(config, "input_format", ""),
            "input_format_key": getattr(config, "input_format_key", ""),
            "record_path": getattr(config, "record_path", ""),
        },
        "task_context": tc,
        "structure_profile": dict(structure_profile or {}),
        "parsing_summary": dict(parsing_summary or {}),
        "parser_warnings": list(parser_warnings or []),
        "profile_summary": _profile_to_dict(profile),
        "pipelines_tested": len(results),
        "n_models": best.get("n_models", 1),
        "results": [
            {
                "rank":              rank + 1,
                "pipeline_name":     r["spec"].name(),
                "pipeline_config":   r["spec"].to_dict(),
                "selected_metric":   r.get("selected_metric", config.metric),
                "metrics":           r["metrics"],
                "raw_metrics":       r.get("raw_metrics", r["metrics"]),
                "metrics_std":       r.get("metrics_std", {}),
                "normalized_metrics": r.get("normalized_metrics", {}),
                "normalized_metrics_std": r.get("normalized_metrics_std", {}),
                "normalized_score":  r.get("normalized_score", r.get("final_score")),
                "final_score":       r.get("final_score"),
                "final_score_std":   r.get("final_score_std"),
                "per_model_metrics": r.get("per_model_metrics", {}),
                "evaluation_mode":   r.get("evaluation_mode", ""),
                "evaluation_summary": r.get("evaluation_summary", ""),
                "evaluator_details": r.get("evaluator_details", {}),
                "success":           r.get("success", True),
                "reason":            r.get("reason", ""),
                "n_splits":          r.get("n_splits"),
                "elapsed_sec":       r["elapsed_sec"],
            }
            for rank, r in enumerate(sorted_results)
        ],
        "best_pipeline": {
            "name":              best["spec"].name(),
            "config":            best["spec"].to_dict(),
            "selected_metric":   best.get("selected_metric", config.metric),
            "metrics":           best["metrics"],
            "raw_metrics":       best.get("raw_metrics", best["metrics"]),
            "metrics_std":       best.get("metrics_std", {}),
            "normalized_metrics": best.get("normalized_metrics", {}),
            "normalized_metrics_std": best.get("normalized_metrics_std", {}),
            "normalized_score":  best.get("normalized_score", best.get("final_score")),
            "final_score":       best.get("final_score"),
            "final_score_std":   best.get("final_score_std"),
            "per_model_metrics": best.get("per_model_metrics", {}),
            "evaluation_mode":   best.get("evaluation_mode", ""),
            "evaluation_summary": best.get("evaluation_summary", ""),
            "evaluator_details": best.get("evaluator_details", {}),
            "n_splits":          best.get("n_splits"),
            "n_models":          best.get("n_models"),
            "elapsed_sec":       best["elapsed_sec"],
        },
        "explanation": explanation,
        "learning_summary": learning_summary,
    }


def _json_safe(value: Any) -> Any:
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            return value.tolist()
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_json_safe)
    return path


_W = 100


def print_profile_summary(profile: DataProfile) -> None:
    print()
    print("  Dataset Profile")
    print("  " + "-" * 48)
    print(f"  Rows               : {profile.n_rows:,}")
    print(f"  Feature columns    : {profile.n_cols}  "
          f"(num={len(profile.num_cols)}, cat={len(profile.cat_cols)})")
    print(f"  Missing ratio      : {profile.total_missing_ratio:.2%}"
          + (f"  ({len(profile.high_missing_cols)} col(s) >50% missing)"
             if profile.high_missing_cols else ""))
    print(f"  Duplicate rows     : {profile.n_duplicates:,}")
    print(f"  Classes            : {profile.n_classes}  "
          f"(imbalance ratio = {profile.imbalance_ratio:.1f}x, "
          f"min class = {profile.min_class_size} samples)")
    print(f"  High-outlier cols  : {len(profile.high_outlier_cols)}"
          + (f"  {profile.high_outlier_cols}" if profile.high_outlier_cols else ""))
    print(f"  High-skew cols     : {len(profile.high_skew_cols)}"
          + (f"  {profile.high_skew_cols}" if profile.high_skew_cols else ""))
    print(f"  High-kurtosis cols : {len(profile.high_kurtosis_cols)}"
          + ("  (heavy tails)" if profile.high_kurtosis_cols else ""))
    print(f"  Sparse features    : {'yes' if profile.has_sparse_features else 'no'}")
    print(f"  Binary num cols    : {len(profile.binary_num_cols)}")
    print(f"  High-cardinality   : {len(profile.high_cardinality_cols)}")
    print(f"  Constant/near-cnst : "
          f"{len(profile.constant_cols) + len(profile.near_constant_cols)}")
    print(f"  High-corr pairs    : {profile.n_high_corr_pairs}"
          + ("  (multicollinearity)" if profile.has_multicollinearity else ""))


def print_final_summary(
    profile: DataProfile,
    results: List[Dict[str, Any]],
    best: Dict[str, Any],
    config: Config,
    meta_status: Optional[Dict[str, Any]] = None,
    mem_influence: Optional[Dict[str, Any]] = None,
    mem_update_outcome: Optional[str] = None,
) -> None:
    metric = best.get("selected_metric", config.metric)
    metric_names = valid_metrics_for_task(config.task_type)
    sorted_results = sorted(results, key=lambda r: -r.get("normalized_score", r.get("final_score", 0.0)))

    n_splits = best.get("n_splits", "?")
    n_models = best.get("n_models", 1)

    print()
    print("=" * _W)
    print(f"  PIPELINE EVALUATION RESULTS  "
          f"({n_splits}-fold CV x {n_models} models)")
    print("=" * _W)
    col    = "Score"
    header = f"  {'#':<3} {'Pipeline':<42} {col:>8}  {'inter-model std':>15}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rank, r in enumerate(sorted_results, 1):
        name  = r["spec"].name()
        if len(name) > 40:
            name = name[:37] + "..."
        score = r.get("normalized_score", r.get("final_score", 0.0))
        std   = r.get("final_score_std", r.get("metrics_std", {}).get(f"{metric}_std", 0.0))
        marker = " *" if r is best else ""
        print(f"  {rank:<3} {name:<42} {score:>8.4f}  {std:>15.4f}{marker}")

    print("=" * _W)
    print()
    m = best.get("raw_metrics", best["metrics"])
    s = best.get("metrics_std", {})
    print(f"  Best {metric_label(metric)} : {m[metric]:.4f} "
          f"(normalized {best.get('normalized_score', best.get('final_score', 0.0)):.4f}  |  inter-model std {s.get(metric + '_std', 0.0):.4f}  "
          f"over {n_splits} folds x {n_models} models)")
    print(
        "  All metrics : "
        + "  ".join(
            f"{metric_label(mk)}={m.get(mk, 0.0):.4f}" for mk in metric_names
        )
    )

    pmt = best.get("per_model_metrics", {})
    if pmt:
        print(f"  Per-model   : "
              + "  ".join(f"{mn}={mv.get(metric, 0):.4f}" for mn, mv in pmt.items()))
    if best.get("evaluation_mode"):
        print(f"  Eval mode   : {best['evaluation_mode']}")
    ed = best.get("evaluator_details") or {}
    if ed.get("model_family"):
        print(f"  Eval family : {ed['model_family']}")
    if ed.get("models"):
        print(f"  Models used : {', '.join(str(m) for m in ed['models'])}")
    if ed.get("baselines"):
        print(f"  Baselines   : {', '.join(str(b) for b in ed['baselines'])}")
    if best.get("evaluation_summary"):
        print(f"  Summary     : {best['evaluation_summary']}")

    tc = config.task_context()
    if any(tc.values()):
        print()
        print("  TASK CONTEXT")
        print("  " + "-" * 40)
        if tc.get("task_type"):
            print(f"  Task type   : {tc['task_type']}")
        if tc.get("supervision"):
            print(f"  Supervision : {tc['supervision']}")
        if tc.get("domain"):
            print(f"  Domain      : {tc['domain']}")
        if tc.get("fe_budget"):
            print(f"  FE budget   : {tc['fe_budget']}")
        if tc.get("data_quality"):
            print(f"  Data qual.  : {tc['data_quality']}")
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
        n_train   = meta_status.get("n_train", 0)
        weight    = meta_status.get("weight", 0.0)
        needed    = meta_status.get("min_to_use", 5)
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
