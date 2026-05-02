import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import TextConfig, metric_label
from .memory_manager import mixed_feature_meta_features, text_meta_features
from .preprocessing import TextPipelineSpec
from .profiler import TextProfile

REPORTS_DIR = Path("reports")


def _profile_to_dict(profile: TextProfile) -> dict:
    ir = profile.imbalance_ratio
    return {
        "n_samples": profile.n_samples,
        "original_row_count": profile.original_row_count,
        "removed_empty_or_invalid_count": profile.removed_empty_or_invalid_count,
        "removed_non_english_count": profile.removed_non_english_count,
        "removed_language_uncertain_count": profile.removed_language_uncertain_count,
        "removed_too_noisy_count": profile.removed_too_noisy_count,
        "language_filter_method": profile.language_filter_method,
        "emoji_strategy": profile.emoji_strategy,
        "emoji_translated_count": profile.emoji_translated_count,
        "emoji_removed_count": profile.emoji_removed_count,
        "removed_excessive_emoji_count": profile.removed_excessive_emoji_count,
        "final_row_count": profile.n_samples,
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
        "imbalance_ratio": round(ir, 2) if math.isfinite(ir) else 999.9,
        "min_class_size": profile.min_class_size,
        "missing_target_count": profile.missing_target_count,
        "noise_counts": profile.noise_counts,
        "noise_ratios": profile.noise_ratios,
        "noise_ratio": round(profile.noise_ratio, 6),
        "annotation_validity": profile.annotation_validity,
        "source_target_length_ratio": round(profile.source_target_length_ratio, 6),
        "auxiliary_numeric_columns": profile.auxiliary_numeric_columns,
        "auxiliary_categorical_columns": profile.auxiliary_categorical_columns,
        "auxiliary_skipped_columns": profile.auxiliary_skipped_columns,
        "numeric_feature_profile": profile.numeric_feature_profile,
        "categorical_feature_profile": profile.categorical_feature_profile,
        "extra_feature_missing_ratio": round(profile.extra_feature_missing_ratio, 6),
        "text_to_tabular_feature_ratio": round(profile.text_to_tabular_feature_ratio, 6),
        "has_tabular_features": profile.has_tabular_features,
    }


def generate_explanation(profile: TextProfile, best: Dict[str, Any], metric: str, task_context: Optional[Dict[str, Any]] = None, meta_status: Optional[Dict[str, Any]] = None, mem_influence: Optional[Dict[str, Any]] = None) -> str:
    spec: TextPipelineSpec = best["spec"]
    selected_metric = best.get("selected_metric", metric)
    raw = best.get("raw_metrics", best["metrics"])
    normalized = best.get("normalized_score", best.get("final_score", 0.0))
    fusion_used = bool(best.get("evaluator_details", {}).get("fusion_used", False))
    lines = [
        f"The best text pipeline scored {raw.get(selected_metric, 0.0):.4f} {metric_label(selected_metric)} with a normalized score of {normalized:.4f}.",
        f"Evaluation mode: {best.get('evaluation_mode', 'unknown')}.",
        f"Best pipeline used: {'text + tabular features' if fusion_used else 'text only'}.",
    ]
    if best.get("evaluation_summary"):
        lines.append(best["evaluation_summary"])
    lines.append("")
    lines.append("Text preprocessing decisions and rationale:")
    lines.append(f"- Case handling: {'lowercase' if spec.lowercase else 'preserve case'}.")
    lines.append(f"- URL/email/HTML cleaning: {'enabled' if spec.clean_urls_emails_html else 'disabled'}.")
    lines.append(f"- Emoji handling: {spec.emoji_handling}.")
    lines.append(f"- Punctuation handling: {spec.punctuation_handling}.")
    lines.append(f"- Number normalization: {spec.number_normalization}.")
    lines.append(f"- Stopword removal: {'enabled' if spec.stopword_removal else 'disabled'}.")
    lines.append(f"- Tokenization: {spec.tokenization_strategy}; representation: {spec.representation}.")
    lines.append(f"- Max sequence length: {spec.max_sequence_length}.")
    if spec.imbalance != "none":
        lines.append(f"- Class imbalance handling: {spec.imbalance}.")
    if spec.fusion_strategy != "text_only":
        lines.append(f"- Fusion strategy: {spec.fusion_strategy} (numeric impute={spec.numeric_imputation}, scaling={spec.numeric_scaling}, cat encoding={spec.categorical_encoding}).")
    if best.get("evaluation_mode") == "fallback":
        lines.append("")
        lines.append("This run used an explicit lightweight fallback baseline because no heavier pretrained model is bundled for this task.")
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


def generate_report(profile: TextProfile, results: List[Dict[str, Any]], best: Dict[str, Any], config: TextConfig, meta_status: Optional[Dict[str, Any]] = None, mem_influence: Optional[Dict[str, Any]] = None, mem_update_outcome: Optional[str] = None) -> dict:
    tc = config.task_context()
    sorted_results = sorted(results, key=lambda result: -result.get("normalized_score", result.get("final_score", 0.0)))
    selected_metric = best.get("selected_metric", config.metric)
    profile_dict = _profile_to_dict(profile)
    learning_summary: Dict[str, Any] = {}
    if meta_status:
        learning_summary["meta_learner"] = meta_status
    if mem_influence:
        learning_summary["memory_influence"] = mem_influence
    if mem_update_outcome:
        learning_summary["memory_update"] = mem_update_outcome
    best_pipeline_dict = best["spec"].to_dict()
    fusion_used = bool(best.get("evaluator_details", {}).get("fusion_used", False))
    return {
        "timestamp": datetime.now().isoformat(),
        "modality": "Text",
        "config": {"data_path": str(config.data_path), "metric": config.metric},
        "task_context": tc,
        "profile_summary": profile_dict,
        "text_meta_features": text_meta_features(profile, config.task_type, selected_metric, best_pipeline_dict),
        "mixed_feature_meta_features": mixed_feature_meta_features(profile, config, best_pipeline_dict),
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
            "config": best_pipeline_dict,
            "selected_metric": selected_metric,
            "metrics": best["metrics"],
            "raw_metrics": best.get("raw_metrics", best["metrics"]),
            "metrics_std": best.get("metrics_std", {}),
            "normalized_metrics": best.get("normalized_metrics", {}),
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
            "fusion_used": fusion_used,
            "fusion_strategy": best_pipeline_dict.get("fusion_strategy", "text_only"),
        },
        "text_report_sections": {
            "english_only_support": "This run processed English-only text. Rows detected as non-English were removed before evaluation.",
            "selected_task_type": tc.get("task_type", ""),
            "selected_metric": selected_metric,
            "user_provided_columns": (config.col_overrides or {}),
            "dataset_overview": {"samples": profile.n_samples, "text_columns": profile.primary_text_columns, "target_columns": profile.target_columns},
            "row_filtering_summary": {
                "original_row_count": profile.original_row_count,
                "removed_empty_or_invalid": profile.removed_empty_or_invalid_count,
                "removed_non_english": profile.removed_non_english_count,
                "removed_language_uncertain": profile.removed_language_uncertain_count,
                "removed_too_noisy": profile.removed_too_noisy_count,
                "language_filter_method": profile.language_filter_method,
                "final_usable_rows": profile.n_samples,
            },
            "emoji_handling_summary": {
                "emoji_strategy": profile.emoji_strategy,
                "emoji_translated_count": profile.emoji_translated_count,
                "emoji_removed_count": profile.emoji_removed_count,
                "removed_excessive_emoji_count": profile.removed_excessive_emoji_count,
            },
            "column_validation_result": {
                "resolved_text_columns": profile.primary_text_columns,
                "resolved_target_columns": profile.target_columns,
                "all_resolved_columns": {k: v for k, v in profile.resolved_columns.items() if isinstance(v, str)},
            },
            "auxiliary_feature_summary": {
                "numeric_columns": profile.auxiliary_numeric_columns,
                "categorical_columns": profile.auxiliary_categorical_columns,
                "skipped_columns": profile.auxiliary_skipped_columns,
                "numeric_profile": profile.numeric_feature_profile,
                "categorical_profile": profile.categorical_feature_profile,
                "has_tabular_features": profile.has_tabular_features,
                "extra_feature_missing_ratio": round(profile.extra_feature_missing_ratio, 4),
                "text_to_tabular_feature_ratio": round(profile.text_to_tabular_feature_ratio, 4),
            },
            "noisy_text_handling_summary": {"noise_counts": profile.noise_counts, "noise_ratios": profile.noise_ratios, "overall_noise_ratio": round(profile.noise_ratio, 4)},
            "text_length_statistics": profile.text_length_distribution,
            "vocabulary_noise_summary": {"vocabulary_size_estimate": profile.vocabulary_size_estimate, "unique_token_ratio": profile.unique_token_ratio, "noise_counts": profile.noise_counts},
            "missing_empty_duplicate_counts": {"empty_texts": profile.n_empty_texts, "duplicate_texts": profile.duplicate_text_count, "missing_targets": profile.missing_target_count},
            "label_distribution": profile.label_distribution,
            "annotation_validation": profile.annotation_validity,
            "source_target_length_ratio": profile.source_target_length_ratio,
            "preprocessing_strategies_tested": [r["spec"].to_dict() for r in sorted_results],
            "best_selected_pipeline": best_pipeline_dict,
            "best_pipeline_used_features": "text + tabular features" if fusion_used else "text only",
            "task_specific_raw_metrics": best.get("raw_metrics", best["metrics"]),
            "normalized_score": best.get("normalized_score", best.get("final_score")),
            "evaluation_mode": best.get("evaluation_mode", ""),
        },
        "explanation": generate_explanation(profile, best, selected_metric, task_context=tc, meta_status=meta_status, mem_influence=mem_influence),
        "learning_summary": learning_summary,
    }


def save_report(report: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"report_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return path


def print_profile_summary(profile: TextProfile) -> None:
    ir = profile.imbalance_ratio
    ir_display = f"{ir:.1f}x" if math.isfinite(ir) else ">999x"
    print()
    print("  Text Dataset Profile  [English-only]")
    print("  " + "-" * 48)
    if profile.original_row_count and profile.original_row_count != profile.n_samples:
        print(f"  Original rows      : {profile.original_row_count:,}")
        print(f"  Removed (invalid)  : {profile.removed_empty_or_invalid_count:,}")
        print(f"  Removed (non-EN)   : {profile.removed_non_english_count:,}")
    print(f"  Usable rows        : {profile.n_samples:,}")
    print(f"  Text columns       : {profile.primary_text_columns}")
    print(f"  Target columns     : {profile.target_columns}")
    print(f"  Aux numeric cols   : {profile.auxiliary_numeric_columns}")
    print(f"  Aux categorical    : {profile.auxiliary_categorical_columns}")
    if profile.auxiliary_skipped_columns:
        print(f"  Skipped aux cols   : {profile.auxiliary_skipped_columns}")
    print(f"  Empty / duplicate  : {profile.n_empty_texts} / {profile.duplicate_text_count}")
    print(f"  Avg chars / tokens : {profile.avg_char_length:.1f} / {profile.avg_token_length:.1f}")
    print(f"  Length range       : {profile.min_char_length} to {profile.max_char_length} chars")
    print(f"  Vocabulary         : {profile.vocabulary_size_estimate:,}  (unique token ratio={profile.unique_token_ratio:.3f})")
    print(f"  Labels             : {profile.n_classes}  (imbalance ratio = {ir_display})")
    print(f"  Noise counts       : {profile.noise_counts}")
    print(f"  Annotation valid   : {profile.annotation_validity}")
