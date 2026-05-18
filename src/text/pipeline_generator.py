from typing import Any, Dict, List, Optional, Tuple

from .config import TABULAR_FUSION_COMPATIBLE
from .preprocessing import TextPipelineSpec
from .profiler import TextProfile

_MAX_PIPELINES = 14


def _deduplicate(pipelines: List[TextPipelineSpec]) -> List[TextPipelineSpec]:
    seen = set()
    unique = []
    for spec in pipelines:
        key = str(sorted(spec.to_dict().items()))
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def _apply_constraints(spec: TextPipelineSpec, constraints: List[str]) -> TextPipelineSpec:
    d = spec.to_dict()
    if "no_stemming" in constraints and d.get("normalization_strategy") == "stem":
        d["normalization_strategy"] = "none"
    if "no_lemmatization" in constraints and d.get("normalization_strategy") == "lemma":
        d["normalization_strategy"] = "none"
    if "no_stopword_removal" in constraints:
        d["stopword_removal"] = False
    if "preserve_case" in constraints:
        d["lowercase"] = False
    if "no_truncation" in constraints:
        d["max_sequence_length"] = 100000
    if "keep_punctuation" in constraints:
        d["punctuation_handling"] = "keep"
    if "keep_whitespace" in constraints:
        d["whitespace_normalization"] = False
    return TextPipelineSpec.from_dict(d)


def _matches_bad_pattern(spec: TextPipelineSpec, bad_specs: List[TextPipelineSpec]) -> bool:
    for bad in bad_specs:
        if (spec.representation == bad.representation
                and spec.lowercase == bad.lowercase
                and spec.stopword_removal == bad.stopword_removal
                and spec.punctuation_handling == bad.punctuation_handling
                and spec.fusion_strategy == bad.fusion_strategy):
            return True
    return False


def _make_spec(
    lowercase, clean, emoji, punct, num, ws, stop, norm, tok, maxlen, mindf, repr_, imb,
    fusion="text_only", num_imp="mean", num_sc="standard", cat_enc="onehot",
) -> TextPipelineSpec:
    return TextPipelineSpec(
        lowercase=lowercase, clean_urls_emails_html=clean, emoji_handling=emoji,
        punctuation_handling=punct, number_normalization=num, whitespace_normalization=ws,
        stopword_removal=stop, normalization_strategy=norm, tokenization_strategy=tok,
        max_sequence_length=maxlen, min_df=mindf, representation=repr_, imbalance=imb,
        fusion_strategy=fusion, numeric_imputation=num_imp, numeric_scaling=num_sc,
        categorical_encoding=cat_enc,
    )


def generate_pipelines(
    profile: TextProfile,
    good_cases: Optional[List[Dict[str, Any]]] = None,
    bad_cases: Optional[List[Dict[str, Any]]] = None,
    meta_learner: Any = None,
    task_context: Optional[Dict[str, Any]] = None,
    profile_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[List[TextPipelineSpec], List[str]]:
    candidates: List[TextPipelineSpec] = []
    messages: List[str] = []
    tc = task_context or {}
    constraints = tc.get("active_constraints") or []
    task_type = tc.get("task_type", profile.task_type)
    sequence_labeling = task_type in {"ner"}
    seq2seq = task_type in {"summarization", "question_answering", "semantic_similarity"}
    classification_like = task_type in {"classification_single", "classification_multi"}
    noisy = profile.noise_ratio > 0.1
    long_text = profile.avg_token_length > 300
    imbalance = "class_weight" if classification_like and profile.imbalance_ratio > 1.5 else "none"

    fusion_eligible = task_type in TABULAR_FUSION_COMPATIBLE and profile.has_tabular_features and not sequence_labeling

    if sequence_labeling:
        default_emoji = "preserve"
    elif seq2seq:
        default_emoji = "preserve"
    elif task_type == "topic_modeling":
        default_emoji = "remove"
    else:
        default_emoji = "limit_then_translate"

    baseline = _make_spec(
        False if sequence_labeling else True, noisy,
        default_emoji, "keep", "keep", True, False, "none", "word",
        512 if long_text else 256, 1, "tfidf_word", imbalance,
    )
    candidates.append(baseline)
    candidates.append(_make_spec(
        False if sequence_labeling else True, noisy,
        "remove" if task_type == "topic_modeling" else default_emoji,
        "remove" if classification_like else "keep",
        "replace" if classification_like else "keep",
        True, classification_like and not seq2seq, "none", "word", 256, 1, "tfidf_word", imbalance,
    ))
    candidates.append(_make_spec(False, noisy, "preserve" if sequence_labeling else "keep", "keep", "keep", True, False, "none", "word", 512, 1, "tfidf_char", imbalance))
    candidates.append(_make_spec(False if sequence_labeling else True, True, "translate_to_text" if not sequence_labeling else "preserve", "space", "replace", True, False, "none", "word", 512, 2, "tfidf_word", imbalance))
    candidates.append(_make_spec(False, False, "keep", "keep", "keep", True, False, "none", "word", 100000 if seq2seq else 512, 1, "raw_text", "none"))
    if classification_like or task_type == "topic_modeling":
        candidates.append(_make_spec(True, True, "remove", "remove", "replace", True, True, "stem", "word", 256, 2, "tfidf_word", imbalance))
        candidates.append(_make_spec(True, True, "remove", "keep", "replace", True, False, "none", "char_word", 256, 2, "tfidf_char_word", imbalance))
    if task_type == "topic_modeling":
        candidates.append(_make_spec(True, True, "remove", "remove", "replace", True, True, "none", "word", 512, 2, "tfidf_word", "none"))
    if sequence_labeling:
        candidates.append(_make_spec(False, False, "keep", "keep", "keep", False, False, "none", "pretokenized", 100000, 1, "token_sequence", "none"))
    if seq2seq:
        candidates.append(_make_spec(False, noisy, "keep", "keep", "keep", True, False, "none", "word", 100000, 1, "raw_text", "none"))

    if fusion_eligible:
        fusion_variants: List[TextPipelineSpec] = []
        for src in [baseline, candidates[1]] if len(candidates) > 1 else [baseline]:
            d = src.to_dict()
            d["fusion_strategy"] = "concatenate_text_tabular"
            d["numeric_imputation"] = "mean"
            d["numeric_scaling"] = "standard"
            d["categorical_encoding"] = "onehot"
            fusion_variants.append(TextPipelineSpec.from_dict(d))
        d2 = baseline.to_dict()
        d2.update({
            "fusion_strategy": "concatenate_text_tabular",
            "numeric_imputation": "median",
            "numeric_scaling": "robust",
            "categorical_encoding": "ordinal",
        })
        fusion_variants.append(TextPipelineSpec.from_dict(d2))
        candidates.extend(fusion_variants)
        messages.append(f"Generated {len(fusion_variants)} text+tabular fusion candidate(s) using detected auxiliary features.")

    if bad_cases:
        bad_specs = []
        for case in bad_cases:
            d = case.get("best_pipeline")
            if d:
                try:
                    bad_specs.append(TextPipelineSpec.from_dict(d))
                except Exception:
                    pass
        if bad_specs:
            rest = candidates[1:]
            filtered = [c for c in rest if not _matches_bad_pattern(c, bad_specs)]
            skipped = len(rest) - len(filtered)
            candidates = [baseline] + filtered
            if skipped:
                messages.append(f"Memory (avoidance): skipped {skipped} candidate(s) matching poor past text pipeline pattern(s).")
    if good_cases:
        injected = 0
        for case in good_cases[:3]:
            d = case.get("best_pipeline")
            if d:
                try:
                    candidates.append(TextPipelineSpec.from_dict(d))
                    injected += 1
                except Exception:
                    pass
        if injected:
            messages.append(f"Memory (positive): injected {injected} pipeline(s) from good similar text run(s).")
    if constraints:
        candidates = [_apply_constraints(c, constraints) for c in candidates]
        messages.append("Constraints applied to text candidates.")
    candidates = _deduplicate(candidates)[:_MAX_PIPELINES]
    if meta_learner is not None and task_context is not None and profile_summary is not None:
        try:
            reordered, ml_msgs = meta_learner.rank_candidates(candidates, task_context, profile_summary)
            if len(reordered) == len(candidates):
                candidates = reordered
                messages.extend(ml_msgs)
        except Exception:
            pass
    return candidates, messages
