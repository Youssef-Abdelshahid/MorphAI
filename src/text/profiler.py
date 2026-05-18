from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .columns import detect_auxiliary_features, primary_text_column_keys, resolve_columns

try:
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 42
    _LANGDETECT_AVAILABLE = True
except Exception:
    _LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):
        pass

    def detect_langs(_text: str):
        raise LangDetectException("langdetect not available")


LANGUAGE_FILTER_METHOD = "langdetect+english_likeness" if _LANGDETECT_AVAILABLE else "english_likeness_heuristic"
LANG_EN_PROB_KEEP = 0.50
LANG_NONEN_PROB_REMOVE = 0.85
ENGLISH_LIKENESS_THRESHOLD = 0.45


@dataclass
class TextProfile:
    n_samples: int
    columns: List[str]
    task_type: str
    resolved_columns: Dict[str, object]
    primary_text_columns: List[str]
    target_columns: List[str]
    n_empty_texts: int
    duplicate_text_count: int
    avg_char_length: float
    avg_token_length: float
    min_char_length: int
    max_char_length: int
    char_length_std: float
    token_length_std: float
    text_length_distribution: Dict[str, int]
    vocabulary_size_estimate: int
    unique_token_ratio: float
    language_distribution: Dict[str, int]
    label_distribution: Dict[str, int]
    n_classes: int
    imbalance_ratio: float
    min_class_size: int
    missing_target_count: int
    noise_counts: Dict[str, int]
    noise_ratios: Dict[str, float]
    noise_ratio: float
    annotation_validity: Dict[str, object]
    source_target_length_ratio: float
    auxiliary_numeric_columns: List[str] = field(default_factory=list)
    auxiliary_categorical_columns: List[str] = field(default_factory=list)
    auxiliary_skipped_columns: List[str] = field(default_factory=list)
    numeric_feature_profile: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categorical_feature_profile: Dict[str, Dict[str, object]] = field(default_factory=dict)
    extra_feature_missing_ratio: float = 0.0
    text_to_tabular_feature_ratio: float = 0.0
    has_tabular_features: bool = False
    original_row_count: int = 0
    removed_empty_or_invalid_count: int = 0
    removed_non_english_count: int = 0
    removed_language_uncertain_count: int = 0
    removed_too_noisy_count: int = 0
    language_filter_method: str = ""
    emoji_strategy: str = ""
    emoji_translated_count: int = 0
    emoji_removed_count: int = 0
    removed_excessive_emoji_count: int = 0
    input_format: str = ""
    structure_profile: Dict[str, object] = field(default_factory=dict)
    parsing_summary: Dict[str, object] = field(default_factory=dict)
    parser_warnings: List[str] = field(default_factory=list)
    field_availability: Dict[str, bool] = field(default_factory=dict)


def _tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)


def _length_bucket(words: int) -> str:
    if words == 0:
        return "empty"
    if words < 10:
        return "<10 words"
    if words < 100:
        return "10-99 words"
    if words < 500:
        return "100-499 words"
    return "500+ words"


def _simple_language(text: str) -> str:
    s = str(text)
    if re.search(r"[؀-ۿ]", s):
        return "Arabic"
    if re.search(r"[一-鿿]", s):
        return "Chinese"
    latin = len(re.findall(r"[A-Za-z]", s))
    letters = len(re.findall(r"[^\W\d_]", s, flags=re.UNICODE))
    if letters and latin / max(letters, 1) > 0.7:
        return "Latin"
    return "Unknown"


def _parse_sequence(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    s = str(value)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    return [x.strip() for x in s.split() if x.strip()]


def _parse_entities(value) -> Tuple[bool, int]:
    if pd.isna(value):
        return False, 0
    s = str(value).strip()
    if not s:
        return False, 0
    try:
        parsed = json.loads(s)
    except Exception:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
    if isinstance(parsed, list):
        ok = 0
        for ent in parsed:
            if isinstance(ent, dict) and {"start", "end", "label"}.issubset(ent):
                ok += 1
            elif isinstance(ent, (list, tuple)) and len(ent) >= 3:
                ok += 1
        return ok == len(parsed), ok
    tags = s.split()
    if tags and all(tag == "O" or tag.startswith(("B-", "I-")) for tag in tags):
        return True, len([tag for tag in tags if tag.startswith("B-")])
    return False, 0


def annotation_validity_summary(df: pd.DataFrame, task_type: str, cols: Dict[str, object]) -> Dict[str, object]:
    task = (task_type or "").strip().lower()
    invalid = 0
    valid = 0
    details: Dict[str, object] = {}
    if task == "ner" and cols.get("entities"):
        entity_count = 0
        for value in df[cols["entities"]].fillna(""):
            ok, n = _parse_entities(value)
            valid += int(ok)
            invalid += int(not ok)
            entity_count += n
        details["entity_count"] = entity_count
    elif task == "pos" and cols.get("pos_tags"):
        token_col = cols.get("tokens") or cols.get("text")
        for _, row in df.iterrows():
            tags = _parse_sequence(row.get(cols["pos_tags"], ""))
            toks = _parse_sequence(row.get(token_col, "")) if token_col else []
            ok = bool(tags) and bool(toks) and len(tags) == len(toks)
            valid += int(ok)
            invalid += int(not ok)
    elif task == "question_answering" and cols.get("context") and cols.get("answer"):
        for _, row in df.iterrows():
            context = str(row.get(cols["context"], ""))
            answer = str(row.get(cols["answer"], ""))
            ok = bool(answer.strip()) and answer.lower() in context.lower()
            valid += int(ok)
            invalid += int(not ok)
    return {"valid_count": valid, "invalid_count": invalid, "valid_ratio": valid / max(valid + invalid, 1), **details}


def _label_counts(df: pd.DataFrame, task_type: str, cols: Dict[str, object]) -> Tuple[Dict[str, int], int]:
    task = (task_type or "").strip().lower()
    if task == "classification_multi":
        if cols.get("labels"):
            counter: Counter = Counter()
            missing = 0
            for value in df[cols["labels"]]:
                labels = _parse_sequence(value)
                if labels:
                    counter.update(labels)
                else:
                    missing += 1
            return dict(sorted(counter.items())), missing
        binary = cols.get("binary_label_columns") or []
        if binary:
            return ({col: int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).sum()) for col in binary},
                    int(df[binary].isna().all(axis=1).sum()))
    label_col = cols.get("label") or cols.get("relation") or cols.get("similarity")
    if label_col and label_col in df.columns:
        values = df[label_col]
        missing = int(values.isna().sum() + values.fillna("").astype(str).str.strip().eq("").sum())
        return {str(k): int(v) for k, v in values.dropna().astype(str).value_counts().sort_index().items()}, missing
    return {}, 0


def _primary_text_columns(cols: Dict[str, object]) -> List[str]:
    order = ["text", "tokens", "source_text", "context", "question", "prompt", "completion", "text_a", "text_b", "query", "document"]
    return [cols[k] for k in order if isinstance(cols.get(k), str)]


def _target_columns(cols: Dict[str, object]) -> List[str]:
    names = ["label", "labels", "relation", "similarity", "summary", "answer", "completion", "pos_tags", "entities"]
    out = [cols[k] for k in names if isinstance(cols.get(k), str)]
    out.extend(cols.get("binary_label_columns") or [])
    return list(dict.fromkeys(out))


_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "while", "with", "without",
    "in", "on", "at", "by", "for", "from", "of", "to", "into", "onto", "out",
    "over", "under", "up", "down", "off", "about", "above", "below", "after",
    "before", "between", "through", "during", "against", "than", "then",
    "as", "because", "so", "such", "just", "only", "also",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "hers", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "not", "no", "yes", "all", "any", "some", "many", "much", "more", "most",
    "each", "every", "either", "neither", "both", "another", "other",
    "here", "there", "now", "then", "today",
})

_ENGLISH_TRIGRAMS = frozenset({
    "the", "and", "ing", "ent", "ion", "tio", "for", "ati", "ter", "her",
    "tha", "ere", "ate", "his", "con", "res", "ver", "all", "ons", "nce",
    "men", "ith", "ted", "ers", "pro", "thi", "wit", "are", "ess", "not",
    "ive", "was", "ect", "rea", "com", "eve", "per", "int", "est", "sta",
    "cti", "ica", "ist", "ear", "ain", "one", "our", "iti", "rat", "ell",
    "you", "ave", "out", "use", "but", "edt", " th", "th ", "he ", " he",
    " an", "an ", " in", "in ", "ou ", " of", "of ", " to", "to ",
})


def _strip_noise_for_lang(text: str) -> str:
    s = re.sub(r"https?://\S+|www\.\S+", " ", text)
    s = re.sub(r"\b[\w.\-]+@[\w.\-]+\.\w+\b", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(
        r"[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]",
        " ",
        s,
    )
    return re.sub(r"\s+", " ", s).strip()


def english_likeness_score(text: str) -> float:
    if not text:
        return 0.0
    lower = text.lower()
    tokens = re.findall(r"[a-z]+", lower)
    if not tokens:
        return 0.0
    func_hits = sum(1 for t in tokens if t in _ENGLISH_FUNCTION_WORDS)
    func_ratio = func_hits / len(tokens)
    alpha_run = " ".join(tokens)
    trigrams = [alpha_run[i:i + 3] for i in range(len(alpha_run) - 2)] if len(alpha_run) >= 3 else []
    if trigrams:
        tri_ratio = sum(1 for tg in trigrams if tg in _ENGLISH_TRIGRAMS) / len(trigrams)
    else:
        tri_ratio = 0.0
    score = 0.55 * func_ratio + 0.45 * tri_ratio
    if func_hits >= 2:
        score = max(score, 0.55)
    if len(tokens) == 1 and tokens[0] in _ENGLISH_FUNCTION_WORDS:
        score = max(score, 0.6)
    return score


def _latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if ord(c) < 0x250) / len(letters)


def _has_alpha_after_strip(text) -> bool:
    s = "" if text is None else str(text)
    return bool(re.search(r"[A-Za-zÀ-ɏ]", _strip_noise_for_lang(s)))


def _meaningful_alpha_ratio(text) -> float:
    s = "" if text is None else str(text)
    if not s:
        return 0.0
    raw_chars = sum(1 for c in s if not c.isspace())
    if raw_chars == 0:
        return 0.0
    stripped = _strip_noise_for_lang(s)
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    return alpha_chars / raw_chars


def classify_english(text) -> str:
    if text is None:
        return "noisy"
    s = str(text)
    if not s.strip():
        return "noisy"
    stripped = _strip_noise_for_lang(s)
    if not re.search(r"[A-Za-zÀ-ɏ]", stripped):
        return "noisy"
    if _latin_ratio(stripped) < 0.7:
        return "non_english"
    short_tokens = re.findall(r"[A-Za-z]+", stripped.lower())
    if len(short_tokens) <= 2:
        if any(tok in _ENGLISH_FUNCTION_WORDS for tok in short_tokens):
            return "english"
        if len(short_tokens) == 1 and len(short_tokens[0]) >= 3 and short_tokens[0].isascii():
            return "english"
        score = english_likeness_score(stripped)
        if score >= ENGLISH_LIKENESS_THRESHOLD:
            return "english"
        return "non_english"
    if _LANGDETECT_AVAILABLE:
        try:
            langs = detect_langs(stripped)
        except LangDetectException:
            langs = []
        except Exception:
            langs = []
        if langs:
            top = langs[0]
            top_lang = getattr(top, "lang", "")
            top_prob = float(getattr(top, "prob", 0.0))
            if top_lang == "en" and top_prob >= LANG_EN_PROB_KEEP:
                return "english"
            en_prob = max((float(l.prob) for l in langs if l.lang == "en"), default=0.0)
            if top_lang != "en" and top_prob >= LANG_NONEN_PROB_REMOVE and en_prob < 0.30:
                return "non_english"
            if top_lang != "en" and en_prob >= 0.40:
                if english_likeness_score(stripped) >= ENGLISH_LIKENESS_THRESHOLD:
                    return "english"
                return "non_english"
            if english_likeness_score(stripped) >= ENGLISH_LIKENESS_THRESHOLD:
                return "english"
            return "uncertain"
    score = english_likeness_score(stripped)
    if score >= ENGLISH_LIKENESS_THRESHOLD + 0.1:
        return "english"
    if score < ENGLISH_LIKENESS_THRESHOLD - 0.15:
        return "non_english"
    return "uncertain"


def is_likely_english(text, threshold: float = 0.6) -> bool:
    return classify_english(text) == "english"


_EMOJI_COUNT_RE = re.compile(
    r"[\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF]"
)
_EMOJI_LONG_RUN_RE = re.compile(
    r"([\U0001F300-\U0001F9FF\U0001FA00-\U0001FAFF\U00002600-\U000027BF])\1{2,}"
)


def compute_emoji_stats(values, strategy: str) -> Dict[str, int]:
    strat = (strategy or "preserve").lower()
    total_emoji = 0
    excessive_rows = 0
    for v in values:
        s = "" if v is None else str(v)
        cnt = len(_EMOJI_COUNT_RE.findall(s))
        total_emoji += cnt
        if _EMOJI_LONG_RUN_RE.search(s):
            excessive_rows += 1
    translated = total_emoji if strat in ("translate_to_text", "describe", "limit_then_translate") else 0
    removed = total_emoji if strat == "remove" else 0
    return {
        "emoji_translated_count": translated,
        "emoji_removed_count": removed,
        "removed_excessive_emoji_count": excessive_rows if strat == "limit_then_translate" else 0,
    }


_ENGLISH_CHECK_KEYS = {
    "summarization": ["source_text", "summary"],
    "question_answering": ["context", "question", "answer"],
    "semantic_similarity": ["text_a", "text_b", "query", "document"],
}


def _primary_text_cols_for_task(task_type: str, cols: Dict[str, object]) -> List[str]:
    keys = primary_text_column_keys(task_type)
    out = []
    for k in keys:
        v = cols.get(k)
        if isinstance(v, str):
            out.append(v)
    return list(dict.fromkeys(out))


def _english_check_cols(task_type: str, cols: Dict[str, object]) -> List[str]:
    task = (task_type or "").strip().lower()
    keys = _ENGLISH_CHECK_KEYS.get(task) or primary_text_column_keys(task)
    out = []
    for k in keys:
        v = cols.get(k)
        if isinstance(v, str):
            out.append(v)
    return list(dict.fromkeys(out))


def remove_empty_and_nonenglish_rows(
    df: pd.DataFrame,
    cols: Dict[str, object],
    task_type: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    text_cols = _english_check_cols(task_type, cols)
    if not text_cols and isinstance(cols.get("text"), str):
        text_cols = [cols["text"]]

    n_original = len(df)
    counts = {
        "original": n_original,
        "removed_empty_or_invalid": 0,
        "removed_too_noisy": 0,
        "removed_non_english": 0,
        "removed_language_uncertain": 0,
        "final": n_original,
    }

    if not text_cols or n_original == 0:
        counts["final"] = len(df)
        return df.copy(), counts

    classifications: Dict[str, List[str]] = {}
    for col in text_cols:
        if col and col in df.columns:
            classifications[col] = [classify_english(v) for v in df[col].fillna("")]

    drop_reason: List[Optional[str]] = [None] * n_original
    for i in range(n_original):
        any_empty = False
        any_noisy = False
        any_non_english = False
        any_uncertain = False
        for col in text_cols:
            if col not in classifications:
                continue
            cls = classifications[col][i]
            value = "" if pd.isna(df[col].iloc[i]) else str(df[col].iloc[i])
            if not value.strip():
                any_empty = True
            elif cls == "noisy":
                any_noisy = True
            elif cls == "non_english":
                any_non_english = True
            elif cls == "uncertain":
                any_uncertain = True
        if any_empty:
            drop_reason[i] = "removed_empty_or_invalid"
        elif any_non_english:
            drop_reason[i] = "removed_non_english"
        elif any_uncertain:
            drop_reason[i] = "removed_language_uncertain"
        elif any_noisy:
            drop_reason[i] = "removed_too_noisy"

    keep_mask = pd.Series([r is None for r in drop_reason], index=df.index)
    for r in drop_reason:
        if r is not None:
            counts[r] += 1
    df_kept = df[keep_mask].copy()
    counts["final"] = len(df_kept)
    return df_kept, counts


def _profile_numeric(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col in cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if non_null.empty:
            out[col] = {"missing_ratio": 1.0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "skew": 0.0, "outlier_ratio": 0.0}
            continue
        q1, q3 = float(non_null.quantile(0.25)), float(non_null.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = float(((non_null < lo) | (non_null > hi)).mean())
        else:
            outliers = 0.0
        out[col] = {
            "missing_ratio": float(series.isna().mean()),
            "min": float(non_null.min()),
            "max": float(non_null.max()),
            "mean": float(non_null.mean()),
            "std": float(non_null.std(ddof=0)) if len(non_null) > 1 else 0.0,
            "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
            "outlier_ratio": outliers,
        }
    return out


def _profile_categorical(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for col in cols:
        series = df[col]
        non_null = series.dropna().astype(str)
        cardinality = int(non_null.nunique())
        top = non_null.value_counts().head(5).to_dict()
        out[col] = {
            "missing_ratio": float(series.isna().mean()),
            "cardinality": cardinality,
            "top_values": {str(k): int(v) for k, v in top.items()},
        }
    return out


def profile_text_dataset(
    df: pd.DataFrame,
    task_type: str,
    col_overrides: Optional[Dict[str, str]] = None,
    auxiliary_feature_columns: Optional[List[str]] = None,
    original_row_count: int = 0,
    removed_empty_or_invalid_count: int = 0,
    removed_non_english_count: int = 0,
    removed_language_uncertain_count: int = 0,
    removed_too_noisy_count: int = 0,
    language_filter_method: str = "",
    emoji_strategy: str = "",
    emoji_translated_count: int = 0,
    emoji_removed_count: int = 0,
    removed_excessive_emoji_count: int = 0,
) -> TextProfile:
    cols = resolve_columns(df, task_type, col_overrides=col_overrides)
    if isinstance(col_overrides, dict) and isinstance(col_overrides.get("binary_label_columns"), list):
        binary = [c for c in col_overrides["binary_label_columns"] if c in df.columns]
        if binary:
            cols["binary_label_columns"] = binary

    text_cols = _primary_text_columns(cols)
    primary = text_cols[0] if text_cols else df.columns[0]
    texts = df[primary].fillna("").astype(str)
    char_lengths = texts.str.len().to_numpy(dtype=float)
    token_lists = [_tokens(t) for t in texts]
    token_lengths = np.asarray([len(t) for t in token_lists], dtype=float)
    all_tokens = [tok for toks in token_lists for tok in toks]
    vocab = set(all_tokens)
    noise_patterns = {
        "urls": r"https?://|www\.",
        "emails": r"\b[\w\.\-]+@[\w\.\-]+\.\w+\b",
        "html": r"<[^>]+>",
        "emojis": r"[\U00010000-\U0010ffff]",
        "excessive_punctuation": r"([!?.,])\1{2,}",
        "numbers": r"\d",
        "all_caps": r"\b[A-Z]{4,}\b",
        "mentions": r"@\w+",
        "hashtags": r"#\w+",
    }
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="This pattern is interpreted as a regular expression")
        noise_counts = {name: int(texts.str.contains(pattern, regex=True, na=False).sum()) for name, pattern in noise_patterns.items()}
    noise_ratios = {name: count / max(len(df), 1) for name, count in noise_counts.items()}
    label_distribution, missing_targets = _label_counts(df, task_type, cols)
    counts = sorted(label_distribution.values(), reverse=True)
    imbalance = counts[0] / counts[-1] if counts and counts[-1] > 0 else float("inf") if counts else 1.0
    source_target_ratio = 0.0
    source_col = cols.get("source_text") or cols.get("prompt")
    target_col = cols.get("summary") or cols.get("completion")
    if source_col and target_col:
        source_lens = df[source_col].fillna("").astype(str).map(lambda x: max(len(_tokens(x)), 1))
        target_lens = df[target_col].fillna("").astype(str).map(lambda x: len(_tokens(x)))
        source_target_ratio = float((target_lens / source_lens).replace([np.inf, -np.inf], 0).mean())
    annotation = annotation_validity_summary(df, task_type, cols)

    aux_numeric, aux_categorical, aux_skipped = detect_auxiliary_features(df, cols, explicit_aux=auxiliary_feature_columns)
    numeric_profile = _profile_numeric(df, aux_numeric)
    categorical_profile = _profile_categorical(df, aux_categorical)
    aux_total = len(aux_numeric) + len(aux_categorical)
    extra_missing = 0.0
    if aux_total:
        ratios = [v.get("missing_ratio", 0.0) for v in numeric_profile.values()] + [v.get("missing_ratio", 0.0) for v in categorical_profile.values()]
        extra_missing = float(np.mean(ratios)) if ratios else 0.0
    n_text_cols = len(text_cols)
    text_to_tab_ratio = float(n_text_cols / max(aux_total, 1)) if aux_total else float(n_text_cols)
    return TextProfile(
        n_samples=int(len(df)),
        columns=[str(c) for c in df.columns],
        task_type=task_type,
        resolved_columns=cols,
        primary_text_columns=list(dict.fromkeys(text_cols)),
        target_columns=_target_columns(cols),
        n_empty_texts=int(texts.str.strip().eq("").sum()),
        duplicate_text_count=int(texts.duplicated().sum()),
        avg_char_length=float(np.mean(char_lengths)) if len(char_lengths) else 0.0,
        avg_token_length=float(np.mean(token_lengths)) if len(token_lengths) else 0.0,
        min_char_length=int(np.min(char_lengths)) if len(char_lengths) else 0,
        max_char_length=int(np.max(char_lengths)) if len(char_lengths) else 0,
        char_length_std=float(np.std(char_lengths)) if len(char_lengths) else 0.0,
        token_length_std=float(np.std(token_lengths)) if len(token_lengths) else 0.0,
        text_length_distribution=dict(Counter(_length_bucket(int(v)) for v in token_lengths)),
        vocabulary_size_estimate=len(vocab),
        unique_token_ratio=len(vocab) / max(len(all_tokens), 1),
        language_distribution=dict(Counter(_simple_language(t) for t in texts)),
        label_distribution=label_distribution,
        n_classes=len(label_distribution),
        imbalance_ratio=imbalance,
        min_class_size=counts[-1] if counts else 0,
        missing_target_count=missing_targets,
        noise_counts=noise_counts,
        noise_ratios=noise_ratios,
        noise_ratio=sum(1 for text in texts if any(re.search(p, text) for p in noise_patterns.values())) / max(len(texts), 1),
        annotation_validity=annotation,
        source_target_length_ratio=source_target_ratio,
        auxiliary_numeric_columns=aux_numeric,
        auxiliary_categorical_columns=aux_categorical,
        auxiliary_skipped_columns=aux_skipped,
        numeric_feature_profile=numeric_profile,
        categorical_feature_profile=categorical_profile,
        extra_feature_missing_ratio=extra_missing,
        text_to_tabular_feature_ratio=text_to_tab_ratio,
        has_tabular_features=bool(aux_numeric or aux_categorical),
        original_row_count=original_row_count or int(len(df)),
        removed_empty_or_invalid_count=removed_empty_or_invalid_count,
        removed_non_english_count=removed_non_english_count,
        removed_language_uncertain_count=removed_language_uncertain_count,
        removed_too_noisy_count=removed_too_noisy_count,
        language_filter_method=language_filter_method or LANGUAGE_FILTER_METHOD,
        emoji_strategy=emoji_strategy,
        emoji_translated_count=emoji_translated_count,
        emoji_removed_count=emoji_removed_count,
        removed_excessive_emoji_count=removed_excessive_emoji_count,
    )
