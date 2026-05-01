from __future__ import annotations

import ast
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .columns import resolve_columns


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
    original_row_count: int = 0
    removed_empty_or_invalid_count: int = 0
    removed_non_english_count: int = 0


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
    details = {}
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
    elif task == "relation_extraction":
        needed = [cols.get("text"), cols.get("entity1"), cols.get("entity2"), cols.get("relation")]
        for _, row in df.iterrows():
            ok = all(str(row.get(col, "")).strip() for col in needed if col)
            valid += int(ok)
            invalid += int(not ok)
    return {"valid_count": valid, "invalid_count": invalid, "valid_ratio": valid / max(valid + invalid, 1), **details}


def _label_counts(df: pd.DataFrame, task_type: str, cols: Dict[str, object]) -> Tuple[Dict[str, int], int]:
    task = (task_type or "").strip().lower()
    if task == "classification_multi":
        if cols.get("labels"):
            counter = Counter()
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
            return {col: int(pd.to_numeric(df[col], errors="coerce").fillna(0).astype(bool).sum()) for col in binary}, int(df[binary].isna().all(axis=1).sum())
    label_col = cols.get("label") or cols.get("language_label") or cols.get("relation") or cols.get("similarity")
    if label_col and label_col in df.columns:
        values = df[label_col]
        missing = int(values.isna().sum() + values.fillna("").astype(str).str.strip().eq("").sum())
        return {str(k): int(v) for k, v in values.dropna().astype(str).value_counts().sort_index().items()}, missing
    return {}, 0


def _primary_text_columns(cols: Dict[str, object]) -> List[str]:
    order = ["text", "tokens", "source_text", "target_text", "context", "question", "prompt", "completion", "text_a", "text_b", "query", "document"]
    return [cols[k] for k in order if isinstance(cols.get(k), str)]


def _target_columns(cols: Dict[str, object]) -> List[str]:
    names = ["label", "labels", "language_label", "relation", "similarity", "summary", "target_text", "answer", "completion", "pos_tags", "entities"]
    out = [cols[k] for k in names if isinstance(cols.get(k), str)]
    out.extend(cols.get("binary_label_columns") or [])
    return list(dict.fromkeys(out))


def is_likely_english(text: str, threshold: float = 0.7) -> bool:
    if not text or not text.strip():
        return True
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    latin_count = sum(1 for c in letters if ord(c) < 0x250)
    return (latin_count / len(letters)) >= threshold


def _primary_text_cols_for_task(task_type: str, cols: Dict[str, object]) -> List[str]:
    task = (task_type or "").strip().lower()
    candidates = []
    if task in {"classification_single", "classification_multi", "ner", "pos", "relation_extraction", "topic_modeling", "language_detection"}:
        if cols.get("text"):
            candidates.append(cols["text"])
    elif task == "semantic_similarity":
        for k in ("text_a", "text_b", "query", "document"):
            if cols.get(k):
                candidates.append(cols[k])
    elif task == "summarization":
        if cols.get("source_text"):
            candidates.append(cols["source_text"])
    elif task == "machine_translation":
        if cols.get("source_text"):
            candidates.append(cols["source_text"])
    elif task == "question_answering":
        for k in ("context", "question"):
            if cols.get(k):
                candidates.append(cols[k])
    elif task == "text_generation":
        if cols.get("prompt"):
            candidates.append(cols["prompt"])
    return [c for c in dict.fromkeys(candidates) if isinstance(c, str)]


def remove_empty_and_nonenglish_rows(
    df: pd.DataFrame,
    cols: Dict[str, object],
    task_type: str,
) -> Tuple[pd.DataFrame, int, int]:
    text_cols = _primary_text_cols_for_task(task_type, cols)
    if not text_cols:
        text_cols = [cols.get("text")] if cols.get("text") else []

    n_original = len(df)
    valid_mask = pd.Series([True] * n_original, index=df.index)
    for col in text_cols:
        if col and col in df.columns:
            col_empty = df[col].fillna("").astype(str).str.strip().eq("")
            valid_mask &= ~col_empty
    df_valid = df[valid_mask].copy()
    n_removed_invalid = n_original - len(df_valid)

    if df_valid.empty:
        return df_valid, n_removed_invalid, 0

    english_mask = pd.Series([True] * len(df_valid), index=df_valid.index)
    for col in text_cols:
        if col and col in df_valid.columns:
            col_english = df_valid[col].fillna("").astype(str).map(is_likely_english)
            english_mask &= col_english
    df_english = df_valid[english_mask].copy()
    n_removed_nonenglish = len(df_valid) - len(df_english)

    return df_english, n_removed_invalid, n_removed_nonenglish


def profile_text_dataset(
    df: pd.DataFrame,
    task_type: str,
    col_overrides: Optional[Dict[str, str]] = None,
    original_row_count: int = 0,
    removed_empty_or_invalid_count: int = 0,
    removed_non_english_count: int = 0,
) -> TextProfile:
    cols = resolve_columns(df, task_type, col_overrides=col_overrides)
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
    noise_counts = {name: int(texts.str.contains(pattern, regex=True, na=False).sum()) for name, pattern in noise_patterns.items()}
    noise_ratios = {name: count / max(len(df), 1) for name, count in noise_counts.items()}
    label_distribution, missing_targets = _label_counts(df, task_type, cols)
    counts = sorted(label_distribution.values(), reverse=True)
    imbalance = counts[0] / counts[-1] if counts and counts[-1] > 0 else float("inf") if counts else 1.0
    source_target_ratio = 0.0
    source_col = cols.get("source_text") or cols.get("prompt")
    target_col = cols.get("summary") or cols.get("target_text") or cols.get("completion")
    if source_col and target_col:
        source_lens = df[source_col].fillna("").astype(str).map(lambda x: max(len(_tokens(x)), 1))
        target_lens = df[target_col].fillna("").astype(str).map(lambda x: len(_tokens(x)))
        source_target_ratio = float((target_lens / source_lens).replace([np.inf, -np.inf], 0).mean())
    annotation = annotation_validity_summary(df, task_type, cols)
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
        original_row_count=original_row_count or int(len(df)),
        removed_empty_or_invalid_count=removed_empty_or_invalid_count,
        removed_non_english_count=removed_non_english_count,
    )
