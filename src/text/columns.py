from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


REQUIRED_COLUMN_KEYS = {
    "classification_single": [("text", "text column"), ("label", "label column")],
    "classification_multi": [("text", "text column")],
    "ner": [("text", "text column"), ("entities", "entity annotations column")],
    "semantic_similarity": [],
    "summarization": [("source_text", "source text column"), ("summary", "reference summary column")],
    "question_answering": [("context", "context column"), ("question", "question column"), ("answer", "answer column")],
    "topic_modeling": [("text", "text column")],
}


_SUPPORTED_COL_KEYS = {
    "classification_single": {"text", "label"},
    "classification_multi": {"text", "labels"},
    "ner": {"text", "entities"},
    "semantic_similarity": {"text_a", "text_b", "similarity", "query", "document", "relevance"},
    "summarization": {"source_text", "summary"},
    "question_answering": {"context", "question", "answer", "answer_start"},
    "topic_modeling": {"text", "label"},
}


_TEXT_COL_KEYS = {
    "text", "tokens", "source_text", "context", "question", "prompt",
    "text_a", "text_b", "query", "document",
}


_ANNOTATION_COL_KEYS = {"entities"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def resolve_columns(df: pd.DataFrame, task_type: str, col_overrides: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    task = (task_type or "").strip().lower()
    c: Dict[str, object] = {}
    if not col_overrides:
        return c
    supported = _SUPPORTED_COL_KEYS.get(task, set())
    norm_map = {_norm(col): col for col in df.columns}
    for key, raw_value in col_overrides.items():
        if key == "binary_label_columns":
            if isinstance(raw_value, list):
                resolved = []
                for entry in raw_value:
                    name = str(entry).strip()
                    if not name:
                        continue
                    if name in df.columns:
                        resolved.append(name)
                    elif _norm(name) in norm_map:
                        resolved.append(norm_map[_norm(name)])
                if resolved:
                    c["binary_label_columns"] = resolved
            continue
        if key not in supported:
            continue
        name = str(raw_value or "").strip()
        if not name:
            continue
        if name in df.columns:
            c[key] = name
        elif _norm(name) in norm_map:
            c[key] = norm_map[_norm(name)]
    return c


def primary_text_column_keys(task_type: str) -> List[str]:
    task = (task_type or "").strip().lower()
    if task in {"classification_single", "classification_multi", "ner", "topic_modeling"}:
        return ["text"]
    if task == "semantic_similarity":
        return ["text_a", "text_b", "query", "document"]
    if task == "summarization":
        return ["source_text"]
    if task == "question_answering":
        return ["context", "question"]
    return []


def reserved_columns(cols: Dict[str, object]) -> List[str]:
    used = set()
    for key, value in cols.items():
        if key == "binary_label_columns":
            for c in value or []:
                used.add(c)
        elif isinstance(value, str):
            used.add(value)
    return sorted(used)


def _is_id_like(name: str) -> bool:
    n = _norm(name)
    return n in {"id", "row_id", "uuid", "guid", "index"} or n.endswith("_id") or n.startswith("id_")


def detect_auxiliary_features(
    df: pd.DataFrame,
    cols: Dict[str, object],
    explicit_aux: Optional[List[str]] = None,
    max_categorical_cardinality: int = 50,
) -> Tuple[List[str], List[str], List[str]]:
    used = set(reserved_columns(cols))
    norm_map = {_norm(col): col for col in df.columns}

    if explicit_aux:
        candidates: List[str] = []
        for raw in explicit_aux:
            name = str(raw).strip()
            if not name:
                continue
            actual = name if name in df.columns else norm_map.get(_norm(name))
            if actual and actual not in used and actual not in candidates:
                candidates.append(actual)
    else:
        candidates = [c for c in df.columns if c not in used and not _is_id_like(c)]

    numeric: List[str] = []
    categorical: List[str] = []
    skipped: List[str] = []
    for col in candidates:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric.append(col)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() >= 0.9 and not pd.api.types.is_datetime64_any_dtype(series):
            numeric.append(col)
            continue
        nunique = series.dropna().astype(str).nunique()
        if 1 <= nunique <= max_categorical_cardinality:
            categorical.append(col)
        else:
            skipped.append(col)
    return numeric, categorical, skipped
