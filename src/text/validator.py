from pathlib import Path
from typing import Dict, List

import pandas as pd

from .columns import REQUIRED_COLUMN_KEYS, _SUPPORTED_COL_KEYS, resolve_columns
from .config import SUPPORTED_TASK_TYPES, VALID_TASK_TYPES, default_metric_for_task, normalize_task_type, valid_metrics_for_task
from .profiler import annotation_validity_summary

SUPPORTED_TEXT_EXTENSIONS = {".csv", ".xlsx", ".xls"}

_TASK_FRIENDLY = {
    "classification_single": "text classification (single-label)",
    "classification_multi": "text classification (multi-label)",
    "ner": "named entity recognition",
    "pos": "part-of-speech tagging",
    "relation_extraction": "relation extraction",
    "semantic_similarity": "semantic similarity / search",
    "summarization": "text summarization",
    "question_answering": "question answering",
    "text_generation": "text generation",
    "topic_modeling": "topic modeling",
}

_FIELD_LABELS = {
    "text": "text column",
    "tokens": "tokens column",
    "source_text": "source text column",
    "context": "context column",
    "question": "question column",
    "prompt": "prompt column",
    "completion": "completion / reference column",
    "summary": "reference summary column",
    "answer": "answer column",
    "answer_start": "answer_start column",
    "label": "label column",
    "labels": "labels column",
    "entities": "entity annotations column",
    "pos_tags": "POS tags column",
    "entity1": "entity 1 column",
    "entity2": "entity 2 column",
    "relation": "relation label column",
    "similarity": "similarity score column",
    "text_a": "text A column",
    "text_b": "text B column",
    "query": "query column",
    "document": "document column",
    "relevance": "relevance label/score column",
}


def validate_text_file(path: Path) -> List[str]:
    errors = []
    if not path.exists():
        return [f"File does not exist: {path}"]
    if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        errors.append(
            "The text modality requires a structured CSV or Excel file (.csv, .xlsx, .xls). "
            "Please upload a file with one text sample per row."
        )
    return errors


def load_text_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported text dataset format. Use CSV or Excel.")


def _col_hint(df: pd.DataFrame) -> str:
    sample = ", ".join(f"'{c}'" for c in list(df.columns)[:8])
    more = f" (and {len(df.columns) - 8} more)" if len(df.columns) > 8 else ""
    return f"Available columns: {sample}{more}."


def _missing_required(task: str, col_overrides: Dict[str, str]) -> List[str]:
    required = REQUIRED_COLUMN_KEYS.get(task, [])
    missing = []
    for key, label in required:
        value = (col_overrides.get(key) or "").strip() if isinstance(col_overrides.get(key), str) else ""
        if not value:
            missing.append((key, label))
    return missing


def validate_text_run(config, df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()
    if df.empty:
        return ["The uploaded text dataset is empty — no rows found."]
    if not task_type:
        errors.append("A text task type must be selected before running.")
        return errors
    if task_type not in VALID_TASK_TYPES:
        errors.append(
            f"Task type '{config.task_type}' is not supported for text data. "
            f"Supported tasks: {', '.join(sorted(SUPPORTED_TASK_TYPES))}."
        )
        return errors
    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics and (not metric or metric not in valid_metrics):
        config.metric = default_metric_for_task(task_type)

    col_overrides = config.col_overrides or {}
    task_label = _TASK_FRIENDLY.get(task_type, task_type)
    col_hint = _col_hint(df)

    missing = _missing_required(task_type, col_overrides)
    for key, label in missing:
        errors.append(f"The selected task ({task_label}) requires {label}. Please enter the column name before starting.")

    if task_type == "classification_multi":
        fmt = (getattr(config, "multilabel_format", "single_column") or "single_column").lower()
        if fmt == "single_column":
            if not (col_overrides.get("labels") or "").strip():
                errors.append(f"The selected task ({task_label}) requires labels column. Please enter the column name before starting.")
        else:
            binary = list(getattr(config, "binary_label_columns", []) or [])
            if not binary:
                errors.append(f"The selected task ({task_label}) requires at least one binary label column name. Please enter the column names before starting.")
    elif task_type == "pos":
        text_val = (col_overrides.get("text") or "").strip()
        tokens_val = (col_overrides.get("tokens") or "").strip()
        if not text_val and not tokens_val:
            errors.append(f"The selected task ({task_label}) requires tokens column or text column. Please enter the column name before starting.")
    elif task_type == "semantic_similarity":
        pair_present = all((col_overrides.get(k) or "").strip() for k in ("text_a", "text_b", "similarity"))
        if not pair_present:
            errors.append(
                f"The selected task ({task_label}) requires text A column, text B column, and similarity score column. "
                "Please enter the column names before starting."
            )

    extra_supported = _SUPPORTED_COL_KEYS.get(task_type, set())
    for key, value in col_overrides.items():
        if key == "binary_label_columns":
            continue
        if key not in extra_supported:
            continue
        name = (value or "").strip() if isinstance(value, str) else ""
        if name and name not in df.columns:
            errors.append(f"Column '{name}' was not found in the uploaded dataset. {col_hint}")

    if task_type == "classification_multi":
        for name in list(getattr(config, "binary_label_columns", []) or []):
            n = (name or "").strip()
            if n and n not in df.columns:
                errors.append(f"Column '{n}' was not found in the uploaded dataset. {col_hint}")

    if errors:
        return errors

    cols = resolve_columns(df, task_type, col_overrides=col_overrides)
    if task_type == "classification_multi":
        binary = list(getattr(config, "binary_label_columns", []) or [])
        if binary:
            cols["binary_label_columns"] = [b for b in binary if b in df.columns]

    text_keys = {"text", "tokens", "source_text", "context", "question", "prompt", "text_a", "text_b", "query", "document"}
    for key in text_keys:
        col = cols.get(key)
        if isinstance(col, str) and col in df.columns and df[col].fillna("").astype(str).str.strip().eq("").all():
            errors.append(f"The column '{col}' appears to be entirely empty — no usable text found.")

    if not errors and task_type in {"ner", "pos", "question_answering", "relation_extraction"}:
        validity = annotation_validity_summary(df, task_type, cols)
        if validity.get("invalid_count", 0) > 0:
            errors.append(
                f"{validity.get('invalid_count')} row(s) contain invalid or unparseable annotations for {task_label}. "
                "Check that annotations are formatted as JSON lists or BIO tag sequences."
            )
    return errors
