from pathlib import Path
from typing import List

import pandas as pd

from .columns import resolve_columns
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
    "machine_translation": "machine translation",
    "question_answering": "question answering",
    "text_generation": "text generation",
    "topic_modeling": "topic modeling",
    "language_detection": "language detection",
}

_TASK_REQUIRED_COL_HINTS = {
    "classification_single": "a text column (e.g. 'text', 'sentence', 'review') and a label column (e.g. 'label', 'class', 'category')",
    "classification_multi": "a text column and a labels column or multiple binary label columns (0/1 per class)",
    "ner": "a text column and an entity annotations column (e.g. 'entities', 'bio_tags')",
    "pos": "a tokens or text column and a POS tags column (e.g. 'pos_tags', 'tags')",
    "relation_extraction": "a text column, two entity columns (entity1/entity2), and a relation label column",
    "semantic_similarity": "text_a + text_b + similarity score columns, or query + document + relevance columns",
    "summarization": "a source text column (e.g. 'source_text', 'article') and a reference summary column (e.g. 'summary')",
    "machine_translation": "a source text column and a target translation column",
    "question_answering": "context, question, and answer columns",
    "text_generation": "a prompt column and a completion/target column",
    "topic_modeling": "a text column",
    "language_detection": "a text column and a language label column (e.g. 'language', 'lang')",
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


def _missing(cols: dict, names: list) -> List[str]:
    return [name for name in names if not cols.get(name)]


def _col_hint(df: pd.DataFrame) -> str:
    sample = ", ".join(f"'{c}'" for c in list(df.columns)[:8])
    more = f" (and {len(df.columns) - 8} more)" if len(df.columns) > 8 else ""
    return f"Available columns: {sample}{more}."


def validate_text_run(config, df: pd.DataFrame) -> List[str]:
    errors = []
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
    if valid_metrics and metric and metric not in valid_metrics:
        config.metric = default_metric_for_task(task_type)
    if valid_metrics and not metric:
        config.metric = default_metric_for_task(task_type)
    col_overrides = config.col_overrides or {}
    for key, col_name in col_overrides.items():
        if col_name and col_name not in df.columns:
            errors.append(
                f"The specified column '{col_name}' for '{key}' was not found in the dataset. "
                + _col_hint(df)
            )
    if errors:
        return errors
    cols = resolve_columns(df, task_type, col_overrides=col_overrides)
    task_label = _TASK_FRIENDLY.get(task_type, task_type)
    col_hint = _col_hint(df)
    required_hint = _TASK_REQUIRED_COL_HINTS.get(task_type, "")
    required = {
        "classification_single": ["text", "label"],
        "ner": ["text", "entities"],
        "relation_extraction": ["text", "entity1", "entity2", "relation"],
        "summarization": ["source_text", "summary"],
        "machine_translation": ["source_text", "target_text"],
        "question_answering": ["context", "question", "answer"],
        "text_generation": ["prompt", "completion"],
        "language_detection": ["text", "language_label"],
    }
    if task_type in required:
        miss = _missing(cols, required[task_type])
        if miss:
            errors.append(
                f"For {task_label}, the dataset must include {required_hint}. "
                f"Could not locate: {', '.join(miss)}. "
                + col_hint
                + " Use the column mapping fields to specify the exact column names."
            )
    elif task_type == "classification_multi":
        if not cols.get("text"):
            errors.append(
                f"For {task_label}, a text column is required. "
                + col_hint
            )
        if not cols.get("labels") and not cols.get("binary_label_columns"):
            errors.append(
                f"For {task_label}, the dataset must include {required_hint}. "
                + col_hint
            )
    elif task_type == "pos":
        if not cols.get("pos_tags"):
            errors.append(
                f"For {task_label}, a POS tags column (e.g. 'pos_tags', 'tags') is required. "
                + col_hint
            )
        if not cols.get("tokens") and not cols.get("text"):
            errors.append(
                f"For {task_label}, a tokens or text column is required alongside the POS tags. "
                + col_hint
            )
    elif task_type == "semantic_similarity":
        pair_ok = cols.get("text_a") and cols.get("text_b") and cols.get("similarity")
        retrieval_ok = cols.get("query") and cols.get("document") and cols.get("relevance")
        if not pair_ok and not retrieval_ok:
            errors.append(
                f"For {task_label}, the dataset must include {required_hint}. "
                + col_hint
            )
    elif task_type == "topic_modeling":
        if not cols.get("text"):
            errors.append(
                f"For {task_label}, a text column is required. "
                + col_hint
            )
    text_cols = [v for k, v in cols.items() if isinstance(v, str) and k in {"text", "source_text", "target_text", "context", "question", "prompt", "completion", "text_a", "text_b", "document", "query"}]
    for col in sorted(set(text_cols)):
        if col in df.columns and df[col].fillna("").astype(str).str.strip().eq("").all():
            errors.append(f"The column '{col}' appears to be entirely empty — no usable text found.")
    if not errors and task_type in {"ner", "pos", "question_answering", "relation_extraction"}:
        validity = annotation_validity_summary(df, task_type, cols)
        if validity.get("invalid_count", 0) > 0:
            errors.append(
                f"{validity.get('invalid_count')} row(s) contain invalid or unparseable annotations for {task_label}. "
                "Check that annotations are formatted as JSON lists or BIO tag sequences."
            )
    return errors
