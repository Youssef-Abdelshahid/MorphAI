from pathlib import Path
from typing import List

import pandas as pd

from .columns import resolve_columns
from .config import SUPPORTED_TASK_TYPES, VALID_TASK_TYPES, default_metric_for_task, normalize_task_type, valid_metrics_for_task
from .profiler import annotation_validity_summary

SUPPORTED_TEXT_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def validate_text_file(path: Path) -> List[str]:
    errors = []
    if not path.exists():
        return [f"File does not exist: {path}"]
    if path.suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        errors.append("Text modality currently accepts structured CSV or Excel files only. Please upload a .csv, .xlsx, or .xls dataset with one sample per row.")
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


def validate_text_run(config, df: pd.DataFrame) -> List[str]:
    errors = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()
    if df.empty:
        return ["The text dataset is empty."]
    if not task_type:
        errors.append("A text task type is required.")
    elif task_type not in VALID_TASK_TYPES:
        errors.append(f"Task type '{config.task_type}' is not valid for text data. Supported task types: {sorted(SUPPORTED_TASK_TYPES)}")
    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics and metric and metric not in valid_metrics:
        config.metric = default_metric_for_task(task_type)
    if valid_metrics and not metric:
        config.metric = default_metric_for_task(task_type)
    cols = resolve_columns(df, task_type)
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
            errors.append(f"Missing required column(s) for {task_type}: {', '.join(miss)}.")
    elif task_type == "classification_multi":
        if not cols.get("text"):
            errors.append("Missing required text column for multi-label text classification.")
        if not cols.get("labels") and not cols.get("binary_label_columns"):
            errors.append("Multi-label text classification requires a labels column or multiple binary label columns.")
    elif task_type == "pos":
        if not cols.get("pos_tags"):
            errors.append("Part-of-speech tagging requires a pos_tags column.")
        if not cols.get("tokens") and not cols.get("text"):
            errors.append("Part-of-speech tagging requires tokens or text column aligned with POS tags.")
    elif task_type == "semantic_similarity":
        pair_ok = cols.get("text_a") and cols.get("text_b") and cols.get("similarity")
        retrieval_ok = cols.get("query") and cols.get("document") and cols.get("relevance")
        if not pair_ok and not retrieval_ok:
            errors.append("Semantic similarity requires text_a/text_b plus similarity_score or query/document/relevance columns.")
    elif task_type == "topic_modeling":
        if not cols.get("text"):
            errors.append("Topic modeling requires a text column.")
    text_cols = [v for k, v in cols.items() if isinstance(v, str) and k in {"text", "source_text", "target_text", "context", "question", "prompt", "completion", "text_a", "text_b", "document", "query"}]
    for col in sorted(set(text_cols)):
        if col in df.columns and df[col].fillna("").astype(str).str.strip().eq("").all():
            errors.append(f"Text column '{col}' is empty.")
    if not errors and task_type in {"ner", "pos", "question_answering", "relation_extraction"}:
        validity = annotation_validity_summary(df, task_type, cols)
        if validity.get("invalid_count", 0) > 0:
            errors.append(f"{validity.get('invalid_count')} row(s) contain invalid or unparseable annotations for {task_type}.")
    return errors
