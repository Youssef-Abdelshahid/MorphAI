import pandas as pd

from .config import (
    SUPPORTED_TASK_TYPES,
    VALID_TASK_TYPES,
    default_metric_for_task,
    normalize_task_type,
    valid_metrics_for_task,
)

_SUPERVISED_TASK_TYPES = {"binary", "multiclass", "multilabel", "regression", "ordinal", "ranking", "time_series"}
_NUMERIC_TARGET_TASKS = {"regression", "time_series"}


def validate_csv_run(config, df: pd.DataFrame) -> list:
    errors = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()

    if df.shape[0] < 10:
        errors.append(
            f"Dataset has only {df.shape[0]} rows. At least 10 rows are required."
        )

    if not task_type:
        errors.append("A tabular task type is required.")
    elif task_type not in VALID_TASK_TYPES:
        errors.append(
            f"Task type '{config.task_type}' is not valid for tabular data. "
            f"Supported task types: {SUPPORTED_TASK_TYPES}"
        )
    elif task_type not in SUPPORTED_TASK_TYPES:
        errors.append(
            f"Task type '{task_type}' is not yet supported by the tabular pipeline. "
            f"Supported task types: {SUPPORTED_TASK_TYPES}"
        )

    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics:
        if not metric:
            errors.append(
                f"A priority metric is required for a {task_type} tabular task. "
                f"Suggested default: {default_metric_for_task(task_type)}"
            )
        elif metric not in valid_metrics:
            errors.append(
                f"Metric '{config.metric}' is not valid for a {task_type} task. "
                f"Valid metrics: {valid_metrics}"
            )

    is_supervised = task_type in _SUPERVISED_TASK_TYPES

    if is_supervised:
        if not config.target:
            errors.append("A target column is required for supervised tasks.")
        elif config.target not in df.columns:
            errors.append(
                f"Target column '{config.target}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        else:
            target_series = df[config.target].dropna()
            if len(target_series) == 0:
                errors.append(
                    f"Target column '{config.target}' is entirely missing (all NaN)."
                )
            elif target_series.nunique() < 2:
                errors.append(
                    f"Target column '{config.target}' has only one unique value. "
                    "At least two distinct values are required."
                )
            elif task_type in _NUMERIC_TARGET_TASKS and not pd.api.types.is_numeric_dtype(df[config.target]):
                errors.append(
                    f"Target column '{config.target}' must be numeric for {task_type} tasks."
                )
    elif config.target and config.target not in df.columns:
        errors.append(
            f"Target column '{config.target}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    feature_cols = [c for c in df.columns if c != config.target]
    if len(feature_cols) == 0:
        errors.append("No feature columns found after excluding the target column.")

    num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist() if feature_cols else []
    cat_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist() if feature_cols else []
    if not num_cols and not cat_cols:
        errors.append("No usable feature columns detected in the dataset.")

    if task_type == "time_series":
        if len(df) < 20:
            errors.append("Time-series forecasting needs at least 20 rows for time-aware evaluation.")
    elif task_type in {"clustering", "anomaly", "dimensionality_reduction", "association_rules"} and len(feature_cols) < 1:
        errors.append(f"{task_type} requires at least one feature column.")

    return errors
