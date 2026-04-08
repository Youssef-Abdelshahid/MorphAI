import pandas as pd

_REGRESSION_METRICS = {"f1", "precision", "recall"}
_SUPERVISED_TASK_TYPES = {"binary", "multiclass", "classification", "regression"}


def validate_csv_run(config, df: pd.DataFrame) -> list:
    errors = []

    if df.shape[0] < 10:
        errors.append(
            f"Dataset has only {df.shape[0]} rows. At least 10 rows are required."
        )

    is_supervised = config.task_type in _SUPERVISED_TASK_TYPES

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

    if config.task_type == "regression" and config.metric in _REGRESSION_METRICS:
        errors.append(
            f"Metric '{config.metric}' is not valid for a regression task. "
            "Use 'accuracy' or choose a different task type."
        )

    feature_cols = [c for c in df.columns if c != config.target]
    if len(feature_cols) == 0:
        errors.append("No feature columns found after excluding the target column.")

    num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist() if feature_cols else []
    cat_cols = df[feature_cols].select_dtypes(exclude="number").columns.tolist() if feature_cols else []
    if not num_cols and not cat_cols:
        errors.append("No usable feature columns detected in the dataset.")

    return errors
