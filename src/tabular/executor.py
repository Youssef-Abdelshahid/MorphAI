import math
import re
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .config import default_metric_for_task, metric_label, normalize_task_type
from .preprocessing import PipelineSpec
from .profiler import DataProfile

try:
    from sklearn.preprocessing import OneHotEncoder as _OHE
    _OHE(sparse_output=False)

    def _onehot() -> _OHE:
        return _OHE(handle_unknown="ignore", sparse_output=False)

except TypeError:
    def _onehot() -> _OHE:
        return _OHE(handle_unknown="ignore", sparse=False)


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: Optional[pd.Series]
    num_cols: List[str]
    cat_cols: List[str]


class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, _y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X, _y=None):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)


def _clamp_01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else 0.0


def _safe_std(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(finite)) if finite else 0.0


def _safe_scale(value: float, fallback: float = 1.0) -> float:
    if not np.isfinite(value) or abs(value) <= 1e-12:
        return fallback
    return float(abs(value))


def _target_error_scales(y: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(y, dtype=float)
    std_scale = _safe_scale(float(np.nanstd(arr)), 1.0)
    range_scale = _safe_scale(float(np.nanmax(arr) - np.nanmin(arr)), std_scale)
    mean_scale = _safe_scale(float(np.nanmean(np.abs(arr))), std_scale)
    return {
        "std": std_scale,
        "range": range_scale,
        "mean_abs": mean_scale,
        "rmse": max(std_scale, range_scale / 4.0, 1.0),
        "mae": max(std_scale, range_scale / 4.0, 1.0),
    }


def _build_column_transformer(spec: PipelineSpec, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []

    if num_cols:
        num_steps = []
        if spec.num_imputer == "mean":
            num_steps.append(("imputer", SimpleImputer(strategy="mean")))
        elif spec.num_imputer == "median":
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        elif spec.num_imputer == "knn":
            num_steps.append(("imputer", KNNImputer(n_neighbors=5)))

        if spec.outlier_clip:
            num_steps.append(("clipper", OutlierClipper()))

        if spec.power_transform:
            num_steps.append(("power", PowerTransformer(method="yeo-johnson")))

        if spec.scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif spec.scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif spec.scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        transformers.append(("num", SklearnPipeline(num_steps) if num_steps else "passthrough", num_cols))

    if cat_cols:
        cat_steps = []
        if spec.cat_imputer == "mode":
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        elif spec.cat_imputer == "constant":
            cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))

        if spec.encoder == "onehot":
            cat_steps.append(("encoder", _onehot()))
        elif spec.encoder == "ordinal":
            cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))

        transformers.append(("cat", SklearnPipeline(cat_steps) if cat_steps else "passthrough", cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _as_dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float)


def _prepare_dataframe(spec: PipelineSpec, df: pd.DataFrame, profile: DataProfile, task_type: str) -> PreparedData:
    work_df = df.copy()
    if spec.remove_duplicates and profile.n_duplicates > 0:
        feature_cols = [c for c in work_df.columns if c != profile.target_col]
        work_df = work_df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    target_col = profile.target_col if profile.target_col in work_df.columns else None
    X = work_df.drop(columns=[target_col]) if target_col else work_df.copy()
    y = work_df[target_col] if target_col else None

    drop_set = set(profile.high_missing_cols) if spec.drop_high_missing_cols else set()
    X = X.drop(columns=[c for c in drop_set if c in X.columns], errors="ignore")

    num_cols = [c for c in profile.num_cols if c in X.columns]
    cat_cols = [c for c in profile.cat_cols if c in X.columns]

    if task_type == "time_series" and y is not None:
        X, y = _build_time_series_frame(X, y)
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    return PreparedData(X=X, y=y, num_cols=num_cols, cat_cols=cat_cols)


def _fit_transform_split(
    spec: PipelineSpec,
    prepared: PreparedData,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    X_train_df = prepared.X.iloc[list(train_idx)].copy()
    X_test_df = prepared.X.iloc[list(test_idx)].copy()
    ct = _build_column_transformer(spec, prepared.num_cols, prepared.cat_cols)
    X_train = _as_dense_array(ct.fit_transform(X_train_df))
    X_test = _as_dense_array(ct.transform(X_test_df))
    if spec.remove_low_variance and X_train.shape[1] > 0:
        selector = VarianceThreshold(threshold=0.01)
        X_train = selector.fit_transform(X_train)
        X_test = selector.transform(X_test)
    return X_train, X_test


def _transform_full_features(spec: PipelineSpec, prepared: PreparedData) -> np.ndarray:
    ct = _build_column_transformer(spec, prepared.num_cols, prepared.cat_cols)
    X = _as_dense_array(ct.fit_transform(prepared.X.copy()))
    if spec.remove_low_variance and X.shape[1] > 0:
        selector = VarianceThreshold(threshold=0.01)
        X = selector.fit_transform(X)
    return X


def _apply_sampling(spec: PipelineSpec, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if spec.imbalance == "oversample":
        sampler = RandomOverSampler(random_state=42)
        return sampler.fit_resample(X_train, y_train)
    if spec.imbalance == "smote":
        class_counts = pd.Series(y_train).value_counts()
        if len(class_counts) > 1 and int(class_counts.min()) >= 2:
            k_neighbors = max(1, min(5, int(class_counts.min()) - 1))
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            return sampler.fit_resample(X_train, y_train)
    return X_train, y_train


def _resolve_metric(task_type: str, requested: str, available: Sequence[str], fallback: Optional[str] = None) -> Tuple[str, Optional[str]]:
    requested = (requested or "").strip().lower()
    available_set = set(available)
    if requested and requested in available_set:
        return requested, None
    task_default = fallback or default_metric_for_task(task_type)
    if task_default in available_set:
        if requested and requested not in available_set:
            return task_default, f"Requested metric '{requested}' was unavailable; used {metric_label(task_default)} instead."
        return task_default, None
    if available:
        selected = list(available)[0]
        if requested and requested != selected:
            return selected, f"Requested metric '{requested}' was unavailable; used {metric_label(selected)} instead."
        return selected, None
    return fallback or requested or "", "No valid evaluation metric was available."


def _make_result(
    spec: PipelineSpec,
    task_type: str,
    metric_priority: str,
    selected_metric: str,
    raw_metrics: Dict[str, float],
    normalized_metrics: Dict[str, float],
    model_scores: Dict[str, Dict[str, float]],
    evaluator_details: Dict[str, Any],
    evaluation_mode: str,
    evaluation_summary: str,
    elapsed_sec: float,
    metrics_std: Optional[Dict[str, float]] = None,
    normalized_metrics_std: Optional[Dict[str, float]] = None,
    n_splits: int = 0,
    n_models: int = 0,
    success: bool = True,
    reason: str = "",
) -> Dict[str, Any]:
    final_score = _clamp_01(normalized_metrics.get(selected_metric, 0.0))
    final_score_std = float((normalized_metrics_std or {}).get(f"{selected_metric}_std", 0.0))
    return {
        "spec": spec,
        "task_type": task_type,
        "metric_priority": metric_priority,
        "selected_metric": selected_metric,
        "metrics": raw_metrics,
        "raw_metrics": raw_metrics,
        "metrics_std": metrics_std or {},
        "normalized_metrics": normalized_metrics,
        "normalized_metrics_std": normalized_metrics_std or {},
        "final_score": final_score,
        "normalized_score": final_score,
        "final_score_std": final_score_std,
        "model_scores": model_scores,
        "per_model_metrics": model_scores,
        "evaluator_details": evaluator_details,
        "evaluation_mode": evaluation_mode,
        "evaluation_summary": evaluation_summary,
        "n_splits": n_splits,
        "n_models": n_models,
        "elapsed_sec": round(elapsed_sec, 3),
        "success": success,
        "reason": reason,
    }


def _failed_result(spec: PipelineSpec, task_type: str, metric_priority: str, reason: str, elapsed_sec: float) -> Dict[str, Any]:
    fallback_metric = default_metric_for_task(task_type) or metric_priority or "score"
    return _make_result(
        spec=spec,
        task_type=task_type,
        metric_priority=metric_priority,
        selected_metric=fallback_metric,
        raw_metrics={fallback_metric: 0.0},
        normalized_metrics={fallback_metric: 0.0},
        model_scores={},
        evaluator_details={"failure_reason": reason},
        evaluation_mode="failed",
        evaluation_summary=reason,
        elapsed_sec=elapsed_sec,
        n_splits=0,
        n_models=0,
        success=False,
        reason=reason,
    )


def _safe_predict_proba(model: Any, X_test: np.ndarray) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X_test)
        except Exception:
            return None
    return None


def _aggregate_model_metrics(
    per_model_raw: Dict[str, Dict[str, float]],
    per_model_norm: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    metric_names = sorted({k for metrics in per_model_raw.values() for k in metrics})
    raw_metrics = {k: _safe_mean([m.get(k, np.nan) for m in per_model_raw.values()]) for k in metric_names}
    raw_std = {f"{k}_std": _safe_std([m.get(k, np.nan) for m in per_model_raw.values()]) for k in metric_names}
    norm_metrics = {k: _safe_mean([m.get(k, np.nan) for m in per_model_norm.values()]) for k in metric_names}
    norm_std = {f"{k}_std": _safe_std([m.get(k, np.nan) for m in per_model_norm.values()]) for k in metric_names}
    return raw_metrics, raw_std, norm_metrics, norm_std


def _evaluate_binary_or_multiclass(
    spec: PipelineSpec,
    prepared: PreparedData,
    task_type: str,
    metric_priority: str,
) -> Dict[str, Any]:
    y = prepared.y
    if y is None:
        raise ValueError("A target column is required for classification.")

    y_series = y.reset_index(drop=True)
    class_counts = y_series.value_counts()
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Each class needs at least 2 samples for stratified evaluation.")

    models = {
        "logreg": lambda: LogisticRegression(solver="saga", max_iter=2000, tol=1e-3, random_state=42),
        "tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=42),
        "gnb": lambda: GaussianNB(),
    }
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    splitters = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = np.unique(y_series)

    for model_name, factory in models.items():
        fold_metrics: Dict[str, List[float]] = {}
        for train_idx, test_idx in splitters.split(prepared.X, y_series):
            X_train, X_test = _fit_transform_split(spec, prepared, train_idx, test_idx)
            y_train = y_series.iloc[list(train_idx)].to_numpy()
            y_test = y_series.iloc[list(test_idx)].to_numpy()
            X_train, y_train = _apply_sampling(spec, X_train, y_train)
            model = factory()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if task_type == "binary":
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, pos_label=labels[-1], zero_division=0),
                    "precision": precision_score(y_test, y_pred, pos_label=labels[-1], zero_division=0),
                    "recall": recall_score(y_test, y_pred, pos_label=labels[-1], zero_division=0),
                }
                probs = _safe_predict_proba(model, X_test)
                if probs is not None and probs.ndim == 2 and probs.shape[1] == 2:
                    try:
                        metrics["roc_auc"] = roc_auc_score(y_test, probs[:, 1])
                    except Exception:
                        pass
            else:
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                    "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                    "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                }
            for metric_name, metric_value in metrics.items():
                fold_metrics.setdefault(metric_name, []).append(float(metric_value))

        per_model_raw[model_name] = {k: _safe_mean(v) for k, v in fold_metrics.items()}
        per_model_norm[model_name] = {k: _clamp_01(v) for k, v in per_model_raw[model_name].items()}

    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    fallback = "f1" if task_type == "binary" else "macro_f1"
    selected_metric, metric_note = _resolve_metric(task_type, metric_priority, raw_metrics.keys(), fallback=fallback)
    summary = (
        f"Stratified {n_splits}-fold evaluation across {len(models)} lightweight classifiers. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        task_type,
        metric_priority,
        selected_metric,
        raw_metrics,
        norm_metrics,
        per_model_raw,
        {
            "metric_note": metric_note,
            "model_family": "shallow tabular classifier",
            "models": list(models.keys()),
            "baselines": ["LogisticRegression", "DecisionTreeClassifier", "GaussianNB"],
        },
        "supervised",
        summary,
        0.0,
        metrics_std=raw_std,
        normalized_metrics_std=norm_std,
        n_splits=n_splits,
        n_models=len(models),
    )


def _normalize_regression_metric(metric_name: str, value: float, scales: Dict[str, float]) -> float:
    if metric_name == "r2":
        return _clamp_01((value + 1.0) / 2.0)
    if metric_name in {"rmse", "mae"}:
        scale = scales.get(metric_name, scales.get("std", 1.0))
        return _clamp_01(1.0 / (1.0 + max(value, 0.0) / _safe_scale(scale)))
    return 0.0


def _evaluate_regression_like(
    spec: PipelineSpec,
    prepared: PreparedData,
    task_type: str,
    metric_priority: str,
) -> Dict[str, Any]:
    if prepared.y is None:
        raise ValueError("A target column is required for regression-style evaluation.")
    y = pd.to_numeric(prepared.y, errors="coerce").reset_index(drop=True)
    mask = y.notna()
    prepared = PreparedData(
        X=prepared.X.loc[mask].reset_index(drop=True),
        y=y.loc[mask].reset_index(drop=True),
        num_cols=prepared.num_cols,
        cat_cols=prepared.cat_cols,
    )
    if len(prepared.X) < 10:
        raise ValueError("Not enough rows after cleaning the target column.")

    models = {
        "linear": lambda: LinearRegression(),
        "tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
        "rf": lambda: RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42, n_jobs=1),
    }
    n_splits = min(5, len(prepared.X))
    if n_splits < 2:
        raise ValueError("At least 2 folds are required for regression evaluation.")
    splitters = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scales = _target_error_scales(prepared.y)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}

    for model_name, factory in models.items():
        fold_metrics: Dict[str, List[float]] = {}
        for train_idx, test_idx in splitters.split(prepared.X):
            X_train, X_test = _fit_transform_split(spec, prepared, train_idx, test_idx)
            y_train = prepared.y.iloc[list(train_idx)].to_numpy(dtype=float)
            y_test = prepared.y.iloc[list(test_idx)].to_numpy(dtype=float)
            model = factory()
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=float)
            resid = y_test - y_pred
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
            metrics = {
                "mae": float(np.mean(np.abs(resid))),
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "r2": float(r2),
            }
            for metric_name, metric_value in metrics.items():
                fold_metrics.setdefault(metric_name, []).append(float(metric_value))

        per_model_raw[model_name] = {k: _safe_mean(v) for k, v in fold_metrics.items()}
        per_model_norm[model_name] = {
            k: _normalize_regression_metric(k, v, scales)
            for k, v in per_model_raw[model_name].items()
        }

    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    fallback = "r2" if task_type == "regression" else "mae"
    selected_metric, metric_note = _resolve_metric(task_type, metric_priority, raw_metrics.keys(), fallback=fallback)
    if task_type == "regression" and selected_metric == "r2" and not np.isfinite(raw_metrics.get("r2", np.nan)):
        selected_metric = "rmse"
        metric_note = "R2 was unstable for this dataset, so RMSE was used for ranking."
    summary = (
        f"{n_splits}-fold regression evaluation across {len(models)} lightweight regressors. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        task_type,
        metric_priority,
        selected_metric,
        raw_metrics,
        norm_metrics,
        per_model_raw,
        {
            "target_scales": scales,
            "metric_note": metric_note,
            "model_family": "shallow tabular regressor",
            "models": list(models.keys()),
            "baselines": ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"],
        },
        "supervised",
        summary,
        0.0,
        metrics_std=raw_std,
        normalized_metrics_std=norm_std,
        n_splits=n_splits,
        n_models=len(models),
    )


def _find_time_column(X: pd.DataFrame) -> Optional[str]:
    datetime_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    if datetime_cols:
        return datetime_cols[0]
    for col in X.columns:
        name = col.lower()
        if any(token in name for token in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return col
    return None


def _build_time_series_frame(X: pd.DataFrame, y: pd.Series, max_lag: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
    work_X = X.copy()
    work_y = pd.to_numeric(y, errors="coerce").copy()
    time_col = _find_time_column(work_X)
    if time_col:
        parsed = pd.to_datetime(work_X[time_col], errors="coerce")
        order = np.argsort(parsed.fillna(pd.Timestamp.min).to_numpy())
        work_X = work_X.iloc[order].reset_index(drop=True)
        work_y = work_y.iloc[order].reset_index(drop=True)
        work_X = work_X.drop(columns=[time_col], errors="ignore")
    else:
        work_X = work_X.reset_index(drop=True)
        work_y = work_y.reset_index(drop=True)

    for lag in range(1, max_lag + 1):
        work_X[f"target_lag_{lag}"] = work_y.shift(lag)

    full = pd.concat([work_X, work_y.rename("__target__")], axis=1).dropna().reset_index(drop=True)
    return full.drop(columns=["__target__"]), full["__target__"]


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) <= 1e-12, np.nan, np.abs(y_true))
    values = np.abs((y_true - y_pred) / denom)
    values = values[np.isfinite(values)]
    return float(np.mean(values) * 100.0) if len(values) else 0.0


def _evaluate_time_series(
    spec: PipelineSpec,
    prepared: PreparedData,
    metric_priority: str,
) -> Dict[str, Any]:
    if prepared.y is None:
        raise ValueError("A numeric target column is required for forecasting.")
    y = pd.to_numeric(prepared.y, errors="coerce")
    mask = y.notna()
    prepared = PreparedData(
        X=prepared.X.loc[mask].reset_index(drop=True),
        y=y.loc[mask].reset_index(drop=True),
        num_cols=prepared.num_cols,
        cat_cols=prepared.cat_cols,
    )
    if len(prepared.X) < 20:
        raise ValueError("Time-series forecasting needs at least 20 usable rows.")

    models = {
        "naive_last_value": None,
        "linear": lambda: LinearRegression(),
        "tree": lambda: DecisionTreeRegressor(max_depth=6, random_state=42),
    }
    n_splits = min(5, max(2, len(prepared.X) // 5))
    splitters = TimeSeriesSplit(n_splits=n_splits)
    scales = _target_error_scales(prepared.y)
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}

    for model_name, factory in models.items():
        fold_metrics: Dict[str, List[float]] = {}
        for train_idx, test_idx in splitters.split(prepared.X):
            if model_name == "naive_last_value":
                test_frame = prepared.X.iloc[list(test_idx)]
                if "target_lag_1" not in test_frame.columns:
                    continue
                y_pred = pd.to_numeric(test_frame["target_lag_1"], errors="coerce").to_numpy(dtype=float)
            else:
                X_train, X_test = _fit_transform_split(spec, prepared, train_idx, test_idx)
                y_train = prepared.y.iloc[list(train_idx)].to_numpy(dtype=float)
                model = factory()
                model.fit(X_train, y_train)
                y_pred = np.asarray(model.predict(X_test), dtype=float)
            y_test = prepared.y.iloc[list(test_idx)].to_numpy(dtype=float)
            metrics = {
                "mae": float(np.mean(np.abs(y_test - y_pred))),
                "rmse": float(np.sqrt(np.mean((y_test - y_pred) ** 2))),
                "smape": _smape(y_test, y_pred),
                "mape": _mape(y_test, y_pred),
            }
            for metric_name, metric_value in metrics.items():
                fold_metrics.setdefault(metric_name, []).append(float(metric_value))

        per_model_raw[model_name] = {k: _safe_mean(v) for k, v in fold_metrics.items()}
        per_model_norm[model_name] = {
            "mae": _normalize_regression_metric("mae", per_model_raw[model_name].get("mae", scales["mae"]), scales),
            "rmse": _normalize_regression_metric("rmse", per_model_raw[model_name].get("rmse", scales["rmse"]), scales),
            "smape": _clamp_01(1.0 - min(per_model_raw[model_name].get("smape", 200.0), 200.0) / 200.0),
            "mape": _clamp_01(1.0 - min(per_model_raw[model_name].get("mape", 200.0), 200.0) / 200.0),
        }

    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    selected_metric, metric_note = _resolve_metric("time_series", metric_priority, raw_metrics.keys(), fallback="rmse")
    summary = (
        f"Time-aware {n_splits}-split forecasting evaluation across {len(models)} lightweight baselines/models. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        "time_series",
        metric_priority,
        selected_metric,
        raw_metrics,
        norm_metrics,
        per_model_raw,
        {
            "target_scales": scales,
            "metric_note": metric_note,
            "model_family": "lag-feature time-series forecasters",
            "models": list(models.keys()),
            "baselines": ["naive_last_value", "LinearRegression", "DecisionTreeRegressor"],
            "split_strategy": "TimeSeriesSplit",
        },
        "supervised",
        summary,
        0.0,
        metrics_std=raw_std,
        normalized_metrics_std=norm_std,
        n_splits=n_splits,
        n_models=len(models),
    )


def _estimate_cluster_count(n_rows: int) -> int:
    return max(2, min(8, int(round(math.sqrt(max(n_rows, 4) / 2.0)))))


def _cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, float], Optional[str]]:
    valid_mask = labels != -1
    unique_labels = np.unique(labels[valid_mask] if np.any(valid_mask) else labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        return {}, "Invalid clustering structure for internal metrics."
    X_eval = X[valid_mask] if np.any(labels == -1) else X
    labels_eval = labels[valid_mask] if np.any(labels == -1) else labels
    if len(np.unique(labels_eval)) < 2 or len(X_eval) <= len(np.unique(labels_eval)):
        return {}, "Clustering collapsed to one cluster or all-noise."
    metrics = {
        "silhouette_score": float(silhouette_score(X_eval, labels_eval)),
        "davies_bouldin_score": float(davies_bouldin_score(X_eval, labels_eval)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_eval, labels_eval)),
    }
    return metrics, None


def _evaluate_clustering(
    spec: PipelineSpec,
    prepared: PreparedData,
    metric_priority: str,
) -> Dict[str, Any]:
    X = _transform_full_features(spec, prepared)
    if X.shape[0] < 5 or X.shape[1] < 1:
        raise ValueError("Clustering needs at least 5 rows and 1 usable feature.")
    n_clusters = _estimate_cluster_count(len(X))
    models = {
        "kmeans": lambda: KMeans(n_clusters=n_clusters, n_init=10, random_state=42),
        "agglomerative": lambda: AgglomerativeClustering(n_clusters=n_clusters),
        "dbscan": lambda: DBSCAN(eps=0.7, min_samples=max(3, min(8, len(X) // 20 or 3))),
    }
    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}
    invalid_models: Dict[str, str] = {}

    for model_name, factory in models.items():
        model = factory()
        labels = model.fit_predict(X)
        metrics, error = _cluster_metrics(X, labels)
        if error:
            invalid_models[model_name] = error
            per_model_raw[model_name] = {}
            per_model_norm[model_name] = {}
            continue
        per_model_raw[model_name] = metrics
        per_model_norm[model_name] = {
            "silhouette_score": _clamp_01((metrics.get("silhouette_score", -1.0) + 1.0) / 2.0),
            "davies_bouldin_score": _clamp_01(1.0 / (1.0 + max(metrics.get("davies_bouldin_score", 0.0), 0.0))),
            "calinski_harabasz_score": _clamp_01(
                math.log1p(max(metrics.get("calinski_harabasz_score", 0.0), 0.0))
                / (1.0 + math.log1p(max(metrics.get("calinski_harabasz_score", 0.0), 0.0)))
            ),
        }

    valid_raw = {k: v for k, v in per_model_raw.items() if v}
    valid_norm = {k: v for k, v in per_model_norm.items() if v}
    if not valid_raw:
        raise ValueError("All clustering algorithms produced invalid cluster structures.")
    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(valid_raw, valid_norm)
    selected_metric, metric_note = _resolve_metric("clustering", metric_priority, raw_metrics.keys(), fallback="silhouette_score")
    summary = (
        f"Unsupervised clustering evaluation across {len(valid_raw)} valid clustering algorithms. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if invalid_models:
        summary = f"{summary} Some clustering candidates were invalid and were down-weighted to zero."
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        "clustering",
        metric_priority,
        selected_metric,
        raw_metrics,
        norm_metrics,
        valid_raw,
        {
            "invalid_models": invalid_models,
            "metric_note": metric_note,
            "model_family": "unsupervised tabular clustering",
            "models": list(valid_raw.keys()),
            "baselines": ["KMeans", "AgglomerativeClustering", "DBSCAN"],
        },
        "unsupervised",
        summary,
        0.0,
        metrics_std=raw_std,
        normalized_metrics_std=norm_std,
        n_splits=1,
        n_models=len(models),
    )


def _anomaly_proxy_metrics(scores: np.ndarray, predictions: np.ndarray, X_reference: np.ndarray) -> Dict[str, float]:
    inlier_scores = scores[predictions == 0]
    outlier_scores = scores[predictions == 1]
    if len(outlier_scores) == 0 or len(inlier_scores) == 0:
        separation = 0.0
    else:
        gap = float(np.mean(outlier_scores) - np.mean(inlier_scores))
        spread = _safe_scale(float(np.std(scores)), 1.0)
        separation = _clamp_01(0.5 + 0.5 * np.tanh(gap / spread))
    observed_contamination = float(np.mean(predictions))
    target_contamination = min(0.15, max(0.02, 5.0 / max(len(predictions), 1)))
    contamination_consistency = _clamp_01(1.0 - abs(observed_contamination - target_contamination) / max(target_contamination, 1e-6))
    subset = min(len(X_reference), max(20, len(X_reference) // 2))
    if subset >= 10:
        idx_a = np.arange(subset)
        idx_b = np.arange(len(X_reference) - subset, len(X_reference))
        stability = _clamp_01(1.0 - abs(np.mean(predictions[idx_a]) - np.mean(predictions[idx_b])))
    else:
        stability = 0.5
    proxy_score = _clamp_01(0.4 * separation + 0.3 * stability + 0.3 * contamination_consistency)
    return {
        "proxy_score": proxy_score,
        "score_separation": separation,
        "stability": stability,
        "contamination_consistency": contamination_consistency,
    }


def _evaluate_anomaly(
    spec: PipelineSpec,
    df: pd.DataFrame,
    profile: DataProfile,
    metric_priority: str,
) -> Dict[str, Any]:
    prepared = _prepare_dataframe(spec, df, profile, "anomaly")
    X = _transform_full_features(spec, prepared)
    if X.shape[0] < 10 or X.shape[1] < 1:
        raise ValueError("Anomaly detection needs at least 10 rows and 1 usable feature.")

    contamination = min(0.15, max(0.02, 5.0 / max(len(X), 1)))
    models = {
        "isolation_forest": lambda: IsolationForest(contamination=contamination, random_state=42),
        "lof": lambda: LocalOutlierFactor(contamination=contamination, novelty=False),
        "one_class_svm": lambda: OneClassSVM(nu=contamination, gamma="scale"),
    }

    target_col = profile.target_col if profile.target_col in df.columns else None
    supervised_labels = None
    if target_col and target_col not in prepared.X.columns:
        try:
            raw_target = df[target_col]
            if raw_target.nunique(dropna=True) == 2:
                normalized = raw_target.astype(str).str.lower().str.strip()
                positives = {"1", "true", "yes", "anomaly", "outlier", "fraud"}
                if normalized.isin(positives).any():
                    supervised_labels = normalized.isin(positives).astype(int).to_numpy()
                else:
                    values = list(raw_target.dropna().unique())
                    supervised_labels = (raw_target == values[-1]).astype(int).to_numpy()
        except Exception:
            supervised_labels = None

    per_model_raw: Dict[str, Dict[str, float]] = {}
    per_model_norm: Dict[str, Dict[str, float]] = {}

    for model_name, factory in models.items():
        model = factory()
        if model_name == "lof":
            pred = model.fit_predict(X)
            raw_scores = -np.asarray(model.negative_outlier_factor_, dtype=float)
        else:
            model.fit(X)
            pred = model.predict(X)
            decision = model.decision_function(X) if hasattr(model, "decision_function") else model.score_samples(X)
            raw_scores = -np.asarray(decision, dtype=float)
        anomaly_pred = (pred == -1).astype(int)
        if supervised_labels is not None and len(supervised_labels) == len(anomaly_pred):
            metrics = {
                "f1": f1_score(supervised_labels, anomaly_pred, zero_division=0),
                "precision": precision_score(supervised_labels, anomaly_pred, zero_division=0),
                "recall": recall_score(supervised_labels, anomaly_pred, zero_division=0),
            }
            if len(np.unique(supervised_labels)) == 2:
                try:
                    metrics["roc_auc"] = roc_auc_score(supervised_labels, raw_scores)
                except Exception:
                    pass
            norm_metrics = {k: _clamp_01(v) for k, v in metrics.items()}
        else:
            metrics = _anomaly_proxy_metrics(raw_scores, anomaly_pred, X)
            norm_metrics = {k: _clamp_01(v) for k, v in metrics.items()}
        per_model_raw[model_name] = metrics
        per_model_norm[model_name] = norm_metrics

    raw_metrics, raw_std, norm_metrics, norm_std = _aggregate_model_metrics(per_model_raw, per_model_norm)
    if supervised_labels is not None:
        fallback = "f1"
        mode = "supervised"
    else:
        fallback = "proxy_score"
        mode = "proxy"
    selected_metric, metric_note = _resolve_metric("anomaly", metric_priority, raw_metrics.keys(), fallback=fallback)
    summary = (
        f"Anomaly evaluation across {len(models)} anomaly detectors. "
        f"Selected metric: {metric_label(selected_metric)} = {raw_metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {norm_metrics.get(selected_metric, 0.0):.4f}."
    )
    if mode == "proxy":
        summary = f"{summary} This score is based on internal proxy metrics because no ground-truth anomaly labels were available."
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        "anomaly",
        metric_priority,
        selected_metric,
        raw_metrics,
        norm_metrics,
        per_model_raw,
        {
            "metric_note": metric_note,
            "has_ground_truth": supervised_labels is not None,
            "model_family": "unsupervised tabular anomaly detector",
            "models": list(models.keys()),
            "baselines": ["IsolationForest", "LocalOutlierFactor", "OneClassSVM"],
        },
        mode,
        summary,
        0.0,
        metrics_std=raw_std,
        normalized_metrics_std=norm_std,
        n_splits=1,
        n_models=len(models),
    )


_TXN_DELIMS = ["|", ";", ","]


def _detect_basket_column(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        sample = df[col].dropna().astype(str).head(200)
        if len(sample) == 0:
            continue
        for delim in _TXN_DELIMS:
            hits = sample.str.contains(re.escape(delim), regex=True).sum()
            if hits >= max(5, int(0.5 * len(sample))):
                return col, delim
    return None


def _to_transactions(df: pd.DataFrame) -> List[set]:
    basket = _detect_basket_column(df)
    if basket is not None:
        col, delim = basket
        transactions: List[set] = []
        for value in df[col]:
            if pd.isna(value):
                continue
            items = {tok.strip() for tok in str(value).split(delim) if tok.strip()}
            if items:
                transactions.append(items)
        if transactions:
            return transactions
    work = df.copy()
    transactions = []
    for _, row in work.iterrows():
        items = set()
        for col, value in row.items():
            if pd.isna(value):
                continue
            if pd.api.types.is_numeric_dtype(type(value)):
                items.add(f"{col}={round(float(value), 4)}")
            else:
                items.add(f"{col}={str(value)}")
        if items:
            transactions.append(items)
    return transactions


def _build_rule_transactions(prepared: PreparedData) -> List[set]:
    work = prepared.X.copy()
    if _detect_basket_column(work) is not None:
        return _to_transactions(work)
    for col in work.select_dtypes(include=[np.number]).columns:
        series = pd.to_numeric(work[col], errors="coerce")
        if series.notna().sum() >= 4:
            try:
                work[col] = pd.qcut(series, q=min(4, series.nunique()), duplicates="drop").astype(str)
            except Exception:
                work[col] = series.round(2).astype(str)
        else:
            work[col] = series.round(2).astype(str)
    return _to_transactions(work)


def _mine_association_rules(transactions: List[set]) -> Dict[str, float]:
    n_transactions = len(transactions)
    if n_transactions < 5:
        return {"rule_quality": 0.0, "support": 0.0, "confidence": 0.0, "lift": 0.0, "coverage": 0.0, "number_of_rules": 0.0}
    item_counts: Dict[str, int] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}
    for txn in transactions:
        for item in txn:
            item_counts[item] = item_counts.get(item, 0) + 1
        for a, b in combinations(sorted(txn), 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

    min_support = max(2, int(round(0.05 * n_transactions)))
    supports = []
    confidences = []
    lifts = []
    covered_transactions = set()
    for (a, b), count in pair_counts.items():
        if count < min_support:
            continue
        support = count / n_transactions
        conf_ab = count / max(item_counts.get(a, 1), 1)
        conf_ba = count / max(item_counts.get(b, 1), 1)
        lift_ab = conf_ab / max(item_counts.get(b, 1) / n_transactions, 1e-12)
        lift_ba = conf_ba / max(item_counts.get(a, 1) / n_transactions, 1e-12)
        supports.extend([support, support])
        confidences.extend([conf_ab, conf_ba])
        lifts.extend([lift_ab, lift_ba])
        for idx, txn in enumerate(transactions):
            if a in txn or b in txn:
                covered_transactions.add(idx)

    number_of_rules = len(confidences)
    if number_of_rules == 0:
        return {"rule_quality": 0.0, "support": 0.0, "confidence": 0.0, "lift": 0.0, "coverage": 0.0, "number_of_rules": 0.0}

    support = _safe_mean(supports)
    confidence = _safe_mean(confidences)
    lift = _safe_mean(lifts)
    coverage = len(covered_transactions) / n_transactions
    rule_count_score = math.exp(-abs(number_of_rules - 20) / 20.0)
    lift_score = lift / (lift + 1.0)
    rule_quality = _clamp_01(0.3 * confidence + 0.25 * lift_score + 0.2 * support + 0.2 * coverage + 0.05 * rule_count_score)
    return {
        "rule_quality": rule_quality,
        "support": support,
        "confidence": confidence,
        "lift": lift,
        "coverage": coverage,
        "number_of_rules": float(number_of_rules),
    }


def _evaluate_association_rules(
    spec: PipelineSpec,
    prepared: PreparedData,
    metric_priority: str,
) -> Dict[str, Any]:
    transactions = _build_rule_transactions(prepared)
    metrics = _mine_association_rules(transactions)
    normalized = {
        "rule_quality": _clamp_01(metrics.get("rule_quality", 0.0)),
        "support": _clamp_01(metrics.get("support", 0.0)),
        "confidence": _clamp_01(metrics.get("confidence", 0.0)),
        "lift": _clamp_01(metrics.get("lift", 0.0) / (metrics.get("lift", 0.0) + 1.0)) if metrics.get("lift", 0.0) > 0 else 0.0,
        "coverage": _clamp_01(metrics.get("coverage", 0.0)),
        "number_of_rules": _clamp_01(1.0 - math.exp(-metrics.get("number_of_rules", 0.0) / 20.0)),
    }
    selected_metric, metric_note = _resolve_metric("association_rules", metric_priority, metrics.keys(), fallback="rule_quality")
    summary = (
        f"Association rule mining evaluated with transaction-level rule quality metrics. "
        f"Selected metric: {metric_label(selected_metric)} = {metrics.get(selected_metric, 0.0):.4f}. "
        f"Normalized score = {normalized.get(selected_metric, 0.0):.4f}."
    )
    if metrics.get("number_of_rules", 0.0) <= 0:
        summary = f"{summary} No useful rules were found, so the pipeline score is zero."
    if metric_note:
        summary = f"{summary} {metric_note}"
    return _make_result(
        spec,
        "association_rules",
        metric_priority,
        selected_metric,
        metrics,
        normalized,
        {"rule_miner": metrics},
        {
            "transaction_count": len(transactions),
            "metric_note": metric_note,
            "model_family": "association rule miner",
            "models": ["pairwise_apriori_baseline"],
            "baselines": ["pairwise_support_confidence_lift"],
        },
        "unsupervised",
        summary,
        0.0,
        metrics_std={f"{k}_std": 0.0 for k in metrics},
        normalized_metrics_std={f"{k}_std": 0.0 for k in normalized},
        n_splits=1,
        n_models=1,
    )


def evaluate_pipeline(
    spec: PipelineSpec,
    df: pd.DataFrame,
    profile: DataProfile,
    task_type: str,
    metric: str,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    task_type = normalize_task_type(task_type)
    try:
        if task_type not in {
            "binary",
            "multiclass",
            "regression",
            "time_series",
            "clustering",
            "anomaly",
            "association_rules",
        }:
            raise ValueError(f"Task type '{task_type}' is not supported by the tabular evaluator.")

        if task_type == "binary":
            result = _evaluate_binary_or_multiclass(spec, _prepare_dataframe(spec, df, profile, task_type), "binary", metric)
        elif task_type == "multiclass":
            result = _evaluate_binary_or_multiclass(spec, _prepare_dataframe(spec, df, profile, task_type), "multiclass", metric)
        elif task_type == "regression":
            result = _evaluate_regression_like(spec, _prepare_dataframe(spec, df, profile, task_type), "regression", metric)
        elif task_type == "time_series":
            result = _evaluate_time_series(spec, _prepare_dataframe(spec, df, profile, task_type), metric)
        elif task_type == "clustering":
            result = _evaluate_clustering(spec, _prepare_dataframe(spec, df, profile, task_type), metric)
        elif task_type == "anomaly":
            result = _evaluate_anomaly(spec, df, profile, metric)
        elif task_type == "association_rules":
            result = _evaluate_association_rules(spec, _prepare_dataframe(spec, df, profile, task_type), metric)
        else:
            raise ValueError(f"Task type '{task_type}' is not supported by the tabular evaluator.")
        result["elapsed_sec"] = round(time.perf_counter() - t_start, 3)
        return result
    except Exception as exc:
        return _failed_result(spec, task_type, metric, str(exc), time.perf_counter() - t_start)
