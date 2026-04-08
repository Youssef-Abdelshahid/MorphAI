"""
executor.py — Pipeline execution and evaluation (Reasoning / Evaluation layer).

Each candidate PipelineSpec is evaluated with Stratified K-Fold cross-validation
(default 5 folds) across THREE diverse lightweight models:

  logreg : LogisticRegression  (linear, regularised)
  tree   : DecisionTreeClassifier  (non-linear, no scaling needed)
  gnb    : GaussianNB  (probabilistic, very fast)

Evaluating across multiple model types ensures that pipeline rankings are not
biased toward any single inductive bias.  The reported metrics are the mean
(and inter-model std) of each model's mean-fold score.

Pipeline structure (inside each fold, per model):
  ColumnTransformer → [OutlierClipper] → [VarianceThreshold] → [Sampler] → Classifier

  • ColumnTransformer and VarianceThreshold are sklearn transformers.
  • OutlierClipper (custom) clips extreme numeric values to IQR bounds.
  • SMOTE / RandomOverSampler are imblearn resamplers applied only during fit.
  • imblearn Pipeline is used when a resampler is present.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import PipelineSpec
from .profiler import DataProfile

# ── OneHotEncoder compatibility (sklearn >= 1.2 renamed sparse -> sparse_output)
try:
    from sklearn.preprocessing import OneHotEncoder as _OHE
    _OHE(sparse_output=False)

    def _onehot() -> _OHE:
        return _OHE(handle_unknown="ignore", sparse_output=False)

except TypeError:
    def _onehot() -> _OHE:  # type: ignore[misc]
        return _OHE(handle_unknown="ignore", sparse=False)


# ── Diverse lightweight models evaluated per pipeline ────────────────────────
# Each entry: (name, factory_fn) — factory called fresh for each pipeline/model
_MODELS: List[Tuple[str, Any]] = [
    ("logreg", lambda: LogisticRegression(
        solver="saga", max_iter=2000, tol=1e-3, random_state=42)),
    ("tree",   lambda: DecisionTreeClassifier(max_depth=6, random_state=42)),
    ("gnb",    lambda: GaussianNB()),
]


# ── Custom transformer: IQR-based outlier clipping ───────────────────────────

class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip each numeric feature to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    Bounds are estimated on the training fold only (no leakage).
    Intended for use after imputation (input should be free of NaN).
    """

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


# ── Column transformer builder ───────────────────────────────────────────────

def _build_column_transformer(
    spec: PipelineSpec,
    num_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """Translate a PipelineSpec into a fitted-ready ColumnTransformer."""
    transformers = []

    if num_cols:
        num_steps = []

        # 1. Imputation
        if spec.num_imputer == "mean":
            num_steps.append(("imputer", SimpleImputer(strategy="mean")))
        elif spec.num_imputer == "median":
            num_steps.append(("imputer", SimpleImputer(strategy="median")))
        elif spec.num_imputer == "knn":
            num_steps.append(("imputer", KNNImputer(n_neighbors=5)))

        # 2. Optional outlier clipping (after imputation so no NaN remain)
        if spec.outlier_clip:
            num_steps.append(("clipper", OutlierClipper()))

        # 3. Optional power transform (before scaling)
        if spec.power_transform:
            num_steps.append(("power", PowerTransformer(method="yeo-johnson")))

        # 4. Scaling
        if spec.scaler == "standard":
            num_steps.append(("scaler", StandardScaler()))
        elif spec.scaler == "minmax":
            num_steps.append(("scaler", MinMaxScaler()))
        elif spec.scaler == "robust":
            num_steps.append(("scaler", RobustScaler()))

        if num_steps:
            transformers.append(("num", SklearnPipeline(num_steps), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

    if cat_cols:
        cat_steps = []
        if spec.cat_imputer == "mode":
            cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
        elif spec.cat_imputer == "constant":
            cat_steps.append(
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing"))
            )

        if spec.encoder == "onehot":
            cat_steps.append(("encoder", _onehot()))
        elif spec.encoder == "ordinal":
            cat_steps.append((
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ))

        if cat_steps:
            transformers.append(("cat", SklearnPipeline(cat_steps), cat_cols))
        else:
            transformers.append(("cat", "passthrough", cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── Main evaluation function ─────────────────────────────────────────────────

def evaluate_pipeline(
    spec: PipelineSpec,
    df: pd.DataFrame,
    profile: DataProfile,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a PipelineSpec using Stratified K-Fold CV across multiple models.

    Returns a result dict with aggregate metrics (mean across models) and
    per-model breakdown, or None if the pipeline cannot be evaluated.

    Cross-validation guarantees
    ---------------------------
    - All transformer/scaler/encoder/clipper/variance-filter steps are fit on
      the training fold only and applied to the validation fold.
    - Oversampling is applied only on the training fold (imblearn Pipeline).
    - No information from the validation fold leaks into preprocessing.

    Multi-model aggregation
    -----------------------
    Each pipeline is evaluated with three diverse models (logreg, tree, gnb).
    The reported metrics are the mean of each model's mean-fold score.
    The reported std is the inter-model std of those means.
    """
    t_start = time.perf_counter()

    try:
        # ── 1. Dataset-level: duplicate removal (before CV) ───────────────
        if spec.remove_duplicates and profile.n_duplicates > 0:
            feature_cols = [c for c in df.columns if c != profile.target_col]
            df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

        X = df.drop(columns=[profile.target_col])
        y = df[profile.target_col]

        # ── 2. Resolve active columns after optional high-missing drop ─────
        drop_set = set(profile.high_missing_cols) if spec.drop_high_missing_cols else set()
        num_cols = [c for c in profile.num_cols if c in X.columns and c not in drop_set]
        cat_cols = [c for c in profile.cat_cols if c in X.columns and c not in drop_set]

        if not num_cols and not cat_cols:
            return None

        # ── 3. Determine number of CV folds ──────────────────────────────
        min_class = int(y.value_counts().min())
        n_splits = min(5, min_class)
        if n_splits < 2:
            return None

        # ── 4. Build shared preprocessing steps (up to but excl. classifier)
        ct = _build_column_transformer(spec, num_cols, cat_cols)
        preproc_steps: list = [("preprocessor", ct)]

        if spec.remove_low_variance:
            preproc_steps.append(("variance_filter", VarianceThreshold(threshold=0.01)))

        if spec.imbalance == "smote":
            preproc_steps.append(("sampler", SMOTE(random_state=42)))
        elif spec.imbalance == "oversample":
            preproc_steps.append(("sampler", RandomOverSampler(random_state=42)))

        # ── 5. Scoring setup ──────────────────────────────────────────────
        avg = "macro"
        scoring = {
            "accuracy":  "accuracy",
            "f1":        make_scorer(f1_score,        average=avg, zero_division=0),
            "precision": make_scorer(precision_score, average=avg, zero_division=0),
            "recall":    make_scorer(recall_score,    average=avg, zero_division=0),
        }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # ── 6. Evaluate each model ─────────────────────────────────────────
        per_model_metrics: Dict[str, Dict[str, float]] = {}

        for model_name, model_factory in _MODELS:
            model_steps = preproc_steps + [("classifier", model_factory())]

            if spec.imbalance != "none":
                pipeline = ImbPipeline(model_steps)
            else:
                pipeline = SklearnPipeline(model_steps)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                cv_result = cross_validate(
                    pipeline, X, y,
                    cv=cv,
                    scoring=scoring,
                    error_score=0.0,
                    n_jobs=1,
                )

            per_model_metrics[model_name] = {
                k: float(np.mean(cv_result[f"test_{k}"]))
                for k in ["accuracy", "f1", "precision", "recall"]
            }

        # ── 7. Aggregate across models ────────────────────────────────────
        # metrics     = mean of per-model means  (central tendency)
        # metrics_std = std  of per-model means  (inter-model consistency)
        metrics: Dict[str, float] = {}
        metrics_std: Dict[str, float] = {}
        for k in ["accuracy", "f1", "precision", "recall"]:
            model_means = [per_model_metrics[mn][k] for mn in per_model_metrics]
            metrics[k] = float(np.mean(model_means))
            metrics_std[f"{k}_std"] = float(np.std(model_means))

        elapsed = time.perf_counter() - t_start
        return {
            "spec":               spec,
            "metrics":            metrics,
            "metrics_std":        metrics_std,
            "per_model_metrics":  per_model_metrics,
            "n_splits":           n_splits,
            "n_models":           len(_MODELS),
            "elapsed_sec":        round(elapsed, 3),
        }

    except Exception:
        return None  # invalid pipeline for this dataset — skip silently
