"""
profiler.py — Dataset profiling (Perception & Context layer).

Analyses a tabular classification dataset and returns a DataProfile
containing a deep, comprehensive set of statistics needed for intelligent
pipeline generation.

Profile dimensions
------------------
  Shape & types       : rows, columns, numeric vs. categorical breakdown
  Missing values      : per-column ratios, total ratio, high-missing columns
  Duplicates          : exact duplicate feature rows
  Class distribution  : counts, imbalance ratio, minority class size
  Outliers            : IQR-based detection per numeric column
  Skewness            : per-column skewness, high-skew detection
  Kurtosis            : per-column excess kurtosis, heavy-tail detection
  Sparsity            : fraction of zero values per numeric column
  Cardinality         : unique count per categorical column
  Constant columns    : single-value or near-constant (>=95%) columns
  Binary numeric cols : numeric columns with exactly 2 distinct values
  Multicollinearity   : pairs of numeric columns with |Pearson r| > 0.85
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataProfile:
    # ── Shape ──────────────────────────────────────────────────────────────
    n_rows: int
    n_cols: int                         # feature columns only (target excluded)
    num_cols: List[str]
    cat_cols: List[str]
    target_col: str

    # ── Missing values ─────────────────────────────────────────────────────
    total_missing_ratio: float          # fraction of all feature cells that are NaN
    missing_per_col: Dict[str, float]   # col → fraction missing
    high_missing_cols: List[str]        # cols where > 50% values are missing

    # ── Duplicates ─────────────────────────────────────────────────────────
    n_duplicates: int                   # duplicate feature rows (target ignored)

    # ── Class distribution ─────────────────────────────────────────────────
    class_counts: Dict[str, int]
    n_classes: int
    imbalance_ratio: float              # majority_count / minority_count
    min_class_size: int                 # samples in smallest class

    # ── Outliers (IQR-based, numerical only) ───────────────────────────────
    outlier_ratios: Dict[str, float]    # col → fraction of rows that are outliers
    high_outlier_cols: List[str]        # cols where outlier_ratio > 5 %

    # ── Skewness ───────────────────────────────────────────────────────────
    skewness: Dict[str, float]          # col → skewness value
    high_skew_cols: List[str]           # cols where |skew| > 1.0

    # ── Kurtosis (excess) ──────────────────────────────────────────────────
    kurtosis: Dict[str, float]          # col → excess kurtosis (Fisher definition)
    high_kurtosis_cols: List[str]       # cols where |kurtosis| > 3.0 (heavy tails)

    # ── Sparsity (zeros in numeric columns) ────────────────────────────────
    zero_ratios: Dict[str, float]       # col → fraction of zero values

    # ── Categorical cardinality ────────────────────────────────────────────
    cardinality: Dict[str, int]         # cat col → unique value count
    high_cardinality_cols: List[str]    # cols with > 20 unique values

    # ── Constant / near-constant columns ──────────────────────────────────
    constant_cols: List[str]            # single unique value
    near_constant_cols: List[str]       # dominant value covers >= 95 % of rows

    # ── Binary numerical columns ───────────────────────────────────────────
    binary_num_cols: List[str]          # numeric cols with exactly 2 unique non-NaN values

    # ── Multicollinearity (numeric features only) ──────────────────────────
    n_high_corr_pairs: int              # pairs with |Pearson r| > 0.85
    correlated_col_pairs: List[Tuple[str, str]]  # the (col_a, col_b) pairs

    # ── Summary flags (for rule-based pipeline generation) ─────────────────
    has_missing: bool
    has_high_missing: bool              # total_missing_ratio > 10 %
    has_high_missing_cols: bool         # any individual col with > 50% missing
    has_duplicates: bool
    has_outliers: bool
    has_high_skew: bool
    has_high_kurtosis: bool             # any col with heavy tails
    has_sparse_features: bool           # any numeric col with zero_ratio > 0.5
    is_imbalanced: bool                 # imbalance_ratio > 1.5
    is_highly_imbalanced: bool          # imbalance_ratio > 3.0
    has_categorical: bool
    has_high_cardinality: bool
    has_constant_cols: bool
    has_binary_num_cols: bool           # any binary numeric features present
    has_multicollinearity: bool         # any high-correlation pair found


def profile_dataset(df: pd.DataFrame, target: str) -> DataProfile:
    """Generate a comprehensive profile for a tabular classification dataset."""

    feature_df = df.drop(columns=[target])
    target_series = df[target]

    n_rows, n_cols = feature_df.shape

    # ── Column types ───────────────────────────────────────────────────────
    num_cols: List[str] = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols: List[str] = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── Missing values ─────────────────────────────────────────────────────
    missing_per_col: Dict[str, float] = {
        col: float(feature_df[col].isnull().mean()) for col in feature_df.columns
    }
    total_missing_ratio = sum(missing_per_col.values()) / max(n_cols, 1)
    high_missing_cols: List[str] = [c for c, r in missing_per_col.items() if r > 0.50]

    # ── Duplicates ─────────────────────────────────────────────────────────
    n_duplicates = int(feature_df.duplicated().sum())

    # ── Class distribution ─────────────────────────────────────────────────
    raw_counts = target_series.value_counts()
    class_counts: Dict[str, int] = {str(k): int(v) for k, v in raw_counts.items()}
    n_classes = len(class_counts)
    counts_sorted = sorted(class_counts.values(), reverse=True)
    min_class_size = counts_sorted[-1] if counts_sorted else 1
    imbalance_ratio = (
        counts_sorted[0] / counts_sorted[-1] if counts_sorted[-1] > 0 else float("inf")
    )

    # ── Outliers (IQR method) ──────────────────────────────────────────────
    outlier_ratios: Dict[str, float] = {}
    for col in num_cols:
        col_data = feature_df[col].dropna()
        if len(col_data) < 4:
            outlier_ratios[col] = 0.0
            continue
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_ratios[col] = 0.0
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_ratios[col] = float(((col_data < lower) | (col_data > upper)).mean())
    high_outlier_cols = [c for c, r in outlier_ratios.items() if r > 0.05]

    # ── Skewness ───────────────────────────────────────────────────────────
    skewness: Dict[str, float] = {}
    for col in num_cols:
        col_data = feature_df[col].dropna()
        skewness[col] = float(col_data.skew()) if len(col_data) > 3 else 0.0
    high_skew_cols = [c for c, s in skewness.items() if abs(s) > 1.0]

    # ── Kurtosis (excess, Fisher definition) ──────────────────────────────
    # Normal distribution has kurtosis = 0; heavy tails > 0.
    kurtosis_vals: Dict[str, float] = {}
    for col in num_cols:
        col_data = feature_df[col].dropna()
        kurtosis_vals[col] = float(col_data.kurt()) if len(col_data) > 3 else 0.0
    high_kurtosis_cols = [c for c, k in kurtosis_vals.items() if abs(k) > 3.0]

    # ── Sparsity (fraction of zeros in numeric cols) ───────────────────────
    zero_ratios: Dict[str, float] = {}
    for col in num_cols:
        col_data = feature_df[col].dropna()
        zero_ratios[col] = float((col_data == 0).mean()) if len(col_data) > 0 else 0.0

    # ── Cardinality ────────────────────────────────────────────────────────
    cardinality: Dict[str, int] = {
        col: int(feature_df[col].nunique()) for col in cat_cols
    }
    high_cardinality_cols = [c for c, n in cardinality.items() if n > 20]

    # ── Constant / near-constant columns ──────────────────────────────────
    constant_cols: List[str] = []
    near_constant_cols: List[str] = []
    for col in feature_df.columns:
        vc = feature_df[col].value_counts(normalize=True, dropna=False)
        if len(vc) == 0 or float(vc.iloc[0]) >= 1.0:
            constant_cols.append(col)
        elif float(vc.iloc[0]) >= 0.95:
            near_constant_cols.append(col)

    # ── Binary numerical columns ───────────────────────────────────────────
    binary_num_cols: List[str] = [
        c for c in num_cols if feature_df[c].nunique(dropna=True) == 2
    ]

    # ── Multicollinearity: pairs with |Pearson r| > 0.85 ──────────────────
    correlated_col_pairs: List[Tuple[str, str]] = []
    if len(num_cols) >= 2 and n_rows >= 10:
        try:
            corr_matrix = feature_df[num_cols].corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr_matrix.iloc[i, j]
                    if pd.notna(val) and val > 0.85:
                        correlated_col_pairs.append((num_cols[i], num_cols[j]))
        except Exception:
            pass  # correlation failure is non-fatal
    n_high_corr_pairs = len(correlated_col_pairs)

    return DataProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        target_col=target,
        total_missing_ratio=total_missing_ratio,
        missing_per_col=missing_per_col,
        high_missing_cols=high_missing_cols,
        n_duplicates=n_duplicates,
        class_counts=class_counts,
        n_classes=n_classes,
        imbalance_ratio=imbalance_ratio,
        min_class_size=min_class_size,
        outlier_ratios=outlier_ratios,
        high_outlier_cols=high_outlier_cols,
        skewness=skewness,
        high_skew_cols=high_skew_cols,
        kurtosis=kurtosis_vals,
        high_kurtosis_cols=high_kurtosis_cols,
        zero_ratios=zero_ratios,
        cardinality=cardinality,
        high_cardinality_cols=high_cardinality_cols,
        constant_cols=constant_cols,
        near_constant_cols=near_constant_cols,
        binary_num_cols=binary_num_cols,
        n_high_corr_pairs=n_high_corr_pairs,
        correlated_col_pairs=correlated_col_pairs,
        has_missing=total_missing_ratio > 0,
        has_high_missing=total_missing_ratio > 0.10,
        has_high_missing_cols=len(high_missing_cols) > 0,
        has_duplicates=n_duplicates > 0,
        has_outliers=len(high_outlier_cols) > 0,
        has_high_skew=len(high_skew_cols) > 0,
        has_high_kurtosis=len(high_kurtosis_cols) > 0,
        has_sparse_features=any(r > 0.5 for r in zero_ratios.values()),
        is_imbalanced=imbalance_ratio > 1.5,
        is_highly_imbalanced=imbalance_ratio > 3.0,
        has_categorical=len(cat_cols) > 0,
        has_high_cardinality=len(high_cardinality_cols) > 0,
        has_constant_cols=len(constant_cols) > 0 or len(near_constant_cols) > 0,
        has_binary_num_cols=len(binary_num_cols) > 0,
        has_multicollinearity=n_high_corr_pairs > 0,
    )
