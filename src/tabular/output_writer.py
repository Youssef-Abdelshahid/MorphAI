"""
output_writer.py — Save the best-pipeline-cleaned dataset to disk.

After the best pipeline is selected, it is re-fit on the *full* dataset
(no train/val split) and the transformed features are saved as a CSV.

This gives users a ready-to-use cleaned file for downstream model training.

Note: oversampling (SMOTE / RandomOverSampler) is intentionally NOT applied
here — those are training-time balancing techniques and must not appear in
the saved output file.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .config import Config
from .executor import _build_column_transformer
from .preprocessing import PipelineSpec
from .profiler import DataProfile

PROCESSED_DIR = Path("processed")


def _pipeline_short_id(spec: PipelineSpec) -> str:
    """Build a short, filename-safe identifier from the key spec choices."""
    parts = [
        spec.num_imputer[:3],                                    # mea | med | knn
        spec.scaler[:3] if spec.scaler != "none" else "nsc",    # sta | min | rob | nsc
        spec.encoder[:3],                                        # one | ord
    ]
    if spec.power_transform:
        parts.append("pwr")
    if spec.outlier_clip:
        parts.append("clp")
    if spec.drop_high_missing_cols:
        parts.append("drp")
    if spec.imbalance != "none":
        parts.append(spec.imbalance[:3])                         # ove | smo
    return "_".join(parts)


def save_cleaned_dataset(
    spec: PipelineSpec,
    df: pd.DataFrame,
    profile: DataProfile,
    config: Config,
) -> Tuple[Path, tuple]:
    """
    Fit the best preprocessing pipeline on the full dataset and save the
    result to  processed/<dataset>_<pipeline_id>_cleaned.csv.

    Parameters
    ----------
    spec    : selected best PipelineSpec
    df      : original full DataFrame (features + target, before any split)
    profile : dataset profile
    config  : run configuration

    Returns
    -------
    (path, shape) — path to saved CSV and its (rows, cols) shape.
    """
    # ── 1. Dataset-level: deduplication ───────────────────────────────────
    if spec.remove_duplicates and profile.n_duplicates > 0:
        feature_cols = [c for c in df.columns if c != profile.target_col]
        df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)

    X = df.drop(columns=[profile.target_col])
    y = df[profile.target_col].reset_index(drop=True)

    drop_set = set(profile.high_missing_cols) if spec.drop_high_missing_cols else set()
    num_cols = [c for c in profile.num_cols if c in X.columns and c not in drop_set]
    cat_cols = [c for c in profile.cat_cols if c in X.columns and c not in drop_set]

    # ── 2. Fit column transformer on full X ───────────────────────────────
    ct = _build_column_transformer(spec, num_cols, cat_cols)
    X_transformed: np.ndarray = ct.fit_transform(X)

    # ── 3. Optional variance filter (fit on full X) ───────────────────────
    vt = None
    if spec.remove_low_variance:
        vt = VarianceThreshold(threshold=0.01)
        X_transformed = vt.fit_transform(X_transformed)

    # ── 4. Recover human-readable column names ────────────────────────────
    try:
        col_names = list(ct.get_feature_names_out())
        if vt is not None:
            support = vt.get_support()
            col_names = [n for n, keep in zip(col_names, support) if keep]
        # Strip the leading "num__" / "cat__" prefix for cleaner names
        col_names = [n.split("__", 1)[-1] for n in col_names]
    except Exception:
        col_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # ── 5. Assemble output DataFrame (features + target) ──────────────────
    out_df = pd.DataFrame(X_transformed, columns=col_names)
    out_df[profile.target_col] = y.values

    # ── 6. Save ───────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stem = Path(config.data_path).stem
    pid = _pipeline_short_id(spec)
    out_path = PROCESSED_DIR / f"{dataset_stem}_{pid}_cleaned.csv"
    out_df.to_csv(out_path, index=False)

    return out_path, out_df.shape
