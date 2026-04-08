"""
preprocessing.py — Pipeline specification model.

PipelineSpec is a plain data object describing one candidate preprocessing
pipeline.  It is intentionally framework-agnostic so it can be stored in
JSON and compared easily.  The actual sklearn objects are constructed in
executor.py from this spec.

Supported options
-----------------
num_imputer          : "mean" | "median" | "knn"
cat_imputer          : "mode" | "constant"
scaler               : "none" | "standard" | "minmax" | "robust"
power_transform      : bool  (Yeo-Johnson; applied before scaling)
encoder              : "onehot" | "ordinal"
remove_duplicates    : bool
remove_low_variance  : bool  (VarianceThreshold on full transformed array)
imbalance            : "none" | "oversample" | "smote"
outlier_clip         : bool  (clip numeric values to [Q1-1.5*IQR, Q3+1.5*IQR])
drop_high_missing_cols: bool  (drop cols with >50% missing instead of imputing)
"""

from dataclasses import dataclass


@dataclass
class PipelineSpec:
    num_imputer: str            # "mean" | "median" | "knn"
    cat_imputer: str            # "mode" | "constant"
    scaler: str                 # "none" | "standard" | "minmax" | "robust"
    power_transform: bool
    encoder: str                # "onehot" | "ordinal"
    remove_duplicates: bool
    remove_low_variance: bool
    imbalance: str              # "none" | "oversample" | "smote"
    outlier_clip: bool = False  # clip extreme values to IQR bounds before scaling
    drop_high_missing_cols: bool = False  # drop cols with >50% missing

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "num_imputer":           self.num_imputer,
            "cat_imputer":           self.cat_imputer,
            "scaler":                self.scaler,
            "power_transform":       self.power_transform,
            "encoder":               self.encoder,
            "remove_duplicates":     self.remove_duplicates,
            "remove_low_variance":   self.remove_low_variance,
            "imbalance":             self.imbalance,
            "outlier_clip":          self.outlier_clip,
            "drop_high_missing_cols": self.drop_high_missing_cols,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        return cls(
            num_imputer=d["num_imputer"],
            cat_imputer=d["cat_imputer"],
            scaler=d["scaler"],
            power_transform=bool(d["power_transform"]),
            encoder=d["encoder"],
            remove_duplicates=bool(d["remove_duplicates"]),
            remove_low_variance=bool(d["remove_low_variance"]),
            imbalance=d["imbalance"],
            # backward-compatible defaults for fields added later
            outlier_clip=bool(d.get("outlier_clip", False)),
            drop_high_missing_cols=bool(d.get("drop_high_missing_cols", False)),
        )

    # ── Human-readable name ────────────────────────────────────────────────

    def name(self) -> str:
        parts = [
            f"num={self.num_imputer}",
            f"cat={self.cat_imputer}",
            f"scale={self.scaler}",
        ]
        if self.power_transform:
            parts.append("pwr")
        if self.outlier_clip:
            parts.append("clip")
        parts.append(f"enc={self.encoder}")
        if self.remove_duplicates:
            parts.append("dedup")
        if self.remove_low_variance:
            parts.append("lv")
        if self.drop_high_missing_cols:
            parts.append("drp_miss")
        if self.imbalance != "none":
            parts.append(f"imb={self.imbalance}")
        return " | ".join(parts)

    # ── Complexity score (for tie-breaking — lower = simpler) ──────────────

    def complexity_score(self) -> int:
        score = 0
        if self.num_imputer == "knn":
            score += 2
        if self.power_transform:
            score += 1
        if self.outlier_clip:
            score += 1
        if self.scaler != "none":
            score += 1
        if self.imbalance == "smote":
            score += 2
        elif self.imbalance == "oversample":
            score += 1
        if self.remove_low_variance:
            score += 1
        if self.drop_high_missing_cols:
            score += 1
        return score
