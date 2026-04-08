from dataclasses import dataclass, field
from pathlib import Path

VALID_METRICS    = ["accuracy", "f1", "precision", "recall"]
VALID_TASK_TYPES = ["classification", "binary", "multiclass", "regression", "other"]

_FE_BUDGET_NORM = {
    "": "moderate",
    "Minimal (raw features only)":           "minimal",
    "Light (basic transforms)":              "light",
    "Moderate (interactions + encoding)":    "moderate",
    "Heavy (full feature engineering)":      "heavy",
}

_DATA_QUALITY_NORM = {
    "": "unknown",
    "Clean / well-curated":          "clean",
    "Mostly clean (minor issues)":   "mostly_clean",
    "Noisy / real-world collection": "noisy",
    "Mixed quality":                 "mixed",
    "Unknown":                       "unknown",
}


@dataclass
class Config:
    data_path: Path
    target:    str
    metric:    str = "f1"

    task_type:    str = "classification"
    domain:       str = ""
    constraints:  str = ""
    notes:        str = ""

    modality:     str = "CSV / Tabular"
    fe_budget:    str = ""
    data_quality: str = ""

    @property
    def supervision(self) -> str:
        return "unsupervised" if self.task_type == "other" else "supervised"

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    @property
    def fe_budget_norm(self) -> str:
        return _FE_BUDGET_NORM.get(self.fe_budget, "moderate")

    @property
    def data_quality_norm(self) -> str:
        return _DATA_QUALITY_NORM.get(self.data_quality, "unknown")

    def task_context(self) -> dict:
        return {
            "task_type":   self.task_type,
            "domain":      self.domain,
            "constraints": self.constraints,
            "active_constraints": self.active_constraints,
            "notes":             self.notes,
            "modality":          self.modality,
            "fe_budget":         self.fe_budget,
            "fe_budget_norm":    self.fe_budget_norm,
            "data_quality":      self.data_quality,
            "data_quality_norm": self.data_quality_norm,
            "supervision":       self.supervision,
        }
