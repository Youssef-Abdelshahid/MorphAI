from dataclasses import dataclass
from pathlib import Path


VALID_METRICS = ["accuracy", "f1", "precision", "recall"]

_IMG_TASK_BACKEND = {
    "Image classification (single-label)": "classification",
    "Image classification (multi-label)":  "multiclass",
    "Object detection":                    "other",
    "Semantic segmentation":               "other",
    "Instance segmentation":               "other",
    "Keypoint / pose estimation":          "other",
    "Image similarity / retrieval":        "other",
    "Anomaly / defect detection":          "classification",
    "Optical character recognition":       "other",
    "Image generation / synthesis":        "other",
    "Depth estimation":                    "other",
}

SUPPORTED_TASK_TYPES = {"classification", "multiclass", "binary"}


@dataclass
class ImageConfig:
    data_path:    Path
    metric:       str = "f1"

    task_type:    str = "classification"
    domain:       str = ""
    constraints:  str = ""
    notes:        str = ""

    modality:     str = "Image"
    image_format: str = ""
    color_space:  str = ""

    @property
    def supervision(self) -> str:
        return "supervised"

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    def task_context(self) -> dict:
        return {
            "task_type":          self.task_type,
            "domain":             self.domain,
            "constraints":        self.constraints,
            "active_constraints": self.active_constraints,
            "notes":              self.notes,
            "modality":           self.modality,
            "image_format":       self.image_format,
            "color_space":        self.color_space,
            "supervision":        self.supervision,
        }
