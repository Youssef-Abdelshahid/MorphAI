from dataclasses import dataclass
from pathlib import Path

SINGLE_LABEL_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]
MULTILABEL_CLASSIFICATION_METRICS = ["micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"]
OBJECT_DETECTION_METRICS = ["map", "map_50", "precision", "recall", "mean_iou"]
SEMANTIC_SEGMENTATION_METRICS = ["mean_iou", "pixel_accuracy", "dice_score"]
RETRIEVAL_METRICS = ["recall_at_k", "precision_at_k", "map", "mrr"]
ANOMALY_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "proxy_score"]
OCR_METRICS = ["normalized_edit_similarity", "exact_match_accuracy", "cer", "wer"]

LABEL_MODES = ["single-label", "multi-label"]

_IMG_TASK_BACKEND = {
    "Image classification": "classification",
    "Object detection": "detection",
    "Semantic segmentation": "semantic_segmentation",
    "Image similarity / retrieval": "retrieval",
    "Anomaly / defect detection": "anomaly",
    "Optical character recognition": "ocr",
}

VALID_TASK_TYPES = [
    "classification",
    "multilabel",
    "detection",
    "semantic_segmentation",
    "retrieval",
    "anomaly",
    "ocr",
]
SUPPORTED_TASK_TYPES = set(VALID_TASK_TYPES)

DEPRECATED_TASK_TYPES = {
    "instance_segmentation",
    "keypoint",
    "generation",
    "depth",
}

TASK_DISPLAY_NAMES = {
    "classification": "Image classification",
    "multilabel": "Image classification",
    "detection": "Object detection",
    "semantic_segmentation": "Semantic segmentation",
    "retrieval": "Image similarity / retrieval",
    "anomaly": "Anomaly / defect detection",
    "ocr": "Optical character recognition",
}

_LEGACY_TASK_ALIASES = {
    "Image classification (single-label)": ("Image classification", "single-label"),
    "Image classification (multi-label)": ("Image classification", "multi-label"),
}

_TASK_FAMILIES = {
    "classification": "classification",
    "multilabel": "classification",
    "detection": "detection",
    "semantic_segmentation": "segmentation",
    "retrieval": "retrieval",
    "anomaly": "anomaly",
    "ocr": "ocr",
}

_TASK_METRICS = {
    "classification": SINGLE_LABEL_CLASSIFICATION_METRICS,
    "multilabel": MULTILABEL_CLASSIFICATION_METRICS,
    "detection": OBJECT_DETECTION_METRICS,
    "semantic_segmentation": SEMANTIC_SEGMENTATION_METRICS,
    "retrieval": RETRIEVAL_METRICS,
    "anomaly": ANOMALY_METRICS,
    "ocr": OCR_METRICS,
}

_DEFAULT_METRICS = {
    "classification": "macro_f1",
    "multilabel": "micro_f1",
    "detection": "map",
    "semantic_segmentation": "mean_iou",
    "retrieval": "recall_at_k",
    "anomaly": "auroc",
    "ocr": "normalized_edit_similarity",
}

_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "precision": "Precision",
    "recall": "Recall",
    "micro_f1": "Micro F1",
    "hamming_loss": "Hamming loss",
    "subset_accuracy": "Subset accuracy",
    "map": "mAP",
    "map_50": "mAP@0.5",
    "mean_iou": "Mean IoU",
    "pixel_accuracy": "Pixel accuracy",
    "dice_score": "Dice score",
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "mrr": "Mean reciprocal rank",
    "proxy_score": "Proxy score",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "pixel_iou": "Pixel IoU",
    "normalized_edit_similarity": "Normalized edit similarity",
    "exact_match_accuracy": "Exact match accuracy",
    "cer": "Character error rate",
    "wer": "Word error rate",
}


def normalize_task_type(task_type: str) -> str:
    return (task_type or "").strip().lower()


def task_family(task_type: str) -> str:
    return _TASK_FAMILIES.get(normalize_task_type(task_type), "other")


def valid_metrics_for_task(task_type: str) -> list:
    return list(_TASK_METRICS.get(normalize_task_type(task_type), []))


def default_metric_for_task(task_type: str) -> str:
    return _DEFAULT_METRICS.get(normalize_task_type(task_type), "")


def metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def is_deprecated_task(task_type: str) -> bool:
    return normalize_task_type(task_type) in DEPRECATED_TASK_TYPES


def normalize_label_mode(label_mode: str) -> str:
    value = (label_mode or "").strip().lower()
    if value.startswith("multi"):
        return "multi-label"
    if value.startswith("single"):
        return "single-label"
    return ""


def resolve_image_task(task: str, label_mode: str = "") -> str:
    """Resolve a UI task label (or backend key) plus a label mode to a backend task key."""
    raw = (task or "").strip()
    if raw in _LEGACY_TASK_ALIASES:
        raw, label_mode = _LEGACY_TASK_ALIASES[raw]
    if raw in _IMG_TASK_BACKEND:
        backend = _IMG_TASK_BACKEND[raw]
    else:
        backend = normalize_task_type(raw)
    if backend == "classification" and normalize_label_mode(label_mode) == "multi-label":
        return "multilabel"
    return backend


def label_mode_for_task(task_type: str) -> str:
    key = normalize_task_type(task_type)
    if key == "classification":
        return "single-label"
    if key == "multilabel":
        return "multi-label"
    return ""


def task_display_name(task_type: str, label_mode: str = "") -> str:
    return TASK_DISPLAY_NAMES.get(normalize_task_type(task_type), task_type or "")


@dataclass
class ImageConfig:
    data_path: Path
    metric: str = ""
    task_type: str = "classification"
    domain: str = ""
    constraints: str = ""
    notes: str = ""
    modality: str = "Image"
    input_format: str = ""
    image_format: str = ""
    color_space: str = ""
    label_mode: str = ""

    @property
    def supervision(self) -> str:
        return "unsupervised" if normalize_task_type(self.task_type) == "retrieval" else "supervised"

    @property
    def resolved_label_mode(self) -> str:
        return normalize_label_mode(self.label_mode) or label_mode_for_task(self.task_type)

    @property
    def task_family(self) -> str:
        return task_family(self.task_type)

    @property
    def active_constraints(self) -> list:
        if not self.constraints:
            return []
        return [c.strip() for c in self.constraints.split(",") if c.strip()]

    def task_context(self) -> dict:
        task_type = normalize_task_type(self.task_type)
        return {
            "task_type": task_type,
            "task_family": task_family(task_type),
            "task_name": task_display_name(task_type),
            "label_mode": self.resolved_label_mode,
            "domain": self.domain,
            "constraints": self.constraints,
            "active_constraints": self.active_constraints,
            "notes": self.notes,
            "modality": self.modality,
            "input_format": self.input_format,
            "image_format": self.image_format,
            "color_space": self.color_space,
            "supervision": self.supervision,
        }
