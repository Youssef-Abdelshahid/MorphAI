from dataclasses import dataclass
from pathlib import Path

SINGLE_LABEL_CLASSIFICATION_METRICS = ["accuracy", "macro_f1", "weighted_f1", "precision", "recall"]
MULTILABEL_CLASSIFICATION_METRICS = ["micro_f1", "macro_f1", "hamming_loss", "subset_accuracy"]
OBJECT_DETECTION_METRICS = ["map", "map_50", "precision", "recall", "mean_iou"]
SEMANTIC_SEGMENTATION_METRICS = ["mean_iou", "pixel_accuracy", "dice_score"]
INSTANCE_SEGMENTATION_METRICS = ["mask_map", "mask_iou", "dice_score", "precision", "recall"]
KEYPOINT_METRICS = ["pck", "oks_map", "normalized_keypoint_error"]
RETRIEVAL_METRICS = ["recall_at_k", "precision_at_k", "map", "mrr"]
ANOMALY_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "proxy_score"]
OCR_METRICS = ["normalized_edit_similarity", "exact_match_accuracy", "cer", "wer"]
GENERATION_METRICS = ["clip_similarity", "ssim", "psnr", "fid", "lpips"]
DEPTH_METRICS = ["delta_accuracy", "rmse", "mae", "abs_rel"]

_IMG_TASK_BACKEND = {
    "Image classification (single-label)": "classification",
    "Image classification (multi-label)": "multilabel",
    "Object detection": "detection",
    "Semantic segmentation": "semantic_segmentation",
    "Instance segmentation": "instance_segmentation",
    "Keypoint / pose estimation": "keypoint",
    "Image similarity / retrieval": "retrieval",
    "Anomaly / defect detection": "anomaly",
    "Optical character recognition": "ocr",
    "Image generation / synthesis": "generation",
    "Depth estimation": "depth",
}

VALID_TASK_TYPES = list(_IMG_TASK_BACKEND.values())
SUPPORTED_TASK_TYPES = set(VALID_TASK_TYPES)

_TASK_FAMILIES = {
    "classification": "classification",
    "multilabel": "classification",
    "detection": "detection",
    "semantic_segmentation": "segmentation",
    "instance_segmentation": "segmentation",
    "keypoint": "keypoint",
    "retrieval": "retrieval",
    "anomaly": "anomaly",
    "ocr": "ocr",
    "generation": "generation",
    "depth": "depth",
}

_TASK_METRICS = {
    "classification": SINGLE_LABEL_CLASSIFICATION_METRICS,
    "multilabel": MULTILABEL_CLASSIFICATION_METRICS,
    "detection": OBJECT_DETECTION_METRICS,
    "semantic_segmentation": SEMANTIC_SEGMENTATION_METRICS,
    "instance_segmentation": INSTANCE_SEGMENTATION_METRICS,
    "keypoint": KEYPOINT_METRICS,
    "retrieval": RETRIEVAL_METRICS,
    "anomaly": ANOMALY_METRICS,
    "ocr": OCR_METRICS,
    "generation": GENERATION_METRICS,
    "depth": DEPTH_METRICS,
}

_DEFAULT_METRICS = {
    "classification": "macro_f1",
    "multilabel": "micro_f1",
    "detection": "map",
    "semantic_segmentation": "mean_iou",
    "instance_segmentation": "mask_map",
    "keypoint": "pck",
    "retrieval": "recall_at_k",
    "anomaly": "auroc",
    "ocr": "normalized_edit_similarity",
    "generation": "clip_similarity",
    "depth": "delta_accuracy",
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
    "mask_map": "Mask mAP",
    "mask_iou": "Mask IoU",
    "pck": "PCK",
    "oks_map": "OKS mAP",
    "normalized_keypoint_error": "Normalized keypoint error",
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
    "clip_similarity": "CLIP similarity",
    "ssim": "SSIM",
    "psnr": "PSNR",
    "fid": "FID",
    "lpips": "LPIPS",
    "delta_accuracy": "Delta accuracy",
    "rmse": "RMSE",
    "mae": "MAE",
    "abs_rel": "Absolute relative error",
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

    @property
    def supervision(self) -> str:
        return "unsupervised" if normalize_task_type(self.task_type) in {"retrieval", "generation"} else "supervised"

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
