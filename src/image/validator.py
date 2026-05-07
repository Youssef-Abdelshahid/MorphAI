import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from .config import (
    SUPPORTED_TASK_TYPES,
    VALID_TASK_TYPES,
    default_metric_for_task,
    normalize_task_type,
    valid_metrics_for_task,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_ANNOTATION_EXTENSIONS = {".json", ".txt", ".xml", ".csv", ".npy", ".npz"}


def _scan_image_folder(root: Path) -> Tuple[List[str], dict, int]:
    classes = []
    class_counts = {}
    total = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        imgs = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]
        if imgs:
            classes.append(sub.name)
            class_counts[sub.name] = len(imgs)
            total += len(imgs)
    return classes, class_counts, total


def _annotation_summary(root: Path) -> Dict[str, int]:
    summary = {
        "json": 0,
        "txt": 0,
        "xml": 0,
        "csv": 0,
        "npy": 0,
        "npz": 0,
        "mask_like": 0,
        "depth_like": 0,
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _ANNOTATION_EXTENSIONS:
            summary[suffix.lstrip(".")] += 1
        name = path.stem.lower()
        if any(token in name for token in ["mask", "label", "seg", "instance"]):
            summary["mask_like"] += 1
        if any(token in name for token in ["depth", "disp", "disparity"]):
            summary["depth_like"] += 1
    return summary


def _task_validation_hints(task_type: str, annotations: Dict[str, int], classes: List[str]) -> List[str]:
    errors: List[str] = []
    if task_type == "classification":
        if len(classes) < 2:
            errors.append("Single-label classification needs at least 2 class folders with images.")
    elif task_type == "multilabel":
        if len(classes) < 1:
            errors.append("Multi-label classification needs image folders or another image source.")
    elif task_type == "detection":
        if annotations["json"] + annotations["xml"] + annotations["txt"] <= 0:
            errors.append("Object detection requires bounding box annotations (JSON, XML, or TXT).")
    elif task_type == "semantic_segmentation":
        if annotations["mask_like"] + annotations["json"] + annotations["png"] if False else 0:
            pass
        if annotations["mask_like"] + annotations["json"] + annotations["npy"] + annotations["npz"] <= 0:
            errors.append("Semantic segmentation requires segmentation masks or equivalent annotation files.")
    elif task_type == "instance_segmentation":
        if annotations["mask_like"] + annotations["json"] + annotations["xml"] <= 0:
            errors.append("Instance segmentation requires instance masks or instance-level annotation files.")
    elif task_type == "keypoint":
        if annotations["json"] + annotations["txt"] + annotations["csv"] <= 0:
            errors.append("Keypoint / pose estimation requires keypoint annotations.")
    elif task_type == "retrieval":
        if len(classes) < 2 and annotations["json"] + annotations["csv"] <= 0:
            errors.append("Image retrieval needs labels, query-gallery structure, or retrieval pairs.")
    elif task_type == "anomaly":
        if len(classes) < 1:
            errors.append("Anomaly / defect detection needs at least one image folder.")
    elif task_type == "ocr":
        if annotations["txt"] + annotations["json"] + annotations["csv"] <= 0:
            errors.append("OCR requires text transcriptions or OCR label files.")
    elif task_type == "generation":
        if len(classes) < 1:
            errors.append("Image generation / synthesis needs at least one folder of images.")
    elif task_type == "depth":
        if annotations["depth_like"] + annotations["npy"] + annotations["npz"] <= 0:
            errors.append("Depth estimation requires depth maps or depth annotation files.")
    return errors


def validate_image_zip(zip_path: Path) -> list:
    errors = []

    if not zip_path.exists():
        errors.append(f"File does not exist: {zip_path}")
        return errors

    if zip_path.suffix.lower() != ".zip":
        errors.append(
            f"Image modality requires a .zip file. "
            f"Got: '{zip_path.suffix or zip_path.name}'. "
            "Provide the dataset as a single .zip archive."
        )
        return errors

    try:
        valid = zipfile.is_zipfile(str(zip_path))
    except Exception as exc:
        errors.append(f"Cannot read file: {exc}")
        return errors

    if not valid:
        errors.append(f"'{zip_path.name}' is not a valid zip archive.")
        return errors

    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            bad = zf.testzip()
            if bad:
                errors.append(f"Corrupted entry detected in zip archive: '{bad}'.")
                return errors
            names = zf.namelist()
    except zipfile.BadZipFile as exc:
        errors.append(f"Corrupted zip archive: {exc}")
        return errors
    except Exception as exc:
        errors.append(f"Cannot open zip archive: {exc}")
        return errors

    image_names = [
        n for n in names
        if Path(n).suffix.lower() in _IMAGE_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX" for part in Path(n).parts)
    ]
    if not image_names:
        errors.append(
            "Zip archive contains no valid image files. "
            f"Supported formats: {', '.join(sorted(_IMAGE_EXTENSIONS))}."
        )
    return errors


def validate_image_run(config, root: Path) -> list:
    errors = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()

    if not root.exists():
        errors.append(f"Path does not exist: {root}")
        return errors
    if not root.is_dir():
        errors.append(f"Path is not a directory: {root}")
        return errors

    if not task_type:
        errors.append("An image task type is required.")
    elif task_type not in VALID_TASK_TYPES:
        errors.append(
            f"Task type '{config.task_type}' is not valid for image data. "
            f"Supported task types: {sorted(SUPPORTED_TASK_TYPES)}"
        )
    elif task_type not in SUPPORTED_TASK_TYPES:
        errors.append(
            f"Task type '{task_type}' is not yet supported for image data. "
            f"Supported task types: {sorted(SUPPORTED_TASK_TYPES)}"
        )

    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics:
        if not metric:
            errors.append(
                f"A priority metric is required for an image task of type '{task_type}'. "
                f"Suggested default: {default_metric_for_task(task_type)}"
            )
        elif metric not in valid_metrics:
            errors.append(
                f"Metric '{config.metric}' is not valid for '{task_type}'. "
                f"Valid metrics: {valid_metrics}"
            )

    classes, class_counts, total = _scan_image_folder(root)
    if total < 10:
        errors.append(f"Dataset has only {total} images. At least 10 are required.")

    if task_type in {"classification", "multilabel", "retrieval", "anomaly"}:
        small_classes = [c for c, n in class_counts.items() if n < 2]
        if small_classes and task_type in {"classification", "retrieval"}:
            errors.append(
                f"Classes with fewer than 2 images: {small_classes}. "
                "Each class needs at least 2 samples for evaluation."
            )

    annotations = _annotation_summary(root)
    errors.extend(_task_validation_hints(task_type, annotations, classes))
    return errors


def validate_internal_dataset(config, dataset) -> list:
    errors: List[str] = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()
    input_format = getattr(dataset, "input_format", "") or ""

    if not task_type:
        errors.append("An image task type is required.")
        return errors
    if task_type not in SUPPORTED_TASK_TYPES:
        errors.append(
            f"Task type '{task_type}' is not yet supported for image data. "
            f"Supported task types: {sorted(SUPPORTED_TASK_TYPES)}"
        )
        return errors

    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics:
        if not metric:
            errors.append(
                f"A priority metric is required for an image task of type '{task_type}'. "
                f"Suggested default: {default_metric_for_task(task_type)}"
            )
        elif metric not in valid_metrics:
            errors.append(
                f"Metric '{config.metric}' is not valid for '{task_type}'. "
                f"Valid metrics: {valid_metrics}"
            )

    samples = getattr(dataset, "samples", []) or []
    if len(samples) < 10:
        errors.append(f"Parsed dataset has only {len(samples)} images. At least 10 are required.")
        return errors

    has_bboxes = any(s.bboxes for s in samples)
    has_masks = any(s.masks for s in samples)
    has_keypoints = any(s.keypoints for s in samples)
    has_class_labels = any(s.labels for s in samples)
    has_text = any(s.transcription for s in samples)
    has_depth = any(s.depth_path for s in samples)

    if task_type == "classification":
        if not has_class_labels:
            errors.append(
                "Single-label classification requires image-level labels. "
                f"The selected input format '{input_format}' does not provide image-level class labels."
            )
        else:
            class_counts: Dict[str, int] = {}
            for sample in samples:
                if sample.labels:
                    cls = sample.labels[0]
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            small = [c for c, n in class_counts.items() if n < 2]
            if len(class_counts) < 2:
                errors.append("Single-label classification needs at least 2 distinct classes.")
            if small:
                errors.append(
                    f"Classes with fewer than 2 images: {small}. "
                    "Each class needs at least 2 samples for evaluation."
                )
    elif task_type == "multilabel":
        if not has_class_labels:
            errors.append(
                "Multi-label classification requires image labels but none were provided "
                f"by the '{input_format}' input format."
            )
    elif task_type == "detection":
        if not has_bboxes:
            errors.append(
                f"The selected input format '{input_format}' does not provide the bounding box "
                "annotations required for object detection."
            )
    elif task_type == "semantic_segmentation":
        if not has_masks:
            errors.append(
                f"The selected input format '{input_format}' does not provide the segmentation "
                "masks required for semantic segmentation."
            )
    elif task_type == "instance_segmentation":
        if not has_masks:
            errors.append(
                f"The selected input format '{input_format}' does not provide the instance "
                "masks required for instance segmentation."
            )
    elif task_type == "keypoint":
        if not has_keypoints:
            errors.append(
                f"The selected input format '{input_format}' does not provide the keypoint "
                "annotations required for keypoint / pose estimation."
            )
    elif task_type == "retrieval":
        if not has_class_labels:
            errors.append(
                "Image retrieval needs class labels or query/gallery structure. The selected "
                f"input format '{input_format}' does not provide them."
            )
    elif task_type == "anomaly":
        if not has_class_labels and not has_masks:
            errors.append(
                "Anomaly / defect detection needs labels or masks indicating normal/abnormal samples."
            )
    elif task_type == "ocr":
        if not has_text:
            errors.append(
                f"OCR requires transcriptions / text labels which are not provided by '{input_format}'."
            )
    elif task_type == "generation":
        pass
    elif task_type == "depth":
        if not has_depth:
            errors.append(
                f"Depth estimation requires depth maps / targets which are not provided by '{input_format}'."
            )

    return errors
