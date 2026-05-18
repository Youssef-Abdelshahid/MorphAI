from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ImageSample:
    image_path: str
    image_id: str
    width: int = 0
    height: int = 0
    labels: List[str] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    bbox_classes: List[str] = field(default_factory=list)
    masks: List[Any] = field(default_factory=list)
    keypoints: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    transcription: Optional[str] = None
    depth_path: Optional[str] = None
    split: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalImageDataset:
    modality: str = "image"
    input_format: str = ""
    original_format: str = ""
    samples: List[ImageSample] = field(default_factory=list)
    class_mapping: Dict[int, str] = field(default_factory=dict)
    structure_profile: Dict[str, Any] = field(default_factory=dict)
    parsing_summary: Dict[str, Any] = field(default_factory=dict)
    annotation_profile: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dataset_root: Optional[Path] = None
    raw_root: Optional[Path] = None


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(text or ""))
    return text.strip("._") or "item"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS and not any(
        part.startswith(".") or part == "__MACOSX" for part in path.parts
    )


def extract_zip(zip_path: Path, dest: Path) -> List[str]:
    warnings: List[str] = []
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        bad = zf.testzip()
        if bad:
            warnings.append(f"Corrupted entry in zip: {bad}")
        zf.extractall(str(dest))
    return warnings


def find_dataset_root(extracted: Path) -> Path:
    try:
        entries = [e for e in extracted.iterdir() if e.name not in {"__MACOSX"} and not e.name.startswith(".")]
    except Exception:
        return extracted
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def collect_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def safe_image_size(path: Path) -> Tuple[int, int]:
    try:
        with Image.open(str(path)) as img:
            img.load()
            return int(img.size[0]), int(img.size[1])
    except Exception:
        return 0, 0


def normalize_xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def clamp_box(xyxy: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    if width > 0:
        x1 = max(0.0, min(float(width), x1))
        x2 = max(0.0, min(float(width), x2))
    if height > 0:
        y1 = max(0.0, min(float(height), y1))
        y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def derive_image_label(sample: ImageSample) -> str:
    if sample.labels:
        return sample.labels[0]
    if sample.bbox_classes:
        counter = Counter(sample.bbox_classes)
        return counter.most_common(1)[0][0]
    return "all"


def annotation_profile_from_samples(
    samples: List[ImageSample],
    class_mapping: Dict[int, str],
    parsed_image_count: int,
    original_image_count: int,
    corrupted: int,
    missing: int,
    unused: int,
    invalid_bbox: int,
    invalid_mask: int,
    invalid_keypoint: int,
    unmapped_class: int,
    unsupported_image: int,
    warnings: List[str],
) -> Dict[str, Any]:
    bbox_count = sum(len(s.bboxes) for s in samples)
    mask_count = sum(len(s.masks) for s in samples)
    keypoint_count = sum(len(s.keypoints) for s in samples)
    images_without = sum(1 for s in samples if not s.bboxes and not s.masks and not s.keypoints and not s.labels)
    distribution: Counter = Counter()
    for s in samples:
        distribution.update(s.bbox_classes or s.labels)
    sizes = [(s.width, s.height) for s in samples if s.width and s.height]
    aspect = [w / h for (w, h) in sizes if h > 0]
    splits: Counter = Counter(s.split or "default" for s in samples)
    return {
        "input_format": "",
        "original_image_count": int(original_image_count),
        "parsed_image_count": int(parsed_image_count),
        "corrupted_image_count": int(corrupted),
        "missing_image_count": int(missing),
        "unused_image_count": int(unused),
        "annotation_count": int(bbox_count + mask_count + keypoint_count + sum(1 for s in samples if s.transcription) + sum(1 for s in samples if s.depth_path)),
        "invalid_annotation_count": int(invalid_bbox + invalid_mask + invalid_keypoint),
        "category_count": len(class_mapping) or len(distribution),
        "class_distribution": dict(distribution),
        "bbox_count": bbox_count,
        "invalid_bbox_count": int(invalid_bbox),
        "average_boxes_per_image": (bbox_count / parsed_image_count) if parsed_image_count else 0.0,
        "images_without_annotations": images_without,
        "segmentation_count": mask_count,
        "invalid_mask_count": int(invalid_mask),
        "keypoint_count": keypoint_count,
        "invalid_keypoint_count": int(invalid_keypoint),
        "unmapped_class_count": int(unmapped_class),
        "unsupported_image_count": int(unsupported_image),
        "split_summary": dict(splits),
        "annotation_density": (bbox_count / parsed_image_count) if parsed_image_count else 0.0,
        "image_size_distribution": {
            "min_width": min((w for (w, _) in sizes), default=0),
            "max_width": max((w for (w, _) in sizes), default=0),
            "min_height": min((h for (_, h) in sizes), default=0),
            "max_height": max((h for (_, h) in sizes), default=0),
        },
        "aspect_ratio_distribution": {
            "min": float(min(aspect)) if aspect else 0.0,
            "max": float(max(aspect)) if aspect else 0.0,
            "mean": float(sum(aspect) / len(aspect)) if aspect else 0.0,
        },
        "parsing_warnings": list(warnings),
    }


def has_class_label(samples: List[ImageSample]) -> bool:
    return any(s.labels for s in samples)


def has_bboxes(samples: List[ImageSample]) -> bool:
    return any(s.bboxes for s in samples)


def has_masks(samples: List[ImageSample]) -> bool:
    return any(s.masks for s in samples)


def has_keypoints(samples: List[ImageSample]) -> bool:
    return any(s.keypoints for s in samples)


def has_text_labels(samples: List[ImageSample]) -> bool:
    return any(s.transcription for s in samples)


def has_depth_targets(samples: List[ImageSample]) -> bool:
    return any(s.depth_path for s in samples)


def materialize_for_pipeline(
    dataset: InternalImageDataset,
    work_dir: Path,
    task_type: str,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    task_type = (task_type or "").strip().lower()

    if task_type in {"classification", "multilabel", "retrieval", "anomaly", "generation"}:
        return _materialize_class_folders(dataset, work_dir)
    return _materialize_flat_with_sidecars(dataset, work_dir, task_type)


def _materialize_class_folders(dataset: InternalImageDataset, work_dir: Path) -> Path:
    used: Dict[str, int] = {}
    for sample in dataset.samples:
        src = Path(sample.image_path)
        if not src.exists():
            continue
        label = safe_name(derive_image_label(sample))
        class_dir = work_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lower() or ".jpg"
        stem = safe_name(src.stem)
        base = f"{stem}{ext}"
        if base in used:
            used[base] += 1
            target = class_dir / f"{stem}_{used[base]}{ext}"
        else:
            used[base] = 0
            target = class_dir / base
        try:
            shutil.copy2(src, target)
        except Exception:
            continue
    return work_dir


_SIDECAR_EXTS = {".json", ".xml", ".txt", ".csv", ".png", ".npy", ".npz", ".tif", ".tiff", ".bmp"}


def _materialize_flat_with_sidecars(dataset: InternalImageDataset, work_dir: Path, task_type: str) -> Path:
    images_dir = work_dir / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    used: Dict[str, int] = {}
    has_sample_annotations = any(
        bool(s.bboxes) or bool(s.masks) or bool(s.keypoints) or bool(s.transcription) or bool(s.depth_path)
        for s in dataset.samples
    )
    for sample in dataset.samples:
        src = Path(sample.image_path)
        if not src.exists():
            continue
        ext = src.suffix.lower() or ".jpg"
        stem = safe_name(src.stem) or safe_name(sample.image_id) or "img"
        base = f"{stem}{ext}"
        if base in used:
            used[base] += 1
            target_name = f"{stem}_{used[base]}{ext}"
        else:
            used[base] = 0
            target_name = base
        target = images_dir / target_name
        try:
            shutil.copy2(src, target)
        except Exception:
            continue
        target_stem = target.stem
        if has_sample_annotations:
            _write_sidecar(images_dir, target_stem, sample, task_type)
        else:
            _copy_sibling_sidecars(src, images_dir, target_stem)
    return work_dir


def _copy_sibling_sidecars(image_src: Path, dest_dir: Path, target_stem: str) -> None:
    parent = image_src.parent
    src_stem = image_src.stem
    if not parent.exists() or not parent.is_dir():
        return
    for entry in parent.iterdir():
        if not entry.is_file() or entry == image_src:
            continue
        suf = entry.suffix.lower()
        if suf not in _SIDECAR_EXTS:
            continue
        if suf in SUPPORTED_IMAGE_EXTS and not any(
            tok in entry.stem.lower() for tok in ("mask", "seg", "label", "depth")
        ):
            continue
        es = entry.stem
        if es == src_stem:
            new_name = f"{target_stem}{suf}"
        elif es.startswith(f"{src_stem}_") or es.startswith(src_stem):
            tail = es[len(src_stem):]
            new_name = f"{target_stem}{tail}{suf}"
        else:
            continue
        try:
            shutil.copy2(entry, dest_dir / new_name)
        except Exception:
            continue


def _write_sidecar(folder: Path, stem: str, sample: ImageSample, task_type: str) -> None:
    if task_type == "detection":
        boxes = []
        for (x1, y1, x2, y2), cls in zip(sample.bboxes, sample.bbox_classes or [""] * len(sample.bboxes)):
            boxes.append({"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, "class": cls})
        if boxes:
            payload = {"image_id": sample.image_id, "boxes": boxes}
            (folder / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
        return
    if task_type == "semantic_segmentation":
        for idx, mask in enumerate(sample.masks):
            _materialize_mask(mask, folder, f"{stem}_mask_{idx}", sample.width, sample.height)
        return
    if task_type == "ocr":
        if sample.transcription is not None:
            (folder / f"{stem}_text.txt").write_text(str(sample.transcription), encoding="utf-8")
        return


def _materialize_mask(mask_obj: Any, folder: Path, stem: str, width: int, height: int) -> Optional[Path]:
    try:
        from PIL import ImageDraw

        if isinstance(mask_obj, dict) and "path" in mask_obj:
            src = Path(mask_obj["path"])
            if src.exists():
                target = folder / f"{stem}{src.suffix.lower() or '.png'}"
                shutil.copy2(src, target)
                return target
            return None
        if isinstance(mask_obj, list) and width > 0 and height > 0:
            img = Image.new("L", (width, height), color=0)
            drawer = ImageDraw.Draw(img)
            for polygon in mask_obj:
                if not polygon or len(polygon) < 6:
                    continue
                pts = [(float(polygon[i]), float(polygon[i + 1])) for i in range(0, len(polygon) - 1, 2)]
                if len(pts) >= 3:
                    drawer.polygon(pts, fill=255)
            target = folder / f"{stem}.png"
            img.save(str(target))
            return target
    except Exception:
        return None
    return None




