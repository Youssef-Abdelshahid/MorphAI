import io
import json
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from .config import ImageConfig
from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

PROCESSED_DIR = Path("processed")

_FORMAT_FILENAME_TOKENS = {
    "zip_folder": "folder",
    "image_folder_zip": "folder",
    "coco": "coco",
    "pascal_voc": "voc",
    "yolo": "yolo",
}


def _pipeline_short_id(spec: ImagePipelineSpec) -> str:
    parts = [
        f"sz{spec.resize}",
        spec.color_mode[:3],
        spec.normalization[:3] if spec.normalization != "none" else "raw",
    ]
    if spec.histogram_eq:
        parts.append("heq")
    if spec.denoise:
        parts.append("dns")
    if spec.sharpen:
        parts.append("shp")
    if spec.augment_h_flip:
        parts.append("hfl")
    if spec.augment_rotation != "none":
        parts.append("rot")
    if spec.imbalance != "none":
        parts.append(spec.imbalance[:3])
    return "_".join(parts)


def _preprocess_for_output(img: Image.Image, spec: ImagePipelineSpec) -> Image.Image:
    if spec.color_mode == "grayscale":
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    if spec.resize > 0:
        img = img.resize((spec.resize, spec.resize), Image.LANCZOS)

    if spec.histogram_eq:
        try:
            from PIL import ImageOps
            img = ImageOps.equalize(img)
        except Exception:
            pass

    if spec.denoise:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    if spec.sharpen:
        img = img.filter(ImageFilter.SHARPEN)

    arr = np.asarray(img, dtype=np.float32)

    if spec.normalization == "standard":
        mean = arr.mean()
        std = arr.std()
        if std > 0:
            arr = (arr - mean) / std
        else:
            arr = arr - mean
        arr_min, arr_max = arr.min(), arr.max()
        rng = arr_max - arr_min
        if rng > 0:
            arr = (arr - arr_min) / rng * 255.0
        else:
            arr = np.zeros_like(arr)
    elif spec.normalization == "minmax":
        mn, mx = arr.min(), arr.max()
        rng = mx - mn
        if rng > 0:
            arr = (arr - mn) / rng * 255.0
        else:
            arr = arr * 0.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if spec.color_mode == "grayscale":
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def save_processed_dataset(
    spec: ImagePipelineSpec,
    profile: ImageProfile,
    config: ImageConfig,
    internal_dataset=None,
) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stem = config.data_path.stem
    input_fmt = (getattr(profile, "input_format", "") or "image_folder_zip").replace(" ", "_")
    task_token = (getattr(config, "task_type", "") or "task").replace(" ", "_")
    fmt_token = _FORMAT_FILENAME_TOKENS.get(input_fmt, input_fmt)
    out_path = PROCESSED_DIR / f"image_{fmt_token}_{task_token}_cleaned.zip"

    n_saved = 0
    used_names: dict = {}

    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for path_str, label in zip(profile.image_paths, profile.image_labels):
            try:
                img = Image.open(path_str)
                img.load()
                processed = _preprocess_for_output(img, spec)

                src_stem = Path(path_str).stem
                base_name = f"images/{label or 'all'}/{src_stem}.png"
                if base_name in used_names:
                    used_names[base_name] += 1
                    arc_name = f"images/{label or 'all'}/{src_stem}_{used_names[base_name]}.png"
                else:
                    used_names[base_name] = 0
                    arc_name = base_name

                buf = io.BytesIO()
                processed.save(buf, format="PNG")
                zout.writestr(arc_name, buf.getvalue())
                n_saved += 1
            except Exception:
                continue

        if internal_dataset is not None:
            try:
                annotations_payload = _internal_dataset_payload(internal_dataset)
                zout.writestr("annotations/internal_annotations.json", json.dumps(annotations_payload, indent=2))
            except Exception:
                pass

        metadata = {
            "modality": "image",
            "input_format": getattr(profile, "input_format", "") or input_fmt,
            "task_type": getattr(config, "task_type", ""),
            "selected_pipeline": spec.name(),
            "pipeline_config": spec.to_dict(),
            "parsing_summary": dict(getattr(profile, "parsing_summary", {}) or {}),
            "annotation_profile": dict(getattr(profile, "annotation_profile", {}) or {}),
            "structure_profile": dict(getattr(profile, "structure_profile", {}) or {}),
            "class_mapping": {str(k): v for k, v in (getattr(profile, "class_mapping", {}) or {}).items()},
            "warnings": list(getattr(profile, "parser_warnings", []) or []),
            "output_structure": "images/<class>/<image>.png",
        }
        zout.writestr("metadata.json", json.dumps(metadata, indent=2))

    if n_saved == 0:
        raise ValueError("No images could be processed and saved.")

    n_classes = len(set(profile.image_labels)) or 1
    return out_path, (n_saved, n_classes)


def _internal_dataset_payload(internal_dataset) -> dict:
    samples = []
    for s in getattr(internal_dataset, "samples", []) or []:
        samples.append({
            "image_id": s.image_id,
            "image_path": Path(s.image_path).name,
            "width": s.width,
            "height": s.height,
            "labels": s.labels,
            "bboxes": [list(b) for b in s.bboxes],
            "bbox_classes": s.bbox_classes,
            "keypoints": [[list(p) for p in instance] for instance in s.keypoints],
            "transcription": s.transcription,
            "split": s.split,
        })
    return {
        "input_format": getattr(internal_dataset, "input_format", ""),
        "original_format": getattr(internal_dataset, "original_format", ""),
        "class_mapping": {str(k): v for k, v in (getattr(internal_dataset, "class_mapping", {}) or {}).items()},
        "samples": samples,
    }
