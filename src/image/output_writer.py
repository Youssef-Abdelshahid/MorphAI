import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from .config import ImageConfig
from .preprocessing import ImagePipelineSpec
from .profiler import ImageProfile

PROCESSED_DIR = Path("processed")


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
) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_stem = config.data_path.stem
    pid = _pipeline_short_id(spec)
    out_path = PROCESSED_DIR / f"{dataset_stem}_{pid}_processed.zip"

    n_saved = 0
    used_names: dict = {}

    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for path_str, label in zip(profile.image_paths, profile.image_labels):
            try:
                img = Image.open(path_str)
                img.load()
                processed = _preprocess_for_output(img, spec)

                src_stem = Path(path_str).stem
                base_name = f"{label}/{src_stem}.png"
                if base_name in used_names:
                    used_names[base_name] += 1
                    arc_name = f"{label}/{src_stem}_{used_names[base_name]}.png"
                else:
                    used_names[base_name] = 0
                    arc_name = base_name

                buf = io.BytesIO()
                processed.save(buf, format="PNG")
                zout.writestr(arc_name, buf.getvalue())
                n_saved += 1
            except Exception:
                continue

    if n_saved == 0:
        raise ValueError("No images could be processed and saved.")

    n_classes = len(set(profile.image_labels))
    return out_path, (n_saved, n_classes)
