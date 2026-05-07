from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_PROFILE_SAMPLE_LIMIT = 500


@dataclass
class ImageProfile:
    root_path: Path
    n_images: int
    n_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    imbalance_ratio: float
    min_class_size: int

    avg_height: float
    avg_width: float
    min_height: int
    min_width: int
    max_height: int
    max_width: int
    height_std: float
    width_std: float

    avg_aspect_ratio: float
    aspect_ratio_std: float

    dominant_color_channels: int
    grayscale_ratio: float
    rgba_ratio: float

    avg_brightness: float
    brightness_std: float
    avg_contrast: float
    contrast_std: float

    avg_file_size_kb: float
    min_file_size_kb: float
    max_file_size_kb: float

    n_corrupt: int
    corrupt_paths: List[str]

    has_varied_sizes: bool
    has_low_contrast: bool
    has_high_contrast_variance: bool
    has_varied_brightness: bool
    has_grayscale_images: bool
    has_mostly_grayscale: bool
    has_rgba_images: bool
    has_small_images: bool
    has_large_images: bool
    is_imbalanced: bool
    is_highly_imbalanced: bool
    has_corrupt_images: bool
    is_uniform_size: bool

    image_paths: List[str]
    image_labels: List[str]

    input_format: str = ""
    parsing_summary: Dict[str, Any] = field(default_factory=dict)
    annotation_profile: Dict[str, Any] = field(default_factory=dict)
    structure_profile: Dict[str, Any] = field(default_factory=dict)
    parser_warnings: List[str] = field(default_factory=list)
    class_mapping: Dict[int, str] = field(default_factory=dict)
    has_bboxes: bool = False
    has_masks: bool = False
    has_keypoints: bool = False
    has_text_labels: bool = False
    has_depth_targets: bool = False


def _collect_image_paths(root: Path) -> Tuple[List[Path], List[str]]:
    paths = []
    labels = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        class_name = sub.name
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS:
                paths.append(f)
                labels.append(class_name)
    return paths, labels


def profile_image_dataset(root: Path) -> ImageProfile:
    all_paths, all_labels = _collect_image_paths(root)
    n_images = len(all_paths)

    class_counts: Dict[str, int] = {}
    for label in all_labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    class_names = sorted(class_counts.keys())
    n_classes = len(class_names)
    counts_sorted = sorted(class_counts.values(), reverse=True)
    min_class_size = counts_sorted[-1] if counts_sorted else 1
    imbalance_ratio = (
        counts_sorted[0] / counts_sorted[-1]
        if counts_sorted and counts_sorted[-1] > 0 else float("inf")
    )

    sample_indices = list(range(n_images))
    if n_images > _PROFILE_SAMPLE_LIMIT:
        rng = np.random.RandomState(42)
        sample_indices = sorted(rng.choice(n_images, _PROFILE_SAMPLE_LIMIT, replace=False).tolist())

    heights = []
    widths = []
    brightnesses = []
    contrasts = []
    file_sizes_kb = []
    n_grayscale = 0
    n_rgba = 0
    n_corrupt = 0
    corrupt_paths = []
    channel_counts = []

    for idx in sample_indices:
        p = all_paths[idx]
        try:
            file_sizes_kb.append(p.stat().st_size / 1024.0)
        except Exception:
            file_sizes_kb.append(0.0)

        try:
            img = Image.open(str(p))
            img.load()
            w, h = img.size
            heights.append(h)
            widths.append(w)

            mode = img.mode
            if mode in ("L", "1", "P"):
                n_grayscale += 1
                channel_counts.append(1)
            elif mode == "RGBA":
                n_rgba += 1
                channel_counts.append(4)
            else:
                channel_counts.append(3)

            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32)
            brightnesses.append(float(arr.mean()) / 255.0)
            contrasts.append(float(arr.std()) / 255.0)
        except Exception:
            n_corrupt += 1
            corrupt_paths.append(str(p))

    n_sampled = len(heights)

    if n_sampled > 0:
        h_arr = np.array(heights, dtype=float)
        w_arr = np.array(widths, dtype=float)
        avg_height = float(h_arr.mean())
        avg_width = float(w_arr.mean())
        min_height = int(h_arr.min())
        min_width = int(w_arr.min())
        max_height = int(h_arr.max())
        max_width = int(w_arr.max())
        height_std = float(h_arr.std())
        width_std = float(w_arr.std())

        aspects = w_arr / np.maximum(h_arr, 1.0)
        avg_aspect_ratio = float(aspects.mean())
        aspect_ratio_std = float(aspects.std())

        b_arr = np.array(brightnesses)
        avg_brightness = float(b_arr.mean())
        brightness_std = float(b_arr.std())

        c_arr = np.array(contrasts)
        avg_contrast = float(c_arr.mean())
        contrast_std = float(c_arr.std())
    else:
        avg_height = avg_width = 0.0
        min_height = min_width = max_height = max_width = 0
        height_std = width_std = 0.0
        avg_aspect_ratio = 1.0
        aspect_ratio_std = 0.0
        avg_brightness = 0.5
        brightness_std = 0.0
        avg_contrast = 0.0
        contrast_std = 0.0

    if file_sizes_kb:
        fs_arr = np.array(file_sizes_kb)
        avg_file_size_kb = float(fs_arr.mean())
        min_file_size_kb = float(fs_arr.min())
        max_file_size_kb = float(fs_arr.max())
    else:
        avg_file_size_kb = min_file_size_kb = max_file_size_kb = 0.0

    n_sample_valid = max(n_sampled, 1)
    grayscale_ratio = n_grayscale / n_sample_valid
    rgba_ratio = n_rgba / n_sample_valid

    if channel_counts:
        from collections import Counter
        ch_counter = Counter(channel_counts)
        dominant_color_channels = ch_counter.most_common(1)[0][0]
    else:
        dominant_color_channels = 3

    size_cv = (height_std / max(avg_height, 1.0) + width_std / max(avg_width, 1.0)) / 2.0
    has_varied_sizes = size_cv > 0.15
    is_uniform_size = height_std < 1.0 and width_std < 1.0

    has_low_contrast = avg_contrast < 0.10
    has_high_contrast_variance = contrast_std > 0.08
    has_varied_brightness = brightness_std > 0.15
    has_grayscale_images = n_grayscale > 0
    has_mostly_grayscale = grayscale_ratio > 0.5
    has_rgba_images = n_rgba > 0
    has_small_images = min_height < 32 or min_width < 32 if n_sampled > 0 else False
    has_large_images = max_height > 1024 or max_width > 1024 if n_sampled > 0 else False
    is_imbalanced = imbalance_ratio > 1.5
    is_highly_imbalanced = imbalance_ratio > 3.0
    has_corrupt_images = n_corrupt > 0

    return ImageProfile(
        root_path=root,
        n_images=n_images,
        n_classes=n_classes,
        class_names=class_names,
        class_counts=class_counts,
        imbalance_ratio=imbalance_ratio,
        min_class_size=min_class_size,
        avg_height=avg_height,
        avg_width=avg_width,
        min_height=min_height,
        min_width=min_width,
        max_height=max_height,
        max_width=max_width,
        height_std=height_std,
        width_std=width_std,
        avg_aspect_ratio=avg_aspect_ratio,
        aspect_ratio_std=aspect_ratio_std,
        dominant_color_channels=dominant_color_channels,
        grayscale_ratio=grayscale_ratio,
        rgba_ratio=rgba_ratio,
        avg_brightness=avg_brightness,
        brightness_std=brightness_std,
        avg_contrast=avg_contrast,
        contrast_std=contrast_std,
        avg_file_size_kb=avg_file_size_kb,
        min_file_size_kb=min_file_size_kb,
        max_file_size_kb=max_file_size_kb,
        n_corrupt=n_corrupt,
        corrupt_paths=corrupt_paths[:10],
        has_varied_sizes=has_varied_sizes,
        has_low_contrast=has_low_contrast,
        has_high_contrast_variance=has_high_contrast_variance,
        has_varied_brightness=has_varied_brightness,
        has_grayscale_images=has_grayscale_images,
        has_mostly_grayscale=has_mostly_grayscale,
        has_rgba_images=has_rgba_images,
        has_small_images=has_small_images,
        has_large_images=has_large_images,
        is_imbalanced=is_imbalanced,
        is_highly_imbalanced=is_highly_imbalanced,
        has_corrupt_images=has_corrupt_images,
        is_uniform_size=is_uniform_size,
        image_paths=[str(p) for p in all_paths],
        image_labels=all_labels,
    )
