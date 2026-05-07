from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.image.internal import (
    InternalImageDataset,
    ImageSample,
    SUPPORTED_IMAGE_EXTS,
    annotation_profile_from_samples,
    extract_zip,
    find_dataset_root,
    safe_image_size,
)


class YoloImageAdapter(BaseFormatAdapter):
    modality = "Image"
    input_format = "YOLO annotation format"
    is_implemented = True
    format_key = "yolo"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("YOLO ingestion requires a .zip archive containing images, label .txt files, and a class config.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not zipfile.is_zipfile(str(path)):
            errors.append(f"'{path.name}' is not a valid zip archive.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                names = zf.namelist()
        except Exception as exc:
            errors.append(f"Cannot open zip archive: {exc}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not any(n.lower().endswith(".txt") for n in names):
            errors.append("Archive does not contain any YOLO label .txt files.")
        if not any(Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTS for n in names):
            errors.append("Archive does not contain any image files.")
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(
        self,
        path: Path,
        work_dir: Path = None,
        image_dir: str = "",
        label_dir: str = "",
        class_config: str = "",
        split: str = "",
        **kwargs,
    ) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        if work_dir is None:
            return AdapterResult(ok=False, message="work_dir is required for image ingestion.", errors=["work_dir is required."])
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        warnings: List[str] = list(extract_zip(path, work_dir))
        root = find_dataset_root(work_dir)

        class_names = _load_class_names(root, class_config, warnings)
        if not class_names:
            return AdapterResult(
                ok=False,
                message="No YOLO class names found. Expected data.yaml or classes.txt.",
                errors=["YOLO class names file not found. Provide data.yaml or classes.txt."],
            )

        image_root = (root / image_dir).resolve() if image_dir else root
        label_root = (root / label_dir).resolve() if label_dir else root
        if image_dir and not image_root.exists():
            warnings.append(f"image_dir not found, falling back to archive root: {image_dir}")
            image_root = root
        if label_dir and not label_root.exists():
            warnings.append(f"label_dir not found, falling back to archive root: {label_dir}")
            label_root = root

        image_index: Dict[str, Path] = {}
        for p in image_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_index[p.stem.lower()] = p

        label_index: Dict[str, Path] = {}
        for p in label_root.rglob("*.txt"):
            if p.is_file() and p.name.lower() not in {"classes.txt", "names.txt"}:
                label_index[p.stem.lower()] = p

        samples: List[ImageSample] = []
        invalid_bbox = 0
        missing_image = 0
        unmapped = 0
        all_classes: Counter = Counter()

        for stem, label_path in label_index.items():
            img_path = image_index.get(stem)
            if img_path is None:
                missing_image += 1
                continue
            width, height = safe_image_size(img_path)
            sample_bboxes: List = []
            sample_classes: List[str] = []
            try:
                lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                parts = line.strip().split()
                if not parts or len(parts) < 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                except (TypeError, ValueError):
                    invalid_bbox += 1
                    continue
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    invalid_bbox += 1
                    continue
                if cls_id < 0 or cls_id >= len(class_names):
                    unmapped += 1
                    continue
                cls_name = class_names[cls_id]
                if width and height:
                    x1 = (cx - bw / 2.0) * width
                    y1 = (cy - bh / 2.0) * height
                    x2 = (cx + bw / 2.0) * width
                    y2 = (cy + bh / 2.0) * height
                else:
                    x1, y1, x2, y2 = cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
                sample_bboxes.append((x1, y1, x2, y2))
                sample_classes.append(cls_name)
                all_classes[cls_name] += 1

            labels: List[str] = []
            if sample_classes:
                labels = [Counter(sample_classes).most_common(1)[0][0]]

            samples.append(
                ImageSample(
                    image_path=str(img_path),
                    image_id=stem,
                    width=width,
                    height=height,
                    labels=labels,
                    bboxes=sample_bboxes,
                    bbox_classes=sample_classes,
                    split=split,
                )
            )

        used_stems = {s.image_id for s in samples}
        unused = sum(1 for stem in image_index if stem not in used_stems)

        class_mapping = {idx: name for idx, name in enumerate(class_names)}
        ann_profile = annotation_profile_from_samples(
            samples,
            class_mapping,
            parsed_image_count=len(samples),
            original_image_count=len(image_index),
            corrupted=0,
            missing=missing_image,
            unused=max(unused, 0),
            invalid_bbox=invalid_bbox,
            invalid_mask=0,
            invalid_keypoint=0,
            unmapped_class=unmapped,
            unsupported_image=0,
            warnings=warnings,
        )
        ann_profile["input_format"] = self.format_key

        structure_profile = {
            "input_format": self.format_key,
            "image_dir": str(image_root.relative_to(root)) if image_root != root else "",
            "label_dir": str(label_root.relative_to(root)) if label_root != root else "",
            "class_config": class_config or "",
            "category_count": len(class_mapping),
        }
        parsing_summary = {
            "input_format": self.format_key,
            "source_format": "yolo",
            "conversion_strategy": "yolo_normalized_xywh_to_internal_xyxy_absolute",
            "label_count": len(label_index),
            "missing_images": missing_image,
            "invalid_bbox_count": invalid_bbox,
            "unmapped_class_count": unmapped,
        }

        dataset = InternalImageDataset(
            modality="image",
            input_format=self.format_key,
            original_format="yolo",
            samples=samples,
            class_mapping=class_mapping,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            annotation_profile=ann_profile,
            warnings=warnings,
            dataset_root=root,
            raw_root=root,
        )
        return AdapterResult(ok=True, data={"internal_dataset": dataset})


def _load_class_names(root: Path, class_config: str, warnings: List[str]) -> List[str]:
    candidate: Optional[Path] = None
    if class_config:
        cand = (root / class_config).resolve()
        if cand.exists():
            candidate = cand
        else:
            warnings.append(f"class_config file not found: {class_config}")
    if candidate is None:
        for p in root.rglob("data.yaml"):
            candidate = p
            break
    if candidate is None:
        for p in root.rglob("data.yml"):
            candidate = p
            break
    if candidate is None:
        for p in root.rglob("classes.txt"):
            candidate = p
            break
    if candidate is None:
        for p in root.rglob("names.txt"):
            candidate = p
            break
    if candidate is None:
        return []
    if candidate.suffix.lower() in {".yaml", ".yml"}:
        return _parse_yaml_names(candidate, warnings)
    if candidate.suffix.lower() == ".txt":
        try:
            lines = [line.strip() for line in candidate.read_text(encoding="utf-8", errors="ignore").splitlines()]
        except Exception:
            return []
        return [name for name in lines if name]
    return []


def _parse_yaml_names(path: Path, warnings: List[str]) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            names = data.get("names")
            if isinstance(names, list):
                return [str(n) for n in names]
            if isinstance(names, dict):
                items = sorted(names.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0)
                return [str(v) for _, v in items]
        return []
    except Exception:
        names: List[str] = []
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names:"):
                rest = stripped.split(":", 1)[1].strip()
                if rest.startswith("[") and rest.endswith("]"):
                    inner = rest[1:-1]
                    return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
                in_names = True
                continue
            if in_names:
                if stripped.startswith("-"):
                    names.append(stripped.lstrip("-").strip().strip("'\""))
                elif stripped and not stripped[0].isspace() and ":" in stripped:
                    in_names = False
        return names
