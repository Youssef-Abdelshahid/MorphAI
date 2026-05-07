from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.image.internal import (
    InternalImageDataset,
    ImageSample,
    SUPPORTED_IMAGE_EXTS,
    annotation_profile_from_samples,
    clamp_box,
    extract_zip,
    find_dataset_root,
    safe_image_size,
)


class PascalVocImageAdapter(BaseFormatAdapter):
    modality = "Image"
    input_format = "Pascal VOC XML annotations"
    is_implemented = True
    format_key = "pascal_voc"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("Pascal VOC ingestion requires a .zip archive containing images and XML annotations.")
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
        if not any(n.lower().endswith(".xml") for n in names):
            errors.append("Archive does not contain any Pascal VOC XML annotation files.")
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
        annotation_dir: str = "",
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

        image_root = root
        annotation_root = root
        if image_dir:
            candidate = (root / image_dir).resolve()
            if candidate.exists() and candidate.is_dir():
                image_root = candidate
            else:
                warnings.append(f"Provided image_dir not found: {image_dir}")
        if annotation_dir:
            candidate = (root / annotation_dir).resolve()
            if candidate.exists() and candidate.is_dir():
                annotation_root = candidate
            else:
                warnings.append(f"Provided annotation_dir not found: {annotation_dir}")

        image_index: Dict[str, Path] = {}
        for p in image_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_index[p.stem.lower()] = p
                image_index[p.name.lower()] = p

        xml_files = sorted([p for p in annotation_root.rglob("*.xml") if p.is_file()])
        if not xml_files:
            return AdapterResult(
                ok=False,
                message="No Pascal VOC XML annotation files found in archive.",
                errors=["No XML annotation files found."],
            )

        samples: List[ImageSample] = []
        invalid_bbox = 0
        missing_image = 0
        invalid_xml = 0
        all_classes: Counter = Counter()

        for xml_path in xml_files:
            try:
                tree = ET.parse(str(xml_path))
            except ET.ParseError as exc:
                warnings.append(f"Invalid VOC XML '{xml_path.name}': {exc}")
                invalid_xml += 1
                continue
            root_elem = tree.getroot()
            filename_node = root_elem.findtext("filename") or ""
            stem = Path(filename_node).stem.lower() if filename_node else xml_path.stem.lower()
            img_path = image_index.get(stem) or image_index.get(filename_node.lower()) or image_index.get(xml_path.stem.lower())
            if img_path is None:
                missing_image += 1
                continue
            size = root_elem.find("size")
            try:
                width = int(size.findtext("width") or 0) if size is not None else 0
                height = int(size.findtext("height") or 0) if size is not None else 0
            except (TypeError, ValueError):
                width = 0
                height = 0
            if width == 0 or height == 0:
                width, height = safe_image_size(img_path)

            sample_bboxes: List = []
            sample_classes: List[str] = []
            for obj in root_elem.findall("object"):
                cls = (obj.findtext("name") or "").strip()
                if not cls:
                    continue
                box = obj.find("bndbox")
                if box is None:
                    continue
                try:
                    x1 = float(box.findtext("xmin") or 0)
                    y1 = float(box.findtext("ymin") or 0)
                    x2 = float(box.findtext("xmax") or 0)
                    y2 = float(box.findtext("ymax") or 0)
                except (TypeError, ValueError):
                    invalid_bbox += 1
                    continue
                if x2 <= x1 or y2 <= y1:
                    invalid_bbox += 1
                    continue
                clamped = clamp_box((x1, y1, x2, y2), width, height)
                sample_bboxes.append(clamped)
                sample_classes.append(cls)
                all_classes[cls] += 1

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
        unused = sum(1 for stem in image_index if stem not in used_stems and "." not in stem)

        class_mapping = {idx: name for idx, name in enumerate(sorted(all_classes.keys()))}
        ann_profile = annotation_profile_from_samples(
            samples,
            class_mapping,
            parsed_image_count=len(samples),
            original_image_count=len(image_index) // 2 if image_index else 0,
            corrupted=0,
            missing=missing_image,
            unused=max(unused, 0),
            invalid_bbox=invalid_bbox,
            invalid_mask=0,
            invalid_keypoint=0,
            unmapped_class=0,
            unsupported_image=0,
            warnings=warnings,
        )
        ann_profile["input_format"] = self.format_key

        structure_profile = {
            "input_format": self.format_key,
            "image_dir": str(image_root.relative_to(root)) if image_root != root else "",
            "annotation_dir": str(annotation_root.relative_to(root)) if annotation_root != root else "",
            "annotation_count": len(xml_files),
            "category_count": len(class_mapping),
        }

        parsing_summary = {
            "input_format": self.format_key,
            "source_format": "pascal_voc",
            "conversion_strategy": "voc_xyxy_absolute_passthrough",
            "annotation_count": len(xml_files),
            "missing_images": missing_image,
            "invalid_bbox_count": invalid_bbox,
            "invalid_xml_count": invalid_xml,
        }
        if invalid_xml:
            warnings.append(f"{invalid_xml} XML annotation files failed to parse and were skipped.")

        dataset = InternalImageDataset(
            modality="image",
            input_format=self.format_key,
            original_format="pascal_voc",
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
