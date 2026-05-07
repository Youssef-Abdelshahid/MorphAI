from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.image.internal import (
    InternalImageDataset,
    ImageSample,
    SUPPORTED_IMAGE_EXTS,
    annotation_profile_from_samples,
    clamp_box,
    extract_zip,
    find_dataset_root,
    normalize_xywh_to_xyxy,
)


class CocoImageAdapter(BaseFormatAdapter):
    modality = "Image"
    input_format = "COCO JSON annotations"
    is_implemented = True
    format_key = "coco"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("COCO ingestion requires a .zip archive containing images and a COCO JSON file.")
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
        has_json = any(n.lower().endswith(".json") for n in names)
        has_image = any(Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTS for n in names)
        if not has_json:
            errors.append("Archive does not contain any COCO annotation JSON files.")
        if not has_image:
            errors.append("Archive does not contain any image files.")
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(
        self,
        path: Path,
        work_dir: Path = None,
        annotation_path: str = "",
        image_dir: str = "",
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

        json_files = sorted([p for p in root.rglob("*.json") if p.is_file()])
        annotation_file: Optional[Path] = None
        if annotation_path:
            candidate = (root / annotation_path).resolve()
            if candidate.exists() and candidate.is_file():
                annotation_file = candidate
            else:
                return AdapterResult(
                    ok=False,
                    message=f"Provided annotation path does not exist: {annotation_path}",
                    errors=[f"Annotation file not found: {annotation_path}"],
                )
        if annotation_file is None:
            valid_jsons = [p for p in json_files if _looks_like_coco(p)]
            if len(valid_jsons) == 0:
                return AdapterResult(
                    ok=False,
                    message="No COCO-format JSON files found in archive.",
                    errors=["No file with COCO 'images', 'annotations', and 'categories' keys was found."],
                )
            if len(valid_jsons) > 1:
                names = ", ".join(str(p.relative_to(root)) for p in valid_jsons)
                return AdapterResult(
                    ok=False,
                    message=f"Multiple COCO JSON files found: {names}. Specify annotation_path.",
                    errors=[f"Ambiguous COCO annotation files: {names}"],
                )
            annotation_file = valid_jsons[0]

        try:
            data = json.loads(annotation_file.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            return AdapterResult(
                ok=False,
                message=f"Invalid COCO JSON syntax: {exc}",
                errors=[f"Invalid COCO JSON syntax in {annotation_file.name}: {exc}"],
            )
        for required in ("images", "annotations", "categories"):
            if required not in data:
                return AdapterResult(
                    ok=False,
                    message=f"COCO JSON is missing required key: {required}",
                    errors=[f"COCO JSON is missing required '{required}' key."],
                )

        categories = {int(c["id"]): str(c.get("name") or f"cat_{c['id']}") for c in data["categories"] if "id" in c}
        images_meta: Dict[int, Dict[str, Any]] = {}
        for img in data["images"]:
            try:
                images_meta[int(img["id"])] = {
                    "file_name": str(img.get("file_name", "")),
                    "width": int(img.get("width", 0) or 0),
                    "height": int(img.get("height", 0) or 0),
                }
            except (KeyError, ValueError, TypeError):
                continue

        image_dir_root = root
        if image_dir:
            candidate = (root / image_dir).resolve()
            if candidate.exists() and candidate.is_dir():
                image_dir_root = candidate
            else:
                warnings.append(f"Provided image_dir not found, falling back to archive root: {image_dir}")

        image_path_index: Dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                image_path_index[p.name.lower()] = p
                try:
                    rel = str(p.relative_to(root)).lower()
                    image_path_index[rel] = p
                except ValueError:
                    pass

        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data["annotations"]:
            try:
                img_id = int(ann["image_id"])
            except (KeyError, ValueError, TypeError):
                continue
            annotations_by_image.setdefault(img_id, []).append(ann)

        samples: List[ImageSample] = []
        invalid_bbox = 0
        invalid_mask = 0
        invalid_kpt = 0
        unmapped = 0
        missing = 0
        used_image_ids = set()

        for img_id, meta in images_meta.items():
            file_name = meta["file_name"]
            key = Path(file_name).name.lower()
            img_path = image_path_index.get(key) or image_path_index.get(file_name.lower())
            if img_path is None:
                missing += 1
                continue
            used_image_ids.add(img_id)
            width = meta["width"] or 0
            height = meta["height"] or 0
            if width == 0 or height == 0:
                from src.utils.ingestion.image.internal import safe_image_size

                width, height = safe_image_size(img_path)

            sample_bboxes: List = []
            sample_classes: List[str] = []
            sample_masks: List = []
            sample_keypoints: List = []

            for ann in annotations_by_image.get(img_id, []):
                cat_id = ann.get("category_id")
                if cat_id is None:
                    unmapped += 1
                    continue
                try:
                    cat_id = int(cat_id)
                except (TypeError, ValueError):
                    unmapped += 1
                    continue
                cat_name = categories.get(cat_id)
                if cat_name is None:
                    unmapped += 1
                    continue

                bbox = ann.get("bbox")
                if bbox and len(bbox) == 4:
                    try:
                        xyxy = normalize_xywh_to_xyxy(tuple(float(v) for v in bbox))
                        clamped = clamp_box(xyxy, width, height)
                        if (clamped[2] - clamped[0]) <= 0 or (clamped[3] - clamped[1]) <= 0:
                            invalid_bbox += 1
                        else:
                            sample_bboxes.append(clamped)
                            sample_classes.append(cat_name)
                    except (TypeError, ValueError):
                        invalid_bbox += 1

                seg = ann.get("segmentation")
                if seg:
                    if isinstance(seg, list) and seg and all(isinstance(s, list) for s in seg):
                        sample_masks.append(seg)
                    elif isinstance(seg, dict):
                        sample_masks.append(seg)
                    else:
                        invalid_mask += 1

                kpts = ann.get("keypoints")
                if kpts and isinstance(kpts, list) and len(kpts) % 3 == 0:
                    triples = []
                    for i in range(0, len(kpts), 3):
                        try:
                            triples.append((float(kpts[i]), float(kpts[i + 1]), float(kpts[i + 2])))
                        except (TypeError, ValueError):
                            invalid_kpt += 1
                            triples = []
                            break
                    if triples:
                        sample_keypoints.append(triples)

            labels: List[str] = []
            if sample_classes:
                from collections import Counter

                most_common = Counter(sample_classes).most_common(1)[0][0]
                labels = [most_common]

            samples.append(
                ImageSample(
                    image_path=str(img_path),
                    image_id=str(img_id),
                    width=width,
                    height=height,
                    labels=labels,
                    bboxes=sample_bboxes,
                    bbox_classes=sample_classes,
                    masks=sample_masks,
                    keypoints=sample_keypoints,
                    split=split,
                )
            )

        unique_image_paths = {p for p in image_path_index.values()}
        unused = max(0, len(unique_image_paths) - len(used_image_ids))

        ann_profile = annotation_profile_from_samples(
            samples,
            categories,
            parsed_image_count=len(samples),
            original_image_count=len(images_meta),
            corrupted=0,
            missing=missing,
            unused=unused,
            invalid_bbox=invalid_bbox,
            invalid_mask=invalid_mask,
            invalid_keypoint=invalid_kpt,
            unmapped_class=unmapped,
            unsupported_image=0,
            warnings=warnings,
        )
        ann_profile["input_format"] = self.format_key

        structure_profile = {
            "input_format": self.format_key,
            "annotation_file": str(annotation_file.relative_to(root)),
            "image_dir": str(image_dir_root.relative_to(root)) if image_dir_root != root else "",
            "category_count": len(categories),
        }

        parsing_summary = {
            "input_format": self.format_key,
            "source_format": "coco_json",
            "conversion_strategy": "coco_xywh_to_internal_xyxy_absolute",
            "annotation_file": str(annotation_file.name),
            "discovered_images": len(samples),
            "missing_images": missing,
            "unused_images": unused,
            "invalid_bbox_count": invalid_bbox,
            "invalid_mask_count": invalid_mask,
            "invalid_keypoint_count": invalid_kpt,
            "unmapped_class_count": unmapped,
        }

        dataset = InternalImageDataset(
            modality="image",
            input_format=self.format_key,
            original_format="coco",
            samples=samples,
            class_mapping=categories,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            annotation_profile=ann_profile,
            warnings=warnings,
            dataset_root=root,
            raw_root=root,
        )
        return AdapterResult(
            ok=True,
            data={"internal_dataset": dataset},
        )


def _looks_like_coco(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return False
    return isinstance(data, dict) and "images" in data and "annotations" in data and "categories" in data
