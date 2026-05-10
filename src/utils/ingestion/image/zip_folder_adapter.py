from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.image.internal import (
    InternalImageDataset,
    ImageSample,
    SUPPORTED_IMAGE_EXTS,
    annotation_profile_from_samples,
    collect_images,
    extract_zip,
    find_dataset_root,
    safe_image_size,
)


class ZipFolderImageAdapter(BaseFormatAdapter):
    modality = "Image"
    input_format = "Image folder / ZIP"
    is_implemented = True
    format_key = "zip_folder"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("Image folder / ZIP requires a .zip archive.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not zipfile.is_zipfile(str(path)):
            errors.append(f"'{path.name}' is not a valid zip archive.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                bad = zf.testzip()
                if bad:
                    errors.append(f"Corrupted entry detected in zip archive: '{bad}'.")
                names = zf.namelist()
        except Exception as exc:
            errors.append(f"Cannot open zip archive: {exc}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not any(Path(n).suffix.lower() in SUPPORTED_IMAGE_EXTS for n in names):
            errors.append(
                "Zip archive contains no valid image files. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}."
            )
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(self, path: Path, work_dir: Path = None, **kwargs) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        if work_dir is None:
            return AdapterResult(ok=False, message="work_dir is required for image ingestion.", errors=["work_dir is required."])
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        warnings = extract_zip(path, work_dir)
        root = find_dataset_root(work_dir)

        samples: List[ImageSample] = []
        corrupted = 0
        unsupported = 0
        all_files = list(root.rglob("*"))
        original_files = [p for p in all_files if p.is_file() and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)]
        for p in original_files:
            if p.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
                unsupported += 1

        image_paths = collect_images(root)
        class_dirs = sorted({p.parent.name for p in image_paths if p.parent != root})
        transcription_map = {}
        for txt_file in root.rglob("*.txt"):
            if not txt_file.is_file():
                continue
            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                continue
            if not content:
                continue
            stem = txt_file.stem
            for suffix in ("_text", "_ocr", "_transcript"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            transcription_map[(txt_file.parent, stem)] = content
        _ANN_EXTS = {".json", ".xml", ".txt", ".csv"}
        _MASK_SUFFIXES = ("_mask", "_seg", "_segmentation")
        _DEPTH_SUFFIXES = ("_depth", "_depthmap", "_depth_map")
        _KP_SUFFIXES = ("_keypoints", "_kp", "_pose")
        for img_path in image_paths:
            w, h = safe_image_size(img_path)
            if w == 0 and h == 0:
                corrupted += 1
                continue
            try:
                rel = img_path.relative_to(root)
                if len(rel.parts) > 1 and rel.parts[0] != "":
                    label = rel.parts[0]
                else:
                    label = ""
            except ValueError:
                label = ""
            transcription = transcription_map.get((img_path.parent, img_path.stem))
            stem = img_path.stem
            parent = img_path.parent
            bboxes_marker = []
            masks_marker = []
            keypoints_marker = []
            depth_marker = None
            for sib in parent.iterdir():
                if not sib.is_file() or sib == img_path:
                    continue
                sib_stem = sib.stem
                sib_ext = sib.suffix.lower()
                if any(sib_stem.startswith(stem + s) for s in _KP_SUFFIXES) and sib_ext in _ANN_EXTS:
                    keypoints_marker = [[(0.0, 0.0, 0.0)]]
                elif any(sib_stem.startswith(stem + s) for s in _MASK_SUFFIXES) and sib_ext in {".png", ".jpg", ".jpeg", ".bmp", ".npy"}:
                    masks_marker.append(str(sib))
                elif any(sib_stem.startswith(stem + s) for s in _DEPTH_SUFFIXES) and sib_ext in {".png", ".jpg", ".jpeg", ".bmp", ".npy"}:
                    depth_marker = str(sib)
                elif sib_stem == stem and sib_ext in _ANN_EXTS:
                    bboxes_marker = [(0.0, 0.0, 0.0, 0.0)]
            sample = ImageSample(
                image_path=str(img_path),
                image_id=str(img_path.relative_to(root)),
                width=w,
                height=h,
                labels=[label] if label else [],
                transcription=transcription,
                split="",
                bboxes=bboxes_marker,
                masks=masks_marker,
                keypoints=keypoints_marker,
                depth_path=depth_marker,
            )
            samples.append(sample)

        class_mapping = {idx: name for idx, name in enumerate(class_dirs)}
        parsing_summary = {
            "input_format": self.format_key,
            "source_format": "image_folder_zip",
            "conversion_strategy": "passthrough",
            "discovered_images": len(image_paths),
            "extracted_root": str(root),
        }
        ann_profile = annotation_profile_from_samples(
            samples,
            class_mapping,
            parsed_image_count=len(samples),
            original_image_count=len(image_paths),
            corrupted=corrupted,
            missing=0,
            unused=0,
            invalid_bbox=0,
            invalid_mask=0,
            invalid_keypoint=0,
            unmapped_class=0,
            unsupported_image=unsupported,
            warnings=warnings,
        )
        ann_profile["input_format"] = self.format_key
        structure_profile = {
            "input_format": self.format_key,
            "structure_type": "class_folder" if class_dirs else "flat",
            "class_dirs": class_dirs,
        }

        dataset = InternalImageDataset(
            modality="image",
            input_format=self.format_key,
            original_format="image_folder_zip",
            samples=samples,
            class_mapping=class_mapping,
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
