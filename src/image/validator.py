import zipfile
from pathlib import Path
from typing import List, Tuple

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_SUPPORTED_TASKS = {"classification", "multiclass", "binary"}


def _scan_image_folder(root: Path) -> Tuple[List[str], dict, int]:
    classes = []
    class_counts = {}
    total = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        imgs = [f for f in sub.iterdir()
                if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS]
        if imgs:
            classes.append(sub.name)
            class_counts[sub.name] = len(imgs)
            total += len(imgs)
    return classes, class_counts, total


def validate_image_zip(zip_path: Path) -> list:
    errors = []

    if not zip_path.exists():
        errors.append(f"File does not exist: {zip_path}")
        return errors

    if zip_path.suffix.lower() != ".zip":
        errors.append(
            f"Image modality requires a .zip file. "
            f"Got: '{zip_path.suffix or zip_path.name}'. "
            "Provide the dataset as a single .zip archive containing class sub-folders."
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
        and not any(
            part.startswith(".") or part == "__MACOSX"
            for part in Path(n).parts
        )
    ]

    if not image_names:
        errors.append(
            "Zip archive contains no valid image files. "
            f"Supported formats: {', '.join(sorted(_IMAGE_EXTENSIONS))}."
        )
        return errors

    classes_found = set()
    for n in image_names:
        parts = [
            p for p in Path(n).parts
            if not p.startswith(".") and p != "__MACOSX"
        ]
        if len(parts) >= 2:
            classes_found.add(parts[-2])

    if len(classes_found) < 2:
        errors.append(
            f"Expected at least 2 class sub-folders within the zip, "
            f"found {len(classes_found)}. "
            "Structure should be: <class_name>/<image_files> "
            "(or <root_folder>/<class_name>/<image_files>)."
        )

    return errors


def validate_image_run(config, root: Path) -> list:
    errors = []

    if not root.exists():
        errors.append(f"Path does not exist: {root}")
        return errors

    if not root.is_dir():
        errors.append(f"Path is not a directory: {root}")
        return errors

    classes, class_counts, total = _scan_image_folder(root)

    if len(classes) < 2:
        errors.append(
            f"Need at least 2 class sub-folders with images, found {len(classes)}. "
            "Expected structure: root/<class_name>/<images>."
        )
        return errors

    if total < 10:
        errors.append(
            f"Dataset has only {total} images. At least 10 are required."
        )

    small_classes = [c for c, n in class_counts.items() if n < 2]
    if small_classes:
        errors.append(
            f"Classes with fewer than 2 images: {small_classes}. "
            "Each class needs at least 2 samples for evaluation."
        )

    if config.task_type not in _SUPPORTED_TASKS:
        errors.append(
            f"Task type '{config.task_type}' is not yet supported for image data. "
            f"Supported: {sorted(_SUPPORTED_TASKS)}."
        )

    return errors
