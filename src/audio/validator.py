import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from .config import SUPPORTED_TASK_TYPES, VALID_TASK_TYPES, default_metric_for_task, normalize_task_type, valid_metrics_for_task
from .io_utils import AUDIO_EXTENSIONS


def validate_audio_zip(zip_path: Path) -> list:
    errors = []
    if not zip_path.exists():
        errors.append(f"File does not exist: {zip_path}")
        return errors
    if zip_path.suffix.lower() != ".zip":
        errors.append("Audio modality requires a .zip file. Provide the dataset as a single ZIP archive.")
        return errors
    try:
        if not zipfile.is_zipfile(str(zip_path)):
            errors.append(f"'{zip_path.name}' is not a valid zip archive.")
            return errors
    except Exception as exc:
        errors.append(f"Cannot read zip archive: {exc}")
        return errors
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            bad = zf.testzip()
            if bad:
                errors.append(f"Corrupted entry detected in zip archive: '{bad}'.")
                return errors
            names = zf.namelist()
    except zipfile.BadZipFile:
        errors.append("The uploaded file is not a readable ZIP archive.")
        return errors
    audio_names = [
        n for n in names
        if Path(n).suffix.lower() in AUDIO_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX" for part in Path(n).parts)
    ]
    unsupported = [
        n for n in names
        if Path(n).suffix and Path(n).suffix.lower() not in AUDIO_EXTENSIONS
        and Path(n).suffix.lower() not in {".csv", ".json", ".txt", ".rttm", ".lab", ".textgrid"}
        and not any(part.startswith(".") or part == "__MACOSX" for part in Path(n).parts)
    ]
    if not audio_names:
        errors.append(f"Zip archive contains no supported audio files. Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}.")
    non_wav_audio = [n for n in audio_names if Path(n).suffix.lower() != ".wav"]
    if non_wav_audio:
        try:
            import soundfile
        except ImportError:
            sample = ", ".join(Path(n).name for n in non_wav_audio[:5])
            errors.append(f"The ZIP contains compressed/lossless audio files ({sample}), but this environment can only decode WAV unless optional soundfile support is installed.")
    if unsupported:
        sample = ", ".join(Path(n).name for n in unsupported[:5])
        errors.append(f"Unsupported files found in the audio ZIP: {sample}. Supported audio formats are wav, mp3, flac, and ogg.")
    return errors


def _scan_audio_folder(root: Path) -> Tuple[List[str], Dict[str, int], int]:
    classes = []
    class_counts = {}
    total = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith(".") or sub.name == "__MACOSX":
            continue
        files = [f for f in sub.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS]
        if files:
            classes.append(sub.name)
            class_counts[sub.name] = len(files)
            total += len(files)
    if total == 0:
        total = len([f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS])
    return classes, class_counts, total


def _annotation_summary(root: Path) -> Dict[str, int]:
    counts = {"transcripts": 0, "speakers": 0, "segments": 0, "events": 0, "noise_pairs": 0, "annotations": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if path.suffix.lower() in {".csv", ".json", ".txt", ".rttm", ".lab", ".textgrid"}:
            counts["annotations"] += 1
        if any(t in lower for t in ["transcript", "transcription", "asr"]):
            counts["transcripts"] += 1
        if any(t in lower for t in ["speaker", "pair"]):
            counts["speakers"] += 1
        if any(t in lower for t in ["segment", "diar", "rttm", "vad"]):
            counts["segments"] += 1
        if any(t in lower for t in ["event", "sed"]):
            counts["events"] += 1
        if "clean" in lower or "noisy" in lower:
            counts["noise_pairs"] += 1
    return counts


def validate_audio_run(config, root: Path) -> list:
    errors = []
    task_type = normalize_task_type(config.task_type)
    metric = (config.metric or "").strip().lower()
    if not root.exists() or not root.is_dir():
        return [f"Path is not a valid directory: {root}"]
    if not task_type:
        errors.append("An audio task type is required.")
    elif task_type not in VALID_TASK_TYPES:
        errors.append(f"Task type '{config.task_type}' is not valid for audio data. Supported task types: {sorted(SUPPORTED_TASK_TYPES)}")
    elif task_type not in SUPPORTED_TASK_TYPES:
        errors.append(f"Task type '{task_type}' is not yet supported for audio data. Supported task types: {sorted(SUPPORTED_TASK_TYPES)}")
    valid_metrics = valid_metrics_for_task(task_type)
    if valid_metrics and metric and metric not in valid_metrics:
        errors.append(f"Metric '{config.metric}' is not valid for '{task_type}'. Valid metrics: {valid_metrics}")
    if valid_metrics and not metric:
        config.metric = default_metric_for_task(task_type)
    classes, class_counts, total = _scan_audio_folder(root)
    if total <= 0:
        errors.append("The extracted dataset does not contain any supported audio files.")
        return errors
    annotations = _annotation_summary(root)
    if task_type == "classification":
        if len(classes) < 2:
            errors.append("Audio classification needs at least 2 labeled class folders.")
        elif any(n < 2 for n in class_counts.values()):
            errors.append("Each audio class needs at least 2 files for evaluation.")
    elif task_type == "asr" and annotations["transcripts"] <= 0:
        errors.append("Speech recognition requires transcript files or a transcript manifest.")
    elif task_type == "speaker_recognition" and len(classes) < 2 and annotations["speakers"] <= 0:
        errors.append("Speaker recognition requires speaker folders, speaker labels, or verification pairs.")
    elif task_type == "sound_event_detection" and len(classes) < 2 and annotations["events"] <= 0:
        errors.append("Sound event detection requires event labels or temporal event annotations.")
    return errors
