import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .io_utils import ANNOTATION_EXTENSIONS, AUDIO_EXTENSIONS, as_mono, read_audio, safe_duration

_PROFILE_SAMPLE_LIMIT = 500


@dataclass
class AudioProfile:
    root_path: Path
    n_audio_files: int
    n_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    imbalance_ratio: float
    min_class_size: int
    total_duration_sec: float
    avg_duration_sec: float
    min_duration_sec: float
    max_duration_sec: float
    duration_std_sec: float
    duration_distribution: Dict[str, int]
    sample_rate_distribution: Dict[str, int]
    channel_count_distribution: Dict[str, int]
    bit_depth_distribution: Dict[str, int]
    file_format_distribution: Dict[str, int]
    n_corrupt: int
    corrupt_paths: List[str]
    n_silent: int
    n_clipped: int
    avg_rms: float
    rms_std: float
    avg_loudness_db: float
    noise_proxy: float
    silence_ratio: float
    clipping_ratio: float
    corruption_ratio: float
    estimated_noise_ratio: float
    has_labels: bool
    label_distribution: Dict[str, int]
    missing_invalid_labels: int
    transcript_count: int
    speaker_label_count: int
    annotation_counts: Dict[str, int]
    audio_paths: List[str]
    audio_labels: List[str]
    input_format: str = ""
    parsing_summary: Dict[str, int] = None
    metadata_profile: Dict[str, int] = None
    structure_profile: Dict[str, int] = None
    annotation_or_reference_profile: Dict[str, int] = None
    parser_warnings: List[str] = None
    has_class_labels_flag: bool = False
    has_speaker_labels_flag: bool = False
    has_speaker_pairs_flag: bool = False
    has_transcripts_flag: bool = False
    has_temporal_segments_flag: bool = False
    has_event_labels_flag: bool = False
    has_anomaly_labels_flag: bool = False
    has_noisy_clean_pairs_flag: bool = False


def _audio_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)
    )


def _label_for(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return ""


def _duration_bucket(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 5:
        return "1-5s"
    if seconds < 15:
        return "5-15s"
    if seconds < 60:
        return "15-60s"
    return "60s+"


def _annotation_counts(root: Path) -> Dict[str, int]:
    counts = {"csv": 0, "json": 0, "txt": 0, "rttm": 0, "lab": 0, "textgrid": 0, "transcripts": 0, "speakers": 0, "segments": 0, "events": 0, "noise_pairs": 0}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower().lstrip(".")
        lower = path.name.lower()
        if path.suffix in ANNOTATION_EXTENSIONS or path.suffix.lower() in {".csv", ".json", ".txt", ".rttm", ".lab", ".textgrid"}:
            counts[suffix] = counts.get(suffix, 0) + 1
        if any(token in lower for token in ["transcript", "transcription", "asr"]):
            counts["transcripts"] += 1
        if any(token in lower for token in ["speaker", "pair"]):
            counts["speakers"] += 1
        if any(token in lower for token in ["segment", "diar", "rttm", "vad"]):
            counts["segments"] += 1
        if any(token in lower for token in ["event", "sed"]):
            counts["events"] += 1
        if "clean" in lower or "noisy" in lower:
            counts["noise_pairs"] += 1
    return counts


def _read_manifest_labels(root: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for name in ["labels.csv", "metadata.csv", "manifest.csv", "speakers.csv", "speaker_labels.csv"]:
        path = root / name
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    file_value = row.get("file") or row.get("filename") or row.get("path") or row.get("audio")
                    label_value = row.get("label") or row.get("class") or row.get("target") or row.get("speaker")
                    if file_value and label_value:
                        labels[Path(file_value).name] = str(label_value).strip()
        except Exception:
            pass
    return labels


def _count_transcripts(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower in {"transcripts.csv", "transcriptions.csv"}:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    total += sum(1 for _ in csv.DictReader(handle))
            except Exception:
                pass
        elif path.suffix.lower() in {".txt", ".json"} and any(token in lower for token in ["transcript", "transcription", "asr"]):
            total += 1
    return total


def _count_speaker_labels(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if lower in {"speakers.csv", "pairs.csv", "speaker_labels.csv"}:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    total += sum(1 for _ in csv.DictReader(handle))
            except Exception:
                pass
        elif "speaker" in lower and path.suffix.lower() in {".txt", ".json", ".rttm"}:
            total += 1
    return total


def profile_audio_dataset(root: Path) -> AudioProfile:
    paths = _audio_files(root)
    manifest_labels = _read_manifest_labels(root)
    labels = [manifest_labels.get(p.name, _label_for(p, root)) for p in paths]
    label_counts = Counter(label for label in labels if label)
    class_counts = dict(sorted(label_counts.items()))
    class_names = sorted(class_counts)
    n_classes = len(class_names)
    counts_sorted = sorted(class_counts.values(), reverse=True)
    min_class_size = counts_sorted[-1] if counts_sorted else 0
    imbalance_ratio = counts_sorted[0] / counts_sorted[-1] if counts_sorted and counts_sorted[-1] > 0 else float("inf")

    sample = paths
    if len(paths) > _PROFILE_SAMPLE_LIMIT:
        rng = np.random.RandomState(42)
        sample = [paths[i] for i in sorted(rng.choice(len(paths), _PROFILE_SAMPLE_LIMIT, replace=False).tolist())]

    durations: List[float] = []
    rates: List[int] = []
    channels: List[int] = []
    bit_depths: List[Optional[int]] = []
    rms_values: List[float] = []
    loudness_values: List[float] = []
    noise_values: List[float] = []
    n_corrupt = 0
    n_silent = 0
    n_clipped = 0
    corrupt_paths: List[str] = []

    for path in sample:
        try:
            sr, data, bit_depth = read_audio(path)
            mono = as_mono(data)
            duration = safe_duration(sr, mono)
            rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
            clipped = float(np.mean(np.abs(mono) >= 0.98)) if len(mono) else 0.0
            diff_rms = float(np.sqrt(np.mean(np.square(np.diff(mono))))) if len(mono) > 1 else 0.0
            durations.append(duration)
            rates.append(sr)
            channels.append(1 if np.asarray(data).ndim == 1 else int(np.asarray(data).shape[1]))
            bit_depths.append(bit_depth)
            rms_values.append(rms)
            loudness_values.append(float(20.0 * np.log10(max(rms, 1e-8))))
            noise_values.append(diff_rms / max(rms, 1e-8))
            if rms < 1e-4:
                n_silent += 1
            if clipped > 0.001:
                n_clipped += 1
        except Exception:
            n_corrupt += 1
            corrupt_paths.append(str(path))

    dur_arr = np.asarray(durations, dtype=float)
    rms_arr = np.asarray(rms_values, dtype=float)
    duration_distribution = dict(Counter(_duration_bucket(v) for v in durations))
    sample_rate_distribution = {str(k): v for k, v in Counter(rates).items()}
    channel_count_distribution = {str(k): v for k, v in Counter(channels).items()}
    bit_depth_distribution = {str(k if k is not None else "unknown"): v for k, v in Counter(bit_depths).items()}
    file_format_distribution = {k: v for k, v in Counter(p.suffix.lower().lstrip(".") for p in paths).items()}
    n_files = len(paths)
    n_valid = len(durations)
    annotation_counts = _annotation_counts(root)

    return AudioProfile(
        root_path=root,
        n_audio_files=n_files,
        n_classes=n_classes,
        class_names=class_names,
        class_counts=class_counts,
        imbalance_ratio=imbalance_ratio,
        min_class_size=min_class_size,
        total_duration_sec=float(dur_arr.sum()) if n_valid else 0.0,
        avg_duration_sec=float(dur_arr.mean()) if n_valid else 0.0,
        min_duration_sec=float(dur_arr.min()) if n_valid else 0.0,
        max_duration_sec=float(dur_arr.max()) if n_valid else 0.0,
        duration_std_sec=float(dur_arr.std()) if n_valid else 0.0,
        duration_distribution=duration_distribution,
        sample_rate_distribution=sample_rate_distribution,
        channel_count_distribution=channel_count_distribution,
        bit_depth_distribution=bit_depth_distribution,
        file_format_distribution=file_format_distribution,
        n_corrupt=n_corrupt,
        corrupt_paths=corrupt_paths[:10],
        n_silent=n_silent,
        n_clipped=n_clipped,
        avg_rms=float(rms_arr.mean()) if len(rms_arr) else 0.0,
        rms_std=float(rms_arr.std()) if len(rms_arr) else 0.0,
        avg_loudness_db=float(np.mean(loudness_values)) if loudness_values else -120.0,
        noise_proxy=float(np.mean(noise_values)) if noise_values else 0.0,
        silence_ratio=n_silent / max(n_files, 1),
        clipping_ratio=n_clipped / max(n_files, 1),
        corruption_ratio=n_corrupt / max(n_files, 1),
        estimated_noise_ratio=float(np.clip(np.mean(noise_values) if noise_values else 0.0, 0.0, 1.0)),
        has_labels=bool(class_counts),
        label_distribution=class_counts,
        missing_invalid_labels=sum(1 for label in labels if not label),
        transcript_count=_count_transcripts(root),
        speaker_label_count=_count_speaker_labels(root),
        annotation_counts=annotation_counts,
        audio_paths=[str(p) for p in paths],
        audio_labels=labels,
    )
