from __future__ import annotations

import csv
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
SUPPORTED_METADATA_EXTS = {".csv", ".json", ".jsonl", ".ndjson"}


@dataclass
class AudioSample:
    sample_id: str
    audio_path: str
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    label: str = ""
    speaker_id: str = ""
    transcript: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    event_label: str = ""
    anomaly_label: str = ""
    noisy_path: str = ""
    clean_path: str = ""
    audio_path_a: str = ""
    audio_path_b: str = ""
    same_speaker: Optional[int] = None
    split: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InternalAudioDataset:
    modality: str = "audio"
    input_format: str = ""
    original_format: str = ""
    samples: List[AudioSample] = field(default_factory=list)
    label_mapping: Dict[int, str] = field(default_factory=dict)
    speaker_mapping: Dict[int, str] = field(default_factory=dict)
    structure_profile: Dict[str, Any] = field(default_factory=dict)
    parsing_summary: Dict[str, Any] = field(default_factory=dict)
    metadata_profile: Dict[str, Any] = field(default_factory=dict)
    audio_profile: Dict[str, Any] = field(default_factory=dict)
    annotation_or_reference_profile: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dataset_root: Optional[Path] = None
    raw_root: Optional[Path] = None


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(text or ""))
    return text.strip("._") or "item"


def is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS and not any(
        part.startswith(".") or part == "__MACOSX" for part in path.parts
    )


def extract_zip(zip_path: Path, dest: Path) -> List[str]:
    warnings: List[str] = []
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        bad = zf.testzip()
        if bad:
            warnings.append(f"Corrupted entry in zip: {bad}")
        zf.extractall(str(dest))
    return warnings


def find_dataset_root(extracted: Path) -> Path:
    try:
        entries = [e for e in extracted.iterdir() if e.name not in {"__MACOSX"} and not e.name.startswith(".")]
    except Exception:
        return extracted
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted


def collect_audio_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_audio_file(p)])


def safe_relative_path(rel: str) -> Optional[str]:
    if not rel:
        return None
    rel = rel.strip().replace("\\", "/")
    if rel.startswith("/") or ":" in rel.split("/")[0]:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return "/".join(parts) if parts else None


def resolve_audio_path(rel: str, root: Path, audio_index: Dict[str, Path]) -> Optional[Path]:
    if not rel:
        return None
    safe = safe_relative_path(rel)
    if safe is None:
        return None
    candidate = (root / safe)
    if candidate.exists() and candidate.is_file():
        return candidate
    leaf = Path(safe).name
    return audio_index.get(leaf) or audio_index.get(leaf.lower())


def build_audio_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in collect_audio_files(root):
        index.setdefault(path.name, path)
        index.setdefault(path.name.lower(), path)
    return index


def find_metadata_files(root: Path, exts: set) -> List[Path]:
    return sorted([
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in exts
        and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)
    ])


def quick_audio_props(path: Path) -> Tuple[float, int, int, str]:
    try:
        suffix = path.suffix.lower()
        if suffix == ".wav":
            from scipy.io import wavfile
            sr, data = wavfile.read(str(path))
            ch = 1 if data.ndim == 1 else int(data.shape[1])
            duration = float(len(data) / max(sr, 1))
            return duration, int(sr), ch, "ok"
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return float(info.duration or 0.0), int(info.samplerate or 0), int(info.channels or 0), "ok"
        except Exception:
            return 0.0, 0, 0, "unreadable"
    except Exception:
        return 0.0, 0, 0, "unreadable"


def derive_label(sample: AudioSample) -> str:
    if sample.label:
        return sample.label
    if sample.speaker_id:
        return sample.speaker_id
    if sample.event_label:
        return sample.event_label
    if sample.anomaly_label:
        return sample.anomaly_label
    return "unlabeled"


def has_class_labels(samples: List[AudioSample]) -> bool:
    return any(bool(s.label) for s in samples)


def has_speaker_labels(samples: List[AudioSample]) -> bool:
    return any(bool(s.speaker_id) for s in samples)


def has_speaker_pairs(samples: List[AudioSample]) -> bool:
    return any(bool(s.audio_path_a) and bool(s.audio_path_b) for s in samples)


def has_transcripts(samples: List[AudioSample]) -> bool:
    return any(bool(s.transcript) for s in samples)


def has_temporal_segments(samples: List[AudioSample]) -> bool:
    return any(s.start_time is not None and s.end_time is not None for s in samples)


def has_event_labels(samples: List[AudioSample]) -> bool:
    return any(bool(s.event_label) for s in samples)


def has_anomaly_labels(samples: List[AudioSample]) -> bool:
    return any(bool(s.anomaly_label) for s in samples)


def has_noisy_clean_pairs(samples: List[AudioSample]) -> bool:
    return any(bool(s.noisy_path) and bool(s.clean_path) for s in samples)


def annotation_or_reference_profile(samples: List[AudioSample]) -> Dict[str, Any]:
    label_dist = Counter(s.label for s in samples if s.label)
    speakers = sorted({s.speaker_id for s in samples if s.speaker_id})
    event_dist = Counter(s.event_label for s in samples if s.event_label)
    anomaly_dist = Counter(s.anomaly_label for s in samples if s.anomaly_label)
    splits = Counter((s.split or "default") for s in samples)
    transcripts = sum(1 for s in samples if s.transcript)
    seg_count = sum(1 for s in samples if s.start_time is not None and s.end_time is not None)
    pair_count = sum(1 for s in samples if s.noisy_path and s.clean_path)
    speaker_pairs = sum(1 for s in samples if s.audio_path_a and s.audio_path_b)
    return {
        "label_count": len(label_dist),
        "label_distribution": dict(label_dist),
        "speaker_count": len(speakers),
        "transcript_count": transcripts,
        "segment_annotation_count": seg_count,
        "noisy_clean_pair_count": pair_count,
        "speaker_pair_count": speaker_pairs,
        "event_label_distribution": dict(event_dist),
        "anomaly_label_distribution": dict(anomaly_dist),
        "split_summary": dict(splits),
    }


def audio_profile_summary(samples: List[AudioSample], file_format_distribution: Dict[str, int]) -> Dict[str, Any]:
    durations = [s.duration for s in samples if s.duration > 0]
    sample_rates = Counter(str(s.sample_rate) for s in samples if s.sample_rate > 0)
    channels = Counter(str(s.channels) for s in samples if s.channels > 0)
    if durations:
        d_min, d_max = float(min(durations)), float(max(durations))
        d_mean = float(sum(durations) / len(durations))
    else:
        d_min = d_max = d_mean = 0.0
    return {
        "audio_count": len(samples),
        "duration_distribution": {
            "min": d_min, "max": d_max, "mean": d_mean,
            "buckets": _duration_buckets(durations),
        },
        "sample_rate_distribution": dict(sample_rates),
        "channel_distribution": dict(channels),
        "file_format_distribution": dict(file_format_distribution),
    }


def _duration_buckets(durations: List[float]) -> Dict[str, int]:
    counts = Counter()
    for d in durations:
        if d < 1:
            counts["<1s"] += 1
        elif d < 5:
            counts["1-5s"] += 1
        elif d < 15:
            counts["5-15s"] += 1
        elif d < 60:
            counts["15-60s"] += 1
        else:
            counts["60s+"] += 1
    return dict(counts)


def materialize_for_pipeline(dataset: InternalAudioDataset, work_dir: Path, task_type: str) -> Path:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    task_type = (task_type or "").strip().lower()

    used_names: Dict[str, int] = {}

    def _next_name(label: str, src: Path) -> Tuple[Path, str]:
        cls_dir = work_dir / safe_name(label or "unlabeled")
        cls_dir.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lower() or ".wav"
        stem = safe_name(src.stem)
        base = f"{stem}{ext}"
        key = f"{cls_dir.name}/{base}"
        if key in used_names:
            used_names[key] += 1
            target_name = f"{stem}_{used_names[key]}{ext}"
        else:
            used_names[key] = 0
            target_name = base
        return cls_dir / target_name, f"{cls_dir.name}/{target_name}"

    rows: List[Dict[str, Any]] = []
    transcripts: List[Dict[str, str]] = []
    noise_pairs: List[Dict[str, str]] = []
    speaker_pairs: List[Dict[str, str]] = []
    event_segments: List[Dict[str, Any]] = []
    anomaly_rows: List[Dict[str, str]] = []

    for sample in dataset.samples:
        src = Path(sample.audio_path) if sample.audio_path else None
        if src is None or not src.exists():
            continue

        if task_type == "noise_suppression" and sample.noisy_path and sample.clean_path:
            noisy_src = Path(sample.noisy_path)
            clean_src = Path(sample.clean_path)
            if not noisy_src.exists() or not clean_src.exists():
                continue
            target_noisy, rel_noisy = _next_name("noisy", noisy_src)
            target_clean, rel_clean = _next_name("clean", clean_src)
            try:
                shutil.copy2(noisy_src, target_noisy)
                shutil.copy2(clean_src, target_clean)
            except Exception:
                continue
            noise_pairs.append({"noisy_path": rel_noisy, "clean_path": rel_clean, "noise_type": str(sample.metadata.get("noise_type", "")), "snr_level": str(sample.metadata.get("snr_level", ""))})
            rows.append({
                "file": rel_noisy, "audio_path": rel_noisy, "label": "noisy", "split": sample.split,
                "noisy_path": rel_noisy, "clean_path": rel_clean,
            })
            continue

        if task_type == "speaker_recognition" and sample.audio_path_a and sample.audio_path_b:
            a_src = Path(sample.audio_path_a)
            b_src = Path(sample.audio_path_b)
            if not a_src.exists() or not b_src.exists():
                continue
            target_a, rel_a = _next_name("pair_a", a_src)
            target_b, rel_b = _next_name("pair_b", b_src)
            try:
                shutil.copy2(a_src, target_a)
                shutil.copy2(b_src, target_b)
            except Exception:
                continue
            speaker_pairs.append({
                "audio_path_a": rel_a,
                "audio_path_b": rel_b,
                "same_speaker": str(int(sample.same_speaker)) if sample.same_speaker is not None else "",
            })
            continue

        label = derive_label(sample)
        target, rel = _next_name(label, src)
        try:
            shutil.copy2(src, target)
        except Exception:
            continue

        row = {
            "file": rel,
            "audio_path": rel,
            "label": sample.label,
            "speaker_id": sample.speaker_id,
            "transcript": sample.transcript,
            "event_label": sample.event_label,
            "anomaly_label": sample.anomaly_label,
            "split": sample.split,
            "duration": f"{sample.duration:.4f}" if sample.duration else "",
            "sample_rate": str(sample.sample_rate or ""),
            "start_time": "" if sample.start_time is None else f"{sample.start_time:.4f}",
            "end_time": "" if sample.end_time is None else f"{sample.end_time:.4f}",
        }
        rows.append(row)

        if sample.transcript:
            transcripts.append({"file": rel, "transcript": sample.transcript})
        if sample.start_time is not None and sample.end_time is not None and sample.event_label:
            event_segments.append({
                "file": rel, "event_label": sample.event_label,
                "start_time": f"{sample.start_time:.4f}", "end_time": f"{sample.end_time:.4f}",
            })
        if sample.anomaly_label:
            anomaly_rows.append({"file": rel, "anomaly_label": sample.anomaly_label})

    _write_manifest(work_dir / "metadata.csv", rows)
    if transcripts:
        _write_manifest(work_dir / "transcripts.csv", transcripts)
    if noise_pairs:
        _write_manifest(work_dir / "pairs_noise.csv", noise_pairs)
    if speaker_pairs:
        _write_manifest(work_dir / "pairs_speaker.csv", speaker_pairs)
    if event_segments:
        _write_manifest(work_dir / "events.csv", event_segments)
    if anomaly_rows:
        _write_manifest(work_dir / "anomaly.csv", anomaly_rows)

    return work_dir


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})



