import io
import json
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile

from .config import AudioConfig
from .executor import _prepare_signal
from .preprocessing import AudioPipelineSpec
from .profiler import AudioProfile

PROCESSED_DIR = Path("processed")


_TASK_TOKENS = {
    "classification": "classification",
    "asr": "asr",
    "speaker_recognition": "speaker_recognition",
    "sound_event_detection": "sed",
    "vad": "vad",
    "anomaly": "anomaly",
    "noise_suppression": "noise_suppression",
}

_FORMAT_TOKENS = {
    "zip_folder": "folder",
    "metadata_csv": "metadata_csv",
    "metadata_json": "metadata_json",
}


def _output_stem(config: AudioConfig) -> str:
    fmt = _FORMAT_TOKENS.get(config.input_format_key, "folder")
    task = _TASK_TOKENS.get((config.task_type or "").strip().lower(), "audio")
    return f"audio_{fmt}_{task}_cleaned"


def save_processed_dataset(
    spec: AudioPipelineSpec,
    profile: AudioProfile,
    config: AudioConfig,
    internal_dataset: Optional[object] = None,
) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{_output_stem(config)}.zip"
    n_saved = 0
    used = {}
    label_set = set()
    metadata_rows = []
    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for path_str, label in zip(profile.audio_paths, profile.audio_labels):
            try:
                sr, y = _prepare_signal(path_str, spec)
                y16 = np.clip(y, -1.0, 1.0)
                y16 = (y16 * 32767.0).astype(np.int16)
                rel_label = label if label else "unlabeled"
                base = f"{rel_label}/{Path(path_str).stem}.wav"
                used[base] = used.get(base, -1) + 1
                arc_name = base if used[base] == 0 else f"{rel_label}/{Path(path_str).stem}_{used[base]}.wav"
                buf = io.BytesIO()
                wavfile.write(buf, sr, y16)
                zout.writestr(arc_name, buf.getvalue())
                n_saved += 1
                if label:
                    label_set.add(label)
                metadata_rows.append({"file": arc_name, "label": rel_label})
            except Exception:
                continue
        zout.writestr("metadata.json", json.dumps(_build_metadata(config, internal_dataset, metadata_rows), indent=2, default=str))
        zout.writestr("metadata.csv", _build_metadata_csv(metadata_rows))
        zout.writestr("README.txt", _build_readme(config, internal_dataset, n_saved))
    if n_saved == 0:
        raise ValueError("No audio files could be processed and saved.")
    return out_path, (n_saved, len(label_set))


def _build_metadata(config: AudioConfig, internal_dataset, rows: list) -> dict:
    info = {
        "modality": "audio",
        "input_format": config.input_format_key,
        "input_format_label": config.input_format,
        "task_type": config.task_type,
        "metric": config.metric,
        "parsing_strategy": "ingestion adapter -> internal audio dataset -> cleaned WAV per sample",
        "metadata_conversion_strategy": "internal samples -> per-row CSV/JSON manifest with file + label",
        "output_structure": "label/file.wav (label='unlabeled' when no label is available)",
        "n_files": len(rows),
    }
    if internal_dataset is not None:
        info["label_mapping"] = {str(k): v for k, v in (getattr(internal_dataset, "label_mapping", {}) or {}).items()}
        info["parsing_summary"] = getattr(internal_dataset, "parsing_summary", {}) or {}
        info["warnings"] = list(getattr(internal_dataset, "warnings", []) or [])
    return info


def _build_metadata_csv(rows: list) -> str:
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["file", "label"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _build_readme(config: AudioConfig, internal_dataset, n_saved: int) -> str:
    parts = [
        "MorphAI cleaned audio dataset",
        "==============================",
        f"Modality        : Audio",
        f"Input format    : {config.input_format or config.input_format_key}",
        f"Task type       : {config.task_type}",
        f"Metric          : {config.metric}",
        f"Files preserved : {n_saved}",
    ]
    if internal_dataset is not None:
        ps = getattr(internal_dataset, "parsing_summary", {}) or {}
        if ps:
            parts.append("")
            parts.append("Parsing summary")
            for key, value in ps.items():
                parts.append(f"  {key}: {value}")
    return "\n".join(parts) + "\n"
