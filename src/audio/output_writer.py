import io
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile

from .config import AudioConfig
from .executor import _prepare_signal
from .preprocessing import AudioPipelineSpec
from .profiler import AudioProfile

PROCESSED_DIR = Path("processed")


def _pipeline_short_id(spec: AudioPipelineSpec) -> str:
    parts = [
        f"sr{spec.target_sample_rate if spec.target_sample_rate else 'nat'}",
        "mono" if spec.mono else "nat",
        spec.feature_representation[:4],
        spec.loudness_normalization[:3],
    ]
    if spec.trim_silence:
        parts.append("trim")
    if spec.noise_filter != "none":
        parts.append(spec.noise_filter[:3])
    if spec.clipping_handling != "none":
        parts.append("clip")
    return "_".join(parts)


def save_processed_dataset(spec: AudioPipelineSpec, profile: AudioProfile, config: AudioConfig) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{config.data_path.stem}_{_pipeline_short_id(spec)}_processed.zip"
    n_saved = 0
    used = {}
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
            except Exception:
                continue
    if n_saved == 0:
        raise ValueError("No audio files could be processed and saved.")
    return out_path, (n_saved, len(set(x for x in profile.audio_labels if x)))
