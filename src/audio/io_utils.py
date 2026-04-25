from pathlib import Path
from typing import Optional, Tuple

import numpy as np

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
ANNOTATION_EXTENSIONS = {".csv", ".json", ".txt", ".rttm", ".lab", ".TextGrid"}


def read_audio(path: Path) -> Tuple[int, np.ndarray, Optional[int]]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        bit_depth = None
        if data.dtype == np.int16:
            bit_depth = 16
        elif data.dtype == np.int32:
            bit_depth = 32
        elif data.dtype == np.uint8:
            bit_depth = 8
        arr = data.astype(np.float32)
        if data.dtype.kind in {"i", "u"}:
            max_abs = float(np.iinfo(data.dtype).max)
            if max_abs > 0:
                arr = arr / max_abs
        return int(sr), arr, bit_depth
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=False)
        info = sf.info(str(path))
        subtype = (getattr(info, "subtype", "") or "").upper()
        bit_depth = None
        for token in ["PCM_16", "PCM_24", "PCM_32"]:
            if token in subtype:
                bit_depth = int(token.split("_")[1])
        return int(sr), np.asarray(data, dtype=np.float32), bit_depth
    except ImportError as exc:
        raise ValueError(f"{suffix} decoding requires optional soundfile support; WAV is available by default.") from exc


def as_mono(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1).astype(np.float32)


def safe_duration(sr: int, data: np.ndarray) -> float:
    if sr <= 0:
        return 0.0
    return float(len(data) / sr)
