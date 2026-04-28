from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import TextConfig
from .preprocessing import TextPipelineSpec, clean_text_value
from .profiler import TextProfile

PROCESSED_DIR = Path("processed")


def _pipeline_short_id(spec: TextPipelineSpec) -> str:
    parts = [
        "lower" if spec.lowercase else "case",
        "clean" if spec.clean_urls_emails_html else "raw",
        spec.representation.replace("_", "")[:8],
        f"len{spec.max_sequence_length if spec.max_sequence_length < 100000 else 'full'}",
    ]
    if spec.stopword_removal:
        parts.append("nostop")
    if spec.normalization_strategy != "none":
        parts.append(spec.normalization_strategy[:4])
    return "_".join(parts)


def save_processed_dataset(spec: TextPipelineSpec, df: pd.DataFrame, profile: TextProfile, config: TextConfig) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = config.data_path.suffix.lower()
    out_suffix = ".xlsx" if suffix in {".xlsx", ".xls"} else ".csv"
    out_path = PROCESSED_DIR / f"{config.data_path.stem}_{_pipeline_short_id(spec)}_cleaned{out_suffix}"
    out = df.copy()
    preserve_alignment = config.task_type in {"ner", "pos"}
    for col in profile.primary_text_columns:
        if col in out.columns:
            out[f"{col}_processed"] = [clean_text_value(v, spec, preserve_alignment=preserve_alignment) for v in out[col]]
    if out_suffix == ".xlsx":
        out.to_excel(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    return out_path, out.shape
