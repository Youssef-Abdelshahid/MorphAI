import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .config import TextConfig, normalize_task_type
from .preprocessing import TextPipelineSpec, clean_text_value
from .profiler import TextProfile

PROCESSED_DIR = Path("processed")


_TASK_NAME_FOR_FILE = {
    "classification_single": "classification",
    "classification_multi": "multilabel_classification",
    "ner": "ner",
    "semantic_similarity": "semantic_similarity",
    "summarization": "summarization",
    "question_answering": "qa",
    "topic_modeling": "topic_modeling",
}

_FORMAT_SLUG = {
    "csv_excel": "csv",
    "json_text_records": "json",
    "txt_zip": "txtzip",
}


def _safe_token(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(text or ""))
    return text.strip("._") or "item"


def _output_stem(config: TextConfig) -> str:
    fmt = _FORMAT_SLUG.get(config.input_format_key, _safe_token(config.input_format_key or "text"))
    task = _TASK_NAME_FOR_FILE.get(normalize_task_type(config.task_type), _safe_token(config.task_type or "text"))
    return f"text_{fmt}_{task}_cleaned"


def _build_metadata(spec: TextPipelineSpec, profile: TextProfile, config: TextConfig, output_path: Path, output_structure: str, output_record_count: int, dropped_rows: int) -> Dict[str, Any]:
    return {
        "modality": "Text",
        "input_format": config.input_format_key or config.input_format,
        "input_format_label": config.input_format,
        "original_input_path": str(config.data_path),
        "task_type": normalize_task_type(config.task_type),
        "selected_pipeline": spec.to_dict(),
        "parsing_strategy": (profile.parsing_summary or {}).get("conversion_strategy", ""),
        "record_conversion_strategy": (profile.parsing_summary or {}).get("conversion_strategy", ""),
        "selected_text_columns": list(profile.primary_text_columns or []),
        "selected_target_columns": list(profile.target_columns or []),
        "selected_auxiliary_feature_columns": list(profile.auxiliary_numeric_columns or []) + list(profile.auxiliary_categorical_columns or []),
        "row_filtering_summary": {
            "original_row_count": profile.original_row_count,
            "removed_empty_or_invalid": profile.removed_empty_or_invalid_count,
            "removed_non_english": profile.removed_non_english_count,
            "removed_language_uncertain": profile.removed_language_uncertain_count,
            "removed_too_noisy": profile.removed_too_noisy_count,
            "language_filter_method": profile.language_filter_method,
            "final_usable_rows": profile.n_samples,
            "dropped_rows_during_export": dropped_rows,
        },
        "english_filter_summary": {
            "language_filter_method": profile.language_filter_method,
            "removed_non_english": profile.removed_non_english_count,
            "removed_language_uncertain": profile.removed_language_uncertain_count,
        },
        "emoji_handling_summary": {
            "strategy": profile.emoji_strategy,
            "translated": profile.emoji_translated_count,
            "removed": profile.emoji_removed_count,
            "removed_excessive": profile.removed_excessive_emoji_count,
        },
        "output_structure": output_structure,
        "output_path": str(output_path),
        "output_record_count": int(output_record_count),
        "structure_profile": dict(profile.structure_profile or {}),
        "parsing_summary": dict(profile.parsing_summary or {}),
        "warnings": list(profile.parser_warnings or []),
    }


def _write_processed_dataframe(spec: TextPipelineSpec, df: pd.DataFrame, profile: TextProfile, config: TextConfig, *, suffix: str = ".csv") -> Tuple[Path, Tuple[int, int], Dict[str, Any]]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{_output_stem(config)}{suffix}"
    out = df.copy()
    preserve_alignment = config.task_type in {"ner", "pos"}
    for col in profile.primary_text_columns:
        if col in out.columns:
            out[f"{col}_processed"] = [clean_text_value(v, spec, preserve_alignment=preserve_alignment) for v in out[col]]
    if suffix == ".xlsx":
        out.to_excel(out_path, index=False)
    elif suffix == ".jsonl":
        with open(out_path, "w", encoding="utf-8") as fh:
            for _, row in out.iterrows():
                fh.write(json.dumps({str(k): (None if pd.isna(v) else v) for k, v in row.to_dict().items()}, ensure_ascii=False, default=str) + "\n")
    else:
        out.to_csv(out_path, index=False)
    metadata = _build_metadata(spec, profile, config, out_path, output_structure=("excel" if suffix == ".xlsx" else "jsonl" if suffix == ".jsonl" else "csv"), output_record_count=int(len(out)), dropped_rows=0)
    return out_path, out.shape, metadata


def save_processed_dataset(spec: TextPipelineSpec, df: pd.DataFrame, profile: TextProfile, config: TextConfig, internal_dataset: Optional[Any] = None) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fmt_key = config.input_format_key or "csv_excel"
    if fmt_key == "txt_zip":
        return _save_processed_txt_zip(spec, df, profile, config)
    if fmt_key == "json_text_records":
        out_path, shape, metadata = _write_processed_dataframe(spec, df, profile, config, suffix=".csv")
        meta_path = out_path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        return out_path, shape
    suffix = config.data_path.suffix.lower()
    out_suffix = ".xlsx" if suffix in {".xlsx", ".xls"} else ".csv"
    out_path, shape, metadata = _write_processed_dataframe(spec, df, profile, config, suffix=out_suffix)
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return out_path, shape


def _save_processed_txt_zip(spec: TextPipelineSpec, df: pd.DataFrame, profile: TextProfile, config: TextConfig) -> Tuple[Path, tuple]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{_output_stem(config)}.zip"
    text_col = (profile.primary_text_columns or ["text"])[0]
    label_col = profile.resolved_columns.get("label") if isinstance(profile.resolved_columns.get("label"), str) else None
    preserve_alignment = config.task_type in {"ner", "pos"}

    cleaned_rows = []
    used_names: Dict[str, int] = {}

    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for idx, row in df.reset_index(drop=True).iterrows():
            raw_text = "" if text_col not in df.columns else row.get(text_col, "")
            cleaned_text = clean_text_value(raw_text, spec, preserve_alignment=preserve_alignment)
            doc_path = str(row.get("document_path", "")) if "document_path" in df.columns else ""
            doc_id = str(row.get("document_id", "")) if "document_id" in df.columns else ""
            stem = _safe_token(Path(doc_id or doc_path or f"doc_{idx + 1}").stem) or f"doc_{idx + 1}"
            label_value = ""
            if label_col and label_col in df.columns:
                label_value = "" if pd.isna(row.get(label_col)) else str(row.get(label_col, "")).strip()
            cls_dir = _safe_token(label_value) if label_value else ""
            base = f"{stem}.txt"
            key = f"{cls_dir}/{base}" if cls_dir else base
            if key in used_names:
                used_names[key] += 1
                base = f"{stem}_{used_names[key]}.txt"
                key = f"{cls_dir}/{base}" if cls_dir else base
            else:
                used_names[key] = 0
            arc = f"documents/{cls_dir}/{base}" if cls_dir else f"documents/{base}"
            zout.writestr(arc, cleaned_text)
            row_dict: Dict[str, Any] = {}
            for col in df.columns:
                value = row.get(col)
                row_dict[str(col)] = "" if pd.isna(value) else value
            row_dict["document_path"] = arc
            row_dict["document_id"] = doc_id or stem
            row_dict[f"{text_col}_processed" if text_col else "text_processed"] = cleaned_text
            cleaned_rows.append(row_dict)

        if cleaned_rows:
            keys: list = []
            seen: set = set()
            for r in cleaned_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=keys)
            writer.writeheader()
            for r in cleaned_rows:
                writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
            zout.writestr("metadata.csv", buf.getvalue())

        metadata = _build_metadata(spec, profile, config, out_path, output_structure="zip_documents_with_metadata", output_record_count=len(cleaned_rows), dropped_rows=0)
        zout.writestr("metadata.json", json.dumps(metadata, indent=2, default=str))

    return out_path, (len(cleaned_rows), len(df.columns) + 1)
