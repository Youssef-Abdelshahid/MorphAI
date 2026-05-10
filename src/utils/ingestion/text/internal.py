from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SUPPORTED_TXT_EXTS = {".txt", ".md", ".text"}
SUPPORTED_METADATA_EXTS = {".csv", ".json", ".jsonl", ".ndjson"}
SUPPORTED_CONLL_EXTS = {".conll", ".conllu", ".bio", ".iob", ".iob2"}


@dataclass
class InternalTextDataset:
    modality: str = "text"
    input_format: str = ""
    original_format: str = ""
    dataframe: Optional[pd.DataFrame] = None
    structure_profile: Dict[str, Any] = field(default_factory=dict)
    parsing_summary: Dict[str, Any] = field(default_factory=dict)
    field_mapping: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    dataset_root: Optional[Path] = None


def safe_relative_path(rel: str) -> Optional[str]:
    if rel is None:
        return None
    rel = str(rel).strip().replace("\\", "/")
    if not rel:
        return None
    if rel.startswith("/") or ":" in rel.split("/")[0]:
        return None
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        return None
    return "/".join(parts) if parts else None


def _is_visible_file(path: Path) -> bool:
    return path.is_file() and not any(part.startswith(".") or part == "__MACOSX" for part in path.parts)


def extract_zip(zip_path: Path, dest: Path) -> List[str]:
    warnings: List[str] = []
    if not zipfile.is_zipfile(str(zip_path)):
        raise ValueError(f"'{zip_path.name}' is not a valid zip archive.")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        bad = zf.testzip()
        if bad:
            warnings.append(f"Corrupted entry in zip: {bad}")
        for member in zf.infolist():
            name = member.filename
            if not name:
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                warnings.append(f"Skipped unsafe path in zip: {name}")
                continue
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


def collect_files(root: Path, exts: Optional[set] = None) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*"):
        if not _is_visible_file(p):
            continue
        if exts is None or p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def find_metadata_files(root: Path, exts: set = SUPPORTED_METADATA_EXTS) -> List[Path]:
    return collect_files(root, exts)


def read_text_file(path: Path) -> Tuple[str, str]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except Exception:
        return "", "unreadable"
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding), "ok"
        except UnicodeDecodeError:
            continue
    try:
        return raw.decode("utf-8", errors="replace"), "ok"
    except Exception:
        return "", "unreadable"


def normalize_for_storage(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._\-]+", "_", str(text or ""))
    return text.strip("._") or "item"


def export_internal_metadata(dataset: InternalTextDataset, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_format": dataset.input_format,
        "original_format": dataset.original_format,
        "field_mapping": dict(dataset.field_mapping or {}),
        "structure_profile": dict(dataset.structure_profile or {}),
        "parsing_summary": dict(dataset.parsing_summary or {}),
        "warnings": list(dataset.warnings or []),
        "n_records": int(len(dataset.dataframe)) if dataset.dataframe is not None else 0,
        "columns": list(dataset.dataframe.columns) if dataset.dataframe is not None else [],
    }
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


_RECORD_LIST_MIN_LEN = 2


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _flatten_record(rec: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if rec is None or _is_scalar(rec):
        out[prefix or "value"] = rec
        return out
    if isinstance(rec, dict):
        if not rec:
            out[prefix or "value"] = None
            return out
        for k, v in rec.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_record(v, new_prefix, out)
        return out
    if isinstance(rec, list):
        if not rec:
            out[prefix or "value"] = None
            return out
        if all(_is_scalar(x) for x in rec):
            try:
                out[prefix or "value"] = json.dumps(rec, ensure_ascii=False, default=str)
            except Exception:
                out[prefix or "value"] = "|".join("" if x is None else str(x) for x in rec)
        else:
            try:
                out[prefix or "value"] = json.dumps(rec, ensure_ascii=False, default=str)
            except Exception:
                out[prefix or "value"] = str(rec)
        return out
    out[prefix or "value"] = str(rec)
    return out


def _resolve_record_path(data: Any, path: str) -> Tuple[Optional[List[Any]], str]:
    if not path:
        return None, ""
    cur = data
    parts = [p for p in re.split(r"[.\[\]]", path) if p]
    for p in parts:
        if isinstance(cur, dict):
            if p not in cur:
                return None, f"record path segment '{p}' not found"
            cur = cur[p]
        elif isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except Exception:
                return None, f"invalid list index '{p}' in record path"
        else:
            return None, f"cannot descend into '{p}' (not dict/list)"
    if not isinstance(cur, list):
        return None, "record path does not point to a list"
    return cur, ""


def _find_candidate_record_lists(data: Any, prefix: str = "", acc: Optional[List[Tuple[str, List[Any]]]] = None) -> List[Tuple[str, List[Any]]]:
    if acc is None:
        acc = []
    if isinstance(data, list):
        if len(data) >= _RECORD_LIST_MIN_LEN and all(isinstance(x, dict) for x in data):
            acc.append((prefix or "<root>", data))
    elif isinstance(data, dict):
        for k, v in data.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            _find_candidate_record_lists(v, new_prefix, acc)
    return acc


def _select_record_list(data: Any, record_path: str) -> Tuple[Optional[List[Any]], str, str]:
    if record_path:
        recs, err = _resolve_record_path(data, record_path)
        if err:
            return None, record_path, err
        if not all(isinstance(x, dict) for x in recs):
            return None, record_path, "record path elements are not all objects/records"
        return recs, record_path, ""
    if isinstance(data, list):
        if not data:
            return None, "<root>", "input list is empty"
        if all(isinstance(x, dict) for x in data):
            return data, "<root>", ""
        return None, "<root>", "input list elements are not all objects/records"
    candidates = _find_candidate_record_lists(data)
    if not candidates:
        if isinstance(data, dict):
            return [data], "<root>", ""
        return None, "", "no record list found in input"
    if len(candidates) == 1:
        path, recs = candidates[0]
        return recs, path, ""
    keys = ", ".join(c[0] for c in candidates[:5])
    return None, "", f"multiple possible record collections found ({keys}). Provide a record path to disambiguate."


def parse_json_text_records(path: Path, record_path: str = "") -> Tuple[bool, str, Optional[pd.DataFrame], Dict[str, Any], Dict[str, Any], List[str]]:
    suffix = path.suffix.lower()
    is_jsonl = suffix in {".jsonl", ".ndjson"}
    if not is_jsonl:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                head = fh.read(2048).lstrip()
            if "\n" in head and head and head[0] not in "[{":
                is_jsonl = True
        except Exception:
            pass
    warnings: List[str] = []
    structure: Dict[str, Any] = {
        "input_format": "json_text_records",
        "original_record_count": 0,
        "parsed_record_count": 0,
        "invalid_record_count": 0,
        "detected_record_path": "",
        "schema_variant_count": 0,
        "nested_field_count": 0,
        "flattened_metadata_fields": [],
        "max_nesting_depth": 0,
    }
    parsing_summary: Dict[str, Any] = {
        "input_format": "json_text_records",
        "subformat": "jsonl" if is_jsonl else "json",
        "rows_read": 0,
        "columns_read": 0,
    }

    if is_jsonl:
        records: List[Any] = []
        invalid = 0
        total = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    s = line.strip()
                    if not s:
                        continue
                    total += 1
                    try:
                        rec = json.loads(s)
                    except json.JSONDecodeError as exc:
                        invalid += 1
                        if invalid <= 5:
                            warnings.append(f"line {ln}: invalid JSON ({exc.msg})")
                        continue
                    records.append(rec)
        except Exception as exc:
            return False, f"Failed to read JSONL: {exc}", None, structure, parsing_summary, warnings
        structure["original_record_count"] = total
        structure["invalid_record_count"] = invalid
        if not records:
            return False, "No valid JSONL records found.", None, structure, parsing_summary, warnings
        if invalid > 0 and invalid > total * 0.5:
            return False, f"Too many invalid JSONL lines: {invalid}/{total}.", None, structure, parsing_summary, warnings
        if not all(isinstance(r, dict) for r in records):
            return False, "JSONL records must be JSON objects.", None, structure, parsing_summary, warnings
        structure["detected_record_path"] = "<jsonl>"
        df, struct_extra = _records_to_dataframe(records)
        structure.update(struct_extra)
        parsing_summary.update({"rows_read": len(df), "columns_read": len(df.columns), "invalid_lines": invalid, "subformat": "jsonl"})
        return True, "", df, structure, parsing_summary, warnings

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON syntax at line {exc.lineno}, col {exc.colno}: {exc.msg}", None, structure, parsing_summary, warnings
    except Exception as exc:
        return False, f"Failed to read JSON: {exc}", None, structure, parsing_summary, warnings

    records, used_path, err = _select_record_list(data, record_path)
    if err:
        return False, err, None, structure, parsing_summary, warnings
    if not records:
        return False, "No records found in JSON file.", None, structure, parsing_summary, warnings
    structure["original_record_count"] = len(records)
    structure["detected_record_path"] = used_path
    df, struct_extra = _records_to_dataframe(records)
    structure.update(struct_extra)
    parsing_summary.update({"rows_read": len(df), "columns_read": len(df.columns), "record_path": used_path, "subformat": "json"})
    return True, "", df, structure, parsing_summary, warnings


def _records_to_dataframe(records: List[Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    schema_signatures: set = set()
    nested_columns: set = set()
    max_depth = 1
    for rec in records:
        if isinstance(rec, dict):
            row = _flatten_record(rec, "", None)
        else:
            row = {"value": rec}
        flat_rows.append(row)
        schema_signatures.add(tuple(sorted(row.keys())))
        for k in row.keys():
            if "." in k:
                nested_columns.add(k)
                max_depth = max(max_depth, k.count(".") + 1)
    df = pd.DataFrame(flat_rows)
    structure_extra = {
        "parsed_record_count": len(df),
        "schema_variant_count": len(schema_signatures),
        "nested_field_count": len(nested_columns),
        "flattened_metadata_fields": sorted(nested_columns),
        "max_nesting_depth": max_depth,
    }
    return df, structure_extra
