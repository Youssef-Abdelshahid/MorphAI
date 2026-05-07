from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import pandas as pd

try:
    import yaml as _yaml
    _YAML_OK = True
except Exception:
    _yaml = None
    _YAML_OK = False


_SCALAR_ARRAY_JOIN_LIMIT = 16
_RECORD_LIST_MIN_LEN = 2


@dataclass
class StructureProfile:
    input_format: str = ""
    original_record_count: int = 0
    parsed_record_count: int = 0
    invalid_record_count: int = 0
    detected_record_path: str = ""
    max_nesting_depth: int = 0
    average_nesting_depth: float = 0.0
    unique_field_paths: int = 0
    schema_variant_count: int = 0
    flattened_column_count: int = 0
    array_field_count: int = 0
    object_field_count: int = 0
    scalar_field_count: int = 0
    type_inconsistency_count: int = 0
    flattened_column_mapping: Dict[str, str] = field(default_factory=dict)
    parser_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_format": self.input_format,
            "original_record_count": self.original_record_count,
            "parsed_record_count": self.parsed_record_count,
            "invalid_record_count": self.invalid_record_count,
            "detected_record_path": self.detected_record_path,
            "max_nesting_depth": self.max_nesting_depth,
            "average_nesting_depth": round(self.average_nesting_depth, 3),
            "unique_field_paths": self.unique_field_paths,
            "schema_variant_count": self.schema_variant_count,
            "flattened_column_count": self.flattened_column_count,
            "array_field_count": self.array_field_count,
            "object_field_count": self.object_field_count,
            "scalar_field_count": self.scalar_field_count,
            "type_inconsistency_count": self.type_inconsistency_count,
            "flattened_column_mapping": dict(self.flattened_column_mapping),
            "parser_warnings": list(self.parser_warnings),
        }


@dataclass
class ParseOutcome:
    ok: bool
    error: str = ""
    df: Optional[pd.DataFrame] = None
    structure: StructureProfile = field(default_factory=StructureProfile)
    parsing_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _join_path(parts: List[str]) -> str:
    return ".".join(p for p in parts if p)


def _flatten_record(rec: Any, prefix: str = "", out: Optional[Dict[str, Any]] = None,
                    depth_tracker: Optional[List[int]] = None,
                    depth: int = 1, array_paths: Optional[set] = None,
                    object_paths: Optional[set] = None) -> Dict[str, Any]:
    if out is None:
        out = {}
    if depth_tracker is not None:
        depth_tracker[0] = max(depth_tracker[0], depth)
    if array_paths is None:
        array_paths = set()
    if object_paths is None:
        object_paths = set()

    if rec is None or _is_scalar(rec):
        out[prefix or "value"] = rec
        return out

    if isinstance(rec, dict):
        if prefix:
            object_paths.add(prefix)
        if not rec:
            out[prefix or "value"] = None
            return out
        for k, v in rec.items():
            key = str(k)
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_record(v, new_prefix, out, depth_tracker, depth + 1,
                            array_paths, object_paths)
        return out

    if isinstance(rec, list):
        array_paths.add(prefix or "value")
        if not rec:
            out[prefix or "value"] = None
            return out
        if all(_is_scalar(x) for x in rec):
            if len(rec) <= _SCALAR_ARRAY_JOIN_LIMIT:
                out[prefix or "value"] = "|".join("" if x is None else str(x) for x in rec)
            else:
                out[prefix or "value"] = "|".join(
                    "" if x is None else str(x) for x in rec[:_SCALAR_ARRAY_JOIN_LIMIT]
                ) + f"|...(+{len(rec) - _SCALAR_ARRAY_JOIN_LIMIT})"
            out[(prefix + ".count") if prefix else "value.count"] = len(rec)
        else:
            try:
                out[prefix or "value"] = json.dumps(rec, ensure_ascii=False, default=str)
            except Exception:
                out[prefix or "value"] = str(rec)
            out[(prefix + ".count") if prefix else "value.count"] = len(rec)
        return out

    out[prefix or "value"] = str(rec)
    return out


def _records_to_dataframe(
    records: List[Any],
    structure: StructureProfile,
) -> pd.DataFrame:
    flat_rows: List[Dict[str, Any]] = []
    depth_tracker = [0]
    array_paths: set = set()
    object_paths: set = set()
    schema_signatures: set = set()
    depth_per_record: List[int] = []

    for rec in records:
        rec_depth_tracker = [0]
        if isinstance(rec, dict) or isinstance(rec, list):
            row = _flatten_record(
                rec, "", None, rec_depth_tracker, 1, array_paths, object_paths
            )
        else:
            row = {"value": rec}
            rec_depth_tracker[0] = 1
        flat_rows.append(row)
        depth_tracker[0] = max(depth_tracker[0], rec_depth_tracker[0])
        depth_per_record.append(rec_depth_tracker[0])
        schema_signatures.add(tuple(sorted(row.keys())))

    df = pd.DataFrame(flat_rows)
    df = _coerce_types(df)

    structure.max_nesting_depth = depth_tracker[0]
    structure.average_nesting_depth = (
        sum(depth_per_record) / len(depth_per_record) if depth_per_record else 0.0
    )
    structure.unique_field_paths = len(df.columns)
    structure.schema_variant_count = len(schema_signatures)
    structure.flattened_column_count = len(df.columns)
    structure.array_field_count = len(array_paths)
    structure.object_field_count = len(object_paths)
    structure.scalar_field_count = max(
        0, len(df.columns) - structure.array_field_count
    )
    structure.flattened_column_mapping = {c: c for c in df.columns}

    incons = 0
    for col in df.columns:
        types_seen = {type(v).__name__ for v in df[col].dropna().tolist()}
        if len(types_seen) > 1:
            incons += 1
    structure.type_inconsistency_count = incons

    structure.parsed_record_count = len(df)
    return df


_BOOL_STRS_TRUE = {"true", "yes", "y"}
_BOOL_STRS_FALSE = {"false", "no", "n"}


def _coerce_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.dtype != object:
        return series

    sample = series.dropna()
    if sample.empty:
        return series

    str_sample = sample.astype(str).str.strip()
    lower = str_sample.str.lower()

    if (lower.isin(_BOOL_STRS_TRUE | _BOOL_STRS_FALSE)).all() and len(lower) > 0:
        return series.apply(
            lambda v: True if (isinstance(v, str) and v.strip().lower() in _BOOL_STRS_TRUE)
            else (False if (isinstance(v, str) and v.strip().lower() in _BOOL_STRS_FALSE) else v)
        )

    try:
        coerced = pd.to_numeric(str_sample, errors="raise")
        full = pd.to_numeric(series, errors="coerce")
        if coerced.notna().sum() == len(str_sample):
            return full
    except Exception:
        pass

    return series


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = _coerce_series(df[col])
        except Exception:
            continue
    return df


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


def _find_candidate_record_lists(data: Any, prefix: str = "",
                                 acc: Optional[List[Tuple[str, List[Any]]]] = None
                                 ) -> List[Tuple[str, List[Any]]]:
    if acc is None:
        acc = []
    if isinstance(data, list):
        if (len(data) >= _RECORD_LIST_MIN_LEN
                and all(isinstance(x, dict) for x in data)):
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
    return None, "", (
        f"multiple possible record collections found ({keys}). "
        "Provide a record path to disambiguate."
    )


def parse_csv_excel(path: Path, **kwargs) -> ParseOutcome:
    suffix = path.suffix.lower()
    structure = StructureProfile(input_format="csv_excel")
    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        elif suffix in {".tsv"}:
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
    except Exception as exc:
        return ParseOutcome(ok=False, error=f"Failed to read {suffix}: {exc}")

    if df.empty:
        return ParseOutcome(ok=False, error="The uploaded file is empty.")

    structure.original_record_count = len(df)
    structure.parsed_record_count = len(df)
    structure.flattened_column_count = len(df.columns)
    structure.scalar_field_count = len(df.columns)
    structure.unique_field_paths = len(df.columns)
    structure.flattened_column_mapping = {c: c for c in df.columns}
    structure.max_nesting_depth = 1
    structure.average_nesting_depth = 1.0

    return ParseOutcome(
        ok=True,
        df=df,
        structure=structure,
        parsing_summary={
            "input_format": "csv_excel",
            "rows_read": len(df),
            "columns_read": len(df.columns),
        },
    )


def parse_json_records(path: Path, record_path: str = "", **kwargs) -> ParseOutcome:
    suffix = path.suffix.lower()
    is_jsonl = suffix in {".jsonl", ".ndjson"}

    if not is_jsonl:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                head = fh.read(2048).lstrip()
            if "\n" in head and head and head[0] not in "[{":
                is_jsonl = True
        except Exception:
            pass

    structure = StructureProfile(input_format="json_records")
    warnings: List[str] = []

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
                    except json.JSONDecodeError as e:
                        invalid += 1
                        if invalid <= 5:
                            warnings.append(f"line {ln}: invalid JSON ({e.msg})")
                        continue
                    records.append(rec)
        except Exception as exc:
            return ParseOutcome(ok=False, error=f"Failed to read JSONL: {exc}")

        structure.original_record_count = total
        structure.invalid_record_count = invalid
        if not records:
            return ParseOutcome(ok=False, error="No valid JSONL records found.")
        if invalid > 0 and invalid > total * 0.5:
            return ParseOutcome(
                ok=False,
                error=f"Too many invalid JSONL lines: {invalid}/{total}.",
            )
        if not all(isinstance(r, dict) for r in records):
            return ParseOutcome(
                ok=False,
                error="JSONL records must be JSON objects.",
            )
        structure.detected_record_path = "<jsonl>"
        df = _records_to_dataframe(records, structure)
        structure.parser_warnings = warnings
        return ParseOutcome(
            ok=True, df=df, structure=structure,
            parsing_summary={
                "input_format": "json_records",
                "subformat": "jsonl",
                "rows_read": len(df),
                "columns_read": len(df.columns),
                "invalid_lines": invalid,
            },
            warnings=warnings,
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return ParseOutcome(
            ok=False,
            error=f"Invalid JSON syntax at line {exc.lineno}, col {exc.colno}: {exc.msg}",
        )
    except Exception as exc:
        return ParseOutcome(ok=False, error=f"Failed to read JSON: {exc}")

    records, used_path, err = _select_record_list(data, record_path)
    if err:
        return ParseOutcome(ok=False, error=err)
    if not records:
        return ParseOutcome(ok=False, error="No records found in JSON file.")

    structure.original_record_count = len(records)
    structure.detected_record_path = used_path
    df = _records_to_dataframe(records, structure)
    structure.parser_warnings = warnings
    return ParseOutcome(
        ok=True, df=df, structure=structure,
        parsing_summary={
            "input_format": "json_records",
            "subformat": "json",
            "record_path": used_path,
            "rows_read": len(df),
            "columns_read": len(df.columns),
        },
        warnings=warnings,
    )


def _xml_element_to_dict(elem: ET.Element) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in elem.attrib.items():
        out[f"@{k}"] = v
    children = list(elem)
    if not children:
        text = (elem.text or "").strip()
        if out:
            if text:
                out["#text"] = text
            return out
        return {"value": text} if text else {}

    by_tag: Dict[str, List[Any]] = {}
    for child in children:
        sub = _xml_element_to_dict(child)
        # If the child has only "value", collapse to scalar
        if set(sub.keys()) == {"value"}:
            sub_val: Any = sub["value"]
        else:
            sub_val = sub
        by_tag.setdefault(child.tag, []).append(sub_val)
    for tag, vals in by_tag.items():
        out[tag] = vals[0] if len(vals) == 1 else vals
    text = (elem.text or "").strip()
    if text:
        out["#text"] = text
    return out


def _xml_find_record_elements(root: ET.Element) -> List[Tuple[str, List[ET.Element]]]:
    by_tag: Dict[str, List[ET.Element]] = {}
    for child in list(root):
        by_tag.setdefault(child.tag, []).append(child)
    candidates: List[Tuple[str, List[ET.Element]]] = []
    for tag, elems in by_tag.items():
        if len(elems) >= _RECORD_LIST_MIN_LEN:
            candidates.append((tag, elems))
    return candidates


def parse_xml_records(path: Path, record_path: str = "", **kwargs) -> ParseOutcome:
    structure = StructureProfile(input_format="xml_records")

    try:
        tree = ET.parse(str(path))
    except ET.ParseError as exc:
        return ParseOutcome(
            ok=False,
            error=f"Invalid XML syntax: {exc}",
        )
    except Exception as exc:
        return ParseOutcome(ok=False, error=f"Failed to read XML: {exc}")

    root = tree.getroot()

    used_path = ""
    record_elems: List[ET.Element] = []

    if record_path:
        rp = record_path.strip()
        rp = rp.lstrip("/")
        # Allow XPath-like or simple tag name. Resolve relative to root and root.children.
        try:
            if rp == root.tag:
                # User pointed to root; treat its direct children as records
                record_elems = list(root)
                used_path = root.tag
            else:
                found = root.findall(rp)
                if not found:
                    found = root.findall(f".//{rp}")
                if not found:
                    return ParseOutcome(
                        ok=False,
                        error=f"XML record path '{record_path}' not found.",
                    )
                record_elems = found
                used_path = rp
        except Exception as exc:
            return ParseOutcome(
                ok=False,
                error=f"Invalid XML record path '{record_path}': {exc}",
            )
    else:
        candidates = _xml_find_record_elements(root)
        if not candidates:
            # Treat root's children as a single-record group if it's leaf-like
            if list(root):
                record_elems = [root]
                used_path = root.tag
            else:
                return ParseOutcome(
                    ok=False,
                    error="No repeated record elements found in XML.",
                )
        elif len(candidates) == 1:
            tag, elems = candidates[0]
            record_elems = elems
            used_path = tag
        else:
            tags = ", ".join(c[0] for c in candidates)
            return ParseOutcome(
                ok=False,
                error=(
                    f"Multiple repeated record element types found ({tags}). "
                    "Provide a record element name to disambiguate."
                ),
            )

    records: List[Dict[str, Any]] = []
    invalid = 0
    for elem in record_elems:
        try:
            rec = _xml_element_to_dict(elem)
            records.append(rec)
        except Exception:
            invalid += 1

    structure.original_record_count = len(record_elems)
    structure.invalid_record_count = invalid
    structure.detected_record_path = used_path

    if not records:
        return ParseOutcome(ok=False, error="No XML records could be parsed.")

    df = _records_to_dataframe(records, structure)
    return ParseOutcome(
        ok=True, df=df, structure=structure,
        parsing_summary={
            "input_format": "xml_records",
            "record_element": used_path,
            "rows_read": len(df),
            "columns_read": len(df.columns),
            "invalid_records": invalid,
        },
    )


def parse_yaml_records(path: Path, record_path: str = "", **kwargs) -> ParseOutcome:
    structure = StructureProfile(input_format="yaml_records")

    if not _YAML_OK:
        return ParseOutcome(
            ok=False,
            error="PyYAML is not installed. Install it (pip install pyyaml) to ingest YAML.",
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
    except Exception as exc:
        return ParseOutcome(ok=False, error=f"Invalid YAML syntax: {exc}")

    if data is None:
        return ParseOutcome(ok=False, error="YAML file is empty or null.")

    records, used_path, err = _select_record_list(data, record_path)
    if err:
        return ParseOutcome(ok=False, error=err)
    if not records:
        return ParseOutcome(ok=False, error="No records found in YAML file.")

    structure.original_record_count = len(records)
    structure.detected_record_path = used_path

    df = _records_to_dataframe(records, structure)
    return ParseOutcome(
        ok=True, df=df, structure=structure,
        parsing_summary={
            "input_format": "yaml_records",
            "record_path": used_path,
            "rows_read": len(df),
            "columns_read": len(df.columns),
        },
    )


_PARSERS = {
    "csv_excel": parse_csv_excel,
    "json_records": parse_json_records,
    "xml_records": parse_xml_records,
    "yaml_records": parse_yaml_records,
}


def parse_for_format(format_key: str, path: Path, record_path: str = "") -> ParseOutcome:
    parser = _PARSERS.get(format_key)
    if parser is None:
        return ParseOutcome(
            ok=False,
            error=f"Unsupported tabular input format '{format_key}'.",
        )
    return parser(path, record_path=record_path)
