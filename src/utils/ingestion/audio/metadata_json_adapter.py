from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.audio.internal import (
    SUPPORTED_AUDIO_EXTS,
    extract_zip,
    find_dataset_root,
    find_metadata_files,
    safe_relative_path,
)
from src.utils.ingestion.audio.metadata_csv_adapter import (
    _AUDIO_PATH_KEYS,
    _LABEL_KEYS,
    _SPEAKER_KEYS,
    _TRANSCRIPT_KEYS,
    _EVENT_KEYS,
    _ANOMALY_KEYS,
    _NOISY_KEYS,
    _CLEAN_KEYS,
    _PAIR_A_KEYS,
    _PAIR_B_KEYS,
    _SAME_SPEAKER_KEYS,
    _START_KEYS,
    _END_KEYS,
    _SPLIT_KEYS,
    _DURATION_KEYS,
    _SAMPLE_RATE_KEYS,
    _first_match,
    _materialize_records,
)


_JSON_EXTS = {".json", ".jsonl", ".ndjson"}


class MetadataJsonAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio metadata JSON / JSONL"
    is_implemented = True
    format_key = "metadata_json"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("Audio metadata JSON / JSONL requires a .zip archive containing audio files and a JSON/JSONL/NDJSON metadata file.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not zipfile.is_zipfile(str(path)):
            errors.append(f"'{path.name}' is not a valid zip archive.")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        try:
            with zipfile.ZipFile(str(path), "r") as zf:
                bad = zf.testzip()
                if bad:
                    errors.append(f"Corrupted entry detected in zip archive: '{bad}'.")
                names = zf.namelist()
        except Exception as exc:
            errors.append(f"Cannot open zip archive: {exc}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if not any(Path(n).suffix.lower() in SUPPORTED_AUDIO_EXTS for n in names):
            errors.append("Zip archive contains no supported audio files.")
        if not any(Path(n).suffix.lower() in _JSON_EXTS for n in names):
            errors.append("Zip archive contains no JSON / JSONL / NDJSON metadata file.")
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(
        self,
        path: Path,
        work_dir: Path = None,
        task_type: str = "",
        metadata_path: str = "",
        record_path: str = "",
        audio_path_field: str = "",
        label_field: str = "",
        transcript_field: str = "",
        speaker_field: str = "",
        event_field: str = "",
        anomaly_field: str = "",
        noisy_field: str = "",
        clean_field: str = "",
        pair_a_field: str = "",
        pair_b_field: str = "",
        same_speaker_field: str = "",
        **kwargs,
    ) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        if work_dir is None:
            return AdapterResult(ok=False, message="work_dir is required for audio ingestion.", errors=["work_dir is required."])
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        warnings = extract_zip(path, work_dir)
        root = find_dataset_root(work_dir)

        json_files = find_metadata_files(root, _JSON_EXTS)
        if not json_files:
            return AdapterResult(ok=False, message="No JSON/JSONL metadata file found in the package.", errors=["No JSON/JSONL metadata file found in the package."])

        chosen: Optional[Path] = None
        if metadata_path:
            safe = safe_relative_path(metadata_path)
            if safe is not None:
                cand = root / safe
                if cand.exists() and cand.is_file():
                    chosen = cand
                else:
                    leaf = Path(safe).name.lower()
                    for c in json_files:
                        if c.name.lower() == leaf:
                            chosen = c
                            break
            if chosen is None:
                return AdapterResult(ok=False, message=f"Metadata JSON path '{metadata_path}' was not found inside the package.", errors=[f"Metadata JSON path '{metadata_path}' was not found inside the package."])
        else:
            if len(json_files) == 1:
                chosen = json_files[0]
            else:
                preferred = [p for p in json_files if p.name.lower() in {"metadata.json", "manifest.json", "labels.json", "metadata.jsonl", "manifest.jsonl", "data.jsonl"}]
                if len(preferred) == 1:
                    chosen = preferred[0]
                else:
                    rels = ", ".join(str(p.relative_to(root)) for p in json_files[:5])
                    return AdapterResult(ok=False, message=f"Multiple JSON metadata files found in the package ({rels}). Provide the metadata file path/name.", errors=[f"Multiple JSON metadata files: {rels}"])

        suffix = chosen.suffix.lower()
        records: List[Dict[str, Any]] = []
        invalid_lines = 0
        try:
            if suffix in {".jsonl", ".ndjson"}:
                with open(chosen, "r", encoding="utf-8", errors="ignore") as fh:
                    for line_no, line in enumerate(fh, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            invalid_lines += 1
                            warnings.append(f"Invalid JSONL line {line_no} in {chosen.name}.")
                            continue
                        if isinstance(obj, dict):
                            records.append(obj)
                        elif isinstance(obj, list):
                            for item in obj:
                                if isinstance(item, dict):
                                    records.append(item)
                if invalid_lines and invalid_lines >= max(3, len(records)):
                    return AdapterResult(ok=False, message=f"Too many invalid JSONL lines ({invalid_lines}) in '{chosen.name}'.", errors=[f"Too many invalid JSONL lines: {invalid_lines}"])
            else:
                with open(chosen, "r", encoding="utf-8", errors="ignore") as fh:
                    payload = json.load(fh)
                records = _extract_records(payload, record_path)
                if records is None:
                    if isinstance(payload, dict):
                        candidate_keys = [k for k, v in payload.items() if isinstance(v, list) and v and isinstance(v[0], dict)]
                        if len(candidate_keys) == 1:
                            records = payload[candidate_keys[0]]
                        elif len(candidate_keys) > 1:
                            return AdapterResult(ok=False, message=f"Multiple candidate record lists in JSON metadata: {candidate_keys}. Provide the record path.", errors=[f"Multiple candidate record lists: {candidate_keys}"])
                        else:
                            return AdapterResult(ok=False, message="JSON metadata does not contain a list of records.", errors=["JSON metadata does not contain a list of records."])
                    else:
                        return AdapterResult(ok=False, message="JSON metadata is not a list of records or an object containing one.", errors=["JSON metadata is not a list of records."])
        except json.JSONDecodeError as exc:
            return AdapterResult(ok=False, message=f"Invalid JSON metadata: {exc}", errors=[f"Invalid JSON metadata: {exc}"])
        except Exception as exc:
            return AdapterResult(ok=False, message=f"Could not read JSON metadata: {exc}", errors=[f"Could not read JSON metadata: {exc}"])

        if not records:
            return AdapterResult(ok=False, message="The JSON metadata file does not contain any records.", errors=["The JSON metadata file does not contain any records."])

        flat_rows: List[Dict[str, Any]] = []
        for rec in records:
            if isinstance(rec, dict):
                flat_rows.append(_flatten(rec))

        if not flat_rows:
            return AdapterResult(ok=False, message="The JSON metadata file did not yield any usable records.", errors=["The JSON metadata file did not yield any usable records."])

        headers = list(_collect_keys(flat_rows))
        lookup = {h.lower(): h for h in headers}

        path_col = audio_path_field or _first_match(lookup, _AUDIO_PATH_KEYS)
        label_col = label_field or _first_match(lookup, _LABEL_KEYS)
        speaker_col = speaker_field or _first_match(lookup, _SPEAKER_KEYS)
        transcript_col = transcript_field or _first_match(lookup, _TRANSCRIPT_KEYS)
        event_col = event_field or _first_match(lookup, _EVENT_KEYS)
        anomaly_col = anomaly_field or _first_match(lookup, _ANOMALY_KEYS)
        noisy_col = noisy_field or _first_match(lookup, _NOISY_KEYS)
        clean_col = clean_field or _first_match(lookup, _CLEAN_KEYS)
        pair_a_col = pair_a_field or _first_match(lookup, _PAIR_A_KEYS)
        pair_b_col = pair_b_field or _first_match(lookup, _PAIR_B_KEYS)
        same_speaker_col = same_speaker_field or _first_match(lookup, _SAME_SPEAKER_KEYS)
        start_col = _first_match(lookup, _START_KEYS)
        end_col = _first_match(lookup, _END_KEYS)
        split_col = _first_match(lookup, _SPLIT_KEYS)
        duration_col = _first_match(lookup, _DURATION_KEYS)
        sr_col = _first_match(lookup, _SAMPLE_RATE_KEYS)

        if not path_col and not (pair_a_col and pair_b_col) and not (noisy_col and clean_col):
            return AdapterResult(
                ok=False,
                message="Required audio_path/file_path field not found in the JSON metadata.",
                errors=["Required audio_path/file_path field not found in the JSON metadata."],
            )

        if invalid_lines:
            warnings.append(f"Skipped {invalid_lines} invalid JSONL line(s).")

        normalized_rows = [{k: ("" if v is None else (v if isinstance(v, str) else str(v))) for k, v in row.items()} for row in flat_rows]

        result = _materialize_records(
            rows=normalized_rows,
            root=root,
            warnings=warnings,
            chosen_metadata_path=chosen,
            adapter_format_key=self.format_key,
            adapter_original="audio_metadata_json",
            task_type=task_type,
            cols={
                "path": path_col, "label": label_col, "speaker": speaker_col,
                "transcript": transcript_col, "event": event_col, "anomaly": anomaly_col,
                "noisy": noisy_col, "clean": clean_col, "pair_a": pair_a_col,
                "pair_b": pair_b_col, "same_speaker": same_speaker_col,
                "start": start_col, "end": end_col, "split": split_col,
                "duration": duration_col, "sample_rate": sr_col,
            },
        )
        if result.ok and result.data and "internal_dataset" in result.data:
            ds = result.data["internal_dataset"]
            ds.parsing_summary["metadata_format"] = "jsonl" if suffix in {".jsonl", ".ndjson"} else "json"
            ds.parsing_summary["invalid_jsonl_line_count"] = invalid_lines
            ds.metadata_profile["invalid_jsonl_line_count"] = invalid_lines
        return result


def _extract_records(payload: Any, record_path: str) -> Optional[List[Dict[str, Any]]]:
    if record_path:
        cur = payload
        for part in record_path.split("."):
            part = part.strip()
            if not part:
                continue
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list):
                try:
                    cur = cur[int(part)]
                except Exception:
                    return None
            else:
                return None
        if isinstance(cur, list):
            return [r for r in cur if isinstance(r, dict)]
        return None
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return None


def _flatten(rec: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            nested = _flatten(v, key)
            for nk, nv in nested.items():
                out[nk] = nv
            out[key] = json.dumps(v, default=str)
        elif isinstance(v, list):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
            if "." in key:
                out.setdefault(k, v)
    return out


def _collect_keys(rows: List[Dict[str, Any]]) -> List[str]:
    seen: List[str] = []
    seen_set: set = set()
    for row in rows:
        for k in row.keys():
            if k not in seen_set:
                seen_set.add(k)
                seen.append(k)
    return seen
