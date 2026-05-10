from __future__ import annotations

import csv
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.audio.internal import (
    AudioSample,
    InternalAudioDataset,
    SUPPORTED_AUDIO_EXTS,
    annotation_or_reference_profile,
    audio_profile_summary,
    build_audio_index,
    collect_audio_files,
    extract_zip,
    find_dataset_root,
    find_metadata_files,
    quick_audio_props,
    resolve_audio_path,
    safe_relative_path,
)


_AUDIO_PATH_KEYS = ["audio_path", "file_path", "filepath", "path", "file", "filename", "audio", "audio_file", "wav_path"]
_LABEL_KEYS = ["label", "class", "target", "category", "labels"]
_SPEAKER_KEYS = ["speaker_id", "speaker", "spk_id", "spk"]
_TRANSCRIPT_KEYS = ["transcript", "text", "transcription", "reference", "reference_text", "sentence"]
_EVENT_KEYS = ["event_label", "event", "event_class"]
_ANOMALY_KEYS = ["anomaly_label", "anomaly", "is_anomaly"]
_NOISY_KEYS = ["noisy_path", "noisy", "noisy_audio", "noisy_file"]
_CLEAN_KEYS = ["clean_path", "clean", "clean_audio", "clean_file", "reference_path"]
_PAIR_A_KEYS = ["audio_path_a", "audio_a", "path_a", "file_a", "wav_a"]
_PAIR_B_KEYS = ["audio_path_b", "audio_b", "path_b", "file_b", "wav_b"]
_SAME_SPEAKER_KEYS = ["same_speaker", "is_same_speaker", "same", "label_pair"]
_START_KEYS = ["start_time", "start", "begin", "onset"]
_END_KEYS = ["end_time", "end", "stop", "offset"]
_SPLIT_KEYS = ["split", "subset", "partition"]
_DURATION_KEYS = ["duration", "length"]
_SAMPLE_RATE_KEYS = ["sample_rate", "sampling_rate", "sr"]


class MetadataCsvAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio metadata CSV"
    is_implemented = True
    format_key = "metadata_csv"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("Audio metadata CSV requires a .zip archive containing audio files and a CSV metadata file.")
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
        if not any(Path(n).suffix.lower() == ".csv" for n in names):
            errors.append("Zip archive contains no CSV metadata file.")
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(
        self,
        path: Path,
        work_dir: Path = None,
        task_type: str = "",
        metadata_path: str = "",
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

        csv_files = find_metadata_files(root, {".csv"})
        if not csv_files:
            return AdapterResult(ok=False, message="No CSV metadata file found in the package.", errors=["No CSV metadata file found in the package."])

        chosen_csv: Optional[Path] = None
        if metadata_path:
            safe = safe_relative_path(metadata_path)
            if safe is not None:
                candidate = root / safe
                if candidate.exists() and candidate.is_file():
                    chosen_csv = candidate
                else:
                    leaf = Path(safe).name.lower()
                    for cand in csv_files:
                        if cand.name.lower() == leaf:
                            chosen_csv = cand
                            break
            if chosen_csv is None:
                return AdapterResult(ok=False, message=f"Metadata CSV path '{metadata_path}' was not found inside the package.", errors=[f"Metadata CSV path '{metadata_path}' was not found inside the package."])
        else:
            if len(csv_files) == 1:
                chosen_csv = csv_files[0]
            else:
                preferred = [p for p in csv_files if p.name.lower() in {"metadata.csv", "manifest.csv", "labels.csv"}]
                if len(preferred) == 1:
                    chosen_csv = preferred[0]
                else:
                    rels = ", ".join(str(p.relative_to(root)) for p in csv_files[:5])
                    return AdapterResult(ok=False, message=f"Multiple CSV files found in the package ({rels}). Provide the metadata CSV path/name.", errors=[f"Multiple CSV files found: {rels}"])

        rows: List[Dict[str, str]] = []
        try:
            with open(chosen_csv, "r", encoding="utf-8", errors="ignore", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    rows.append({(k.strip() if isinstance(k, str) else k): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k})
        except Exception as exc:
            return AdapterResult(ok=False, message=f"Could not read CSV metadata: {exc}", errors=[f"Could not read CSV metadata: {exc}"])

        if not rows:
            return AdapterResult(ok=False, message="The CSV metadata file is empty.", errors=["The CSV metadata file is empty."])

        headers = list(rows[0].keys())
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
                message="Required audio_path/file_path column not found in the CSV metadata.",
                errors=["Required audio_path/file_path column not found in the CSV metadata."],
            )

        return _materialize_records(
            rows=rows,
            root=root,
            warnings=warnings,
            chosen_metadata_path=chosen_csv,
            adapter_format_key=self.format_key,
            adapter_original="audio_metadata_csv",
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


def _first_match(lookup: Dict[str, str], candidates: List[str]) -> str:
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return ""


def _materialize_records(
    rows: List[Dict[str, Any]],
    root: Path,
    warnings: List[str],
    chosen_metadata_path: Path,
    adapter_format_key: str,
    adapter_original: str,
    task_type: str,
    cols: Dict[str, str],
) -> AdapterResult:
    audio_index = build_audio_index(root)
    all_audio = collect_audio_files(root)
    referenced: set = set()

    samples: List[AudioSample] = []
    missing = 0
    duplicates = 0
    invalid_segments = 0
    invalid_pairs = 0
    seen_keys: set = set()
    format_counter: Counter = Counter()

    def _resolve_required(rel: str) -> Optional[Path]:
        if not rel:
            return None
        return resolve_audio_path(rel, root, audio_index)

    for idx, row in enumerate(rows):
        audio_rel = str(row.get(cols["path"]) or "").strip() if cols["path"] else ""
        noisy_rel = str(row.get(cols["noisy"]) or "").strip() if cols["noisy"] else ""
        clean_rel = str(row.get(cols["clean"]) or "").strip() if cols["clean"] else ""
        pair_a_rel = str(row.get(cols["pair_a"]) or "").strip() if cols["pair_a"] else ""
        pair_b_rel = str(row.get(cols["pair_b"]) or "").strip() if cols["pair_b"] else ""
        same_speaker_raw = str(row.get(cols["same_speaker"]) or "").strip() if cols["same_speaker"] else ""

        primary: Optional[Path] = None
        noisy_path = clean_path = pair_a_path = pair_b_path = None

        if noisy_rel and clean_rel:
            noisy_path = _resolve_required(noisy_rel)
            clean_path = _resolve_required(clean_rel)
            if noisy_path is None or clean_path is None:
                invalid_pairs += 1
                missing += sum(1 for p in (noisy_path, clean_path) if p is None)
                continue
            primary = noisy_path
            referenced.add(noisy_path)
            referenced.add(clean_path)
        elif pair_a_rel and pair_b_rel:
            pair_a_path = _resolve_required(pair_a_rel)
            pair_b_path = _resolve_required(pair_b_rel)
            if pair_a_path is None or pair_b_path is None:
                invalid_pairs += 1
                missing += sum(1 for p in (pair_a_path, pair_b_path) if p is None)
                continue
            primary = pair_a_path
            referenced.add(pair_a_path)
            referenced.add(pair_b_path)
        else:
            if not audio_rel:
                missing += 1
                continue
            primary = _resolve_required(audio_rel)
            if primary is None:
                missing += 1
                continue
            referenced.add(primary)

        key = (str(primary), audio_rel, noisy_rel, clean_rel, pair_a_rel, pair_b_rel)
        if key in seen_keys:
            duplicates += 1
            continue
        seen_keys.add(key)

        duration, sr, ch, status = quick_audio_props(primary)
        if status != "ok":
            warnings.append(f"Could not read audio properties for {primary.name}.")

        if cols.get("duration") and not duration:
            try:
                duration = float(row.get(cols["duration"]) or 0.0)
            except Exception:
                pass
        if cols.get("sample_rate") and not sr:
            try:
                sr = int(float(row.get(cols["sample_rate"]) or 0))
            except Exception:
                pass

        start_v = _safe_float(row.get(cols["start"])) if cols.get("start") else None
        end_v = _safe_float(row.get(cols["end"])) if cols.get("end") else None
        if start_v is not None and end_v is not None:
            if start_v < 0 or end_v <= start_v:
                invalid_segments += 1
                start_v = None
                end_v = None
            elif duration and end_v > duration + 1e-3:
                invalid_segments += 1
                end_v = duration
                if start_v >= end_v:
                    start_v = None
                    end_v = None

        same_speaker_val: Optional[int] = None
        if same_speaker_raw:
            normalized = same_speaker_raw.strip().lower()
            if normalized in {"1", "true", "yes", "same", "y", "t"}:
                same_speaker_val = 1
            elif normalized in {"0", "false", "no", "different", "diff", "n", "f"}:
                same_speaker_val = 0

        format_counter[primary.suffix.lower().lstrip(".")] += 1

        sample = AudioSample(
            sample_id=primary.stem if not (pair_a_path and pair_b_path) else f"pair_{idx}",
            audio_path=str(primary),
            duration=float(duration or 0.0),
            sample_rate=int(sr or 0),
            channels=int(ch or 0),
            label=str(row.get(cols["label"]) or "").strip() if cols.get("label") else "",
            speaker_id=str(row.get(cols["speaker"]) or "").strip() if cols.get("speaker") else "",
            transcript=str(row.get(cols["transcript"]) or "").strip() if cols.get("transcript") else "",
            start_time=start_v,
            end_time=end_v,
            event_label=str(row.get(cols["event"]) or "").strip() if cols.get("event") else "",
            anomaly_label=str(row.get(cols["anomaly"]) or "").strip() if cols.get("anomaly") else "",
            noisy_path=str(noisy_path) if noisy_path else "",
            clean_path=str(clean_path) if clean_path else "",
            audio_path_a=str(pair_a_path) if pair_a_path else "",
            audio_path_b=str(pair_b_path) if pair_b_path else "",
            same_speaker=same_speaker_val,
            split=str(row.get(cols["split"]) or "").strip() if cols.get("split") else "",
            metadata={k: v for k, v in row.items() if k not in {cols.get("path"), cols.get("label"), cols.get("speaker"), cols.get("transcript"), cols.get("event"), cols.get("anomaly"), cols.get("noisy"), cols.get("clean"), cols.get("pair_a"), cols.get("pair_b"), cols.get("same_speaker"), cols.get("start"), cols.get("end"), cols.get("split"), cols.get("duration"), cols.get("sample_rate")} and v not in (None, "")},
        )
        samples.append(sample)

    unused = sum(1 for p in all_audio if p not in referenced)

    label_dirs = sorted({s.label for s in samples if s.label})
    speaker_ids = sorted({s.speaker_id for s in samples if s.speaker_id})
    label_mapping = {idx: name for idx, name in enumerate(label_dirs)}
    speaker_mapping = {idx: name for idx, name in enumerate(speaker_ids)}

    parsing_summary = {
        "input_format": adapter_format_key,
        "source_format": adapter_original,
        "metadata_path": str(chosen_metadata_path.relative_to(root)) if chosen_metadata_path else "",
        "metadata_format": "csv",
        "conversion_strategy": "csv_records_to_internal_audio_samples",
        "metadata_record_count": len(rows),
        "parsed_audio_count": len(samples),
        "missing_audio_count": missing,
        "unused_audio_count": unused,
        "duplicate_metadata_row_count": duplicates,
        "invalid_segment_count": invalid_segments,
        "invalid_pair_count": invalid_pairs,
        "extracted_root": str(root),
        "task_type": task_type,
        "field_mapping": {k: v for k, v in cols.items() if v},
    }

    structure_profile = {
        "input_format": adapter_format_key,
        "structure_type": "metadata_manifest_csv",
        "audio_dir_count": len({Path(s.audio_path).parent for s in samples}),
        "label_count": len(label_dirs),
        "speaker_count": len(speaker_ids),
    }

    ann_profile = annotation_or_reference_profile(samples)
    aud_profile = audio_profile_summary(samples, dict(format_counter))
    metadata_profile = {
        "input_format": adapter_format_key,
        "metadata_record_count": len(rows),
        "missing_audio_count": missing,
        "unused_audio_count": unused,
        "duplicate_metadata_row_count": duplicates,
        "metadata_field_count": len(rows[0]) if rows else 0,
        "metadata_missing_ratio": _missing_ratio(rows),
        "parsing_warnings": list(warnings),
        "original_audio_file_count": len(all_audio),
        "parsed_audio_count": len(samples),
        "corrupted_audio_count": 0,
        "unsupported_audio_count": 0,
        "duplicate_audio_count": 0,
        "label_count": ann_profile["label_count"],
        "label_distribution": ann_profile["label_distribution"],
        "speaker_count": ann_profile["speaker_count"],
        "transcript_count": ann_profile["transcript_count"],
        "segment_annotation_count": ann_profile["segment_annotation_count"],
        "noisy_clean_pair_count": ann_profile["noisy_clean_pair_count"],
        "speaker_pair_count": ann_profile["speaker_pair_count"],
        "anomaly_label_distribution": ann_profile["anomaly_label_distribution"],
        "split_summary": ann_profile["split_summary"],
        "duration_distribution": aud_profile["duration_distribution"],
        "sample_rate_distribution": aud_profile["sample_rate_distribution"],
        "channel_distribution": aud_profile["channel_distribution"],
        "format_distribution": aud_profile["file_format_distribution"],
        "invalid_segment_count": invalid_segments,
        "invalid_pair_count": invalid_pairs,
    }

    dataset = InternalAudioDataset(
        modality="audio",
        input_format=adapter_format_key,
        original_format=adapter_original,
        samples=samples,
        label_mapping=label_mapping,
        speaker_mapping=speaker_mapping,
        structure_profile=structure_profile,
        parsing_summary=parsing_summary,
        metadata_profile=metadata_profile,
        audio_profile=aud_profile,
        annotation_or_reference_profile=ann_profile,
        warnings=warnings,
        dataset_root=root,
        raw_root=root,
    )
    return AdapterResult(ok=True, data={"internal_dataset": dataset})


def _missing_ratio(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = sum(len(r) for r in rows)
    if total == 0:
        return 0.0
    missing = 0
    for row in rows:
        for v in row.values():
            if v in (None, "") or (isinstance(v, str) and not v.strip()):
                missing += 1
    return float(missing) / float(total)


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None
