from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path
from typing import List

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.audio.internal import (
    AudioSample,
    InternalAudioDataset,
    SUPPORTED_AUDIO_EXTS,
    annotation_or_reference_profile,
    audio_profile_summary,
    collect_audio_files,
    extract_zip,
    find_dataset_root,
    quick_audio_props,
)


_TRANSCRIPT_EXTS = {".txt", ".srt", ".vtt", ".lab"}
_TRANSCRIPT_STEM_SUFFIXES = ("_text", "_transcript", "_transcription", "_asr")


def _strip_transcript_suffix(stem: str) -> str:
    for suf in _TRANSCRIPT_STEM_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _read_transcript_text(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    suf = path.suffix.lower()
    if suf in {".srt", ".vtt"}:
        out_lines: List[str] = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.isdigit():
                continue
            if "-->" in stripped:
                continue
            if stripped.upper().startswith("WEBVTT"):
                continue
            out_lines.append(stripped)
        return " ".join(out_lines).strip()
    return raw.strip()


def _build_transcript_map(root: Path) -> dict:
    mapping: dict = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TRANSCRIPT_EXTS:
            continue
        text = _read_transcript_text(path)
        if not text:
            continue
        stem = _strip_transcript_suffix(path.stem)
        mapping.setdefault((path.parent, stem), text)
    return mapping


def _build_clean_pair_map(root: Path, audio_paths: List[Path]) -> dict:
    by_dir: dict = {}
    for ap in audio_paths:
        try:
            rel = ap.relative_to(root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        top = rel.parts[0].lower()
        by_dir.setdefault(top, []).append((ap, rel))
    if "noisy" not in by_dir or "clean" not in by_dir:
        return {}
    clean_index: dict = {}
    for ap, rel in by_dir["clean"]:
        sub = Path(*rel.parts[1:])
        clean_index[(sub.parent.as_posix(), sub.stem.lower())] = ap
    pairs: dict = {}
    for ap, rel in by_dir["noisy"]:
        sub = Path(*rel.parts[1:])
        match = clean_index.get((sub.parent.as_posix(), sub.stem.lower()))
        if match is not None:
            pairs[ap] = match
    return pairs


class ZipFolderAudioAdapter(BaseFormatAdapter):
    modality = "Audio"
    input_format = "Audio folder / ZIP"
    is_implemented = True
    format_key = "zip_folder"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("Audio folder / ZIP requires a .zip archive.")
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
            errors.append(
                "Zip archive contains no supported audio files. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTS))}."
            )
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(self, path: Path, work_dir: Path = None, **kwargs) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        if work_dir is None:
            return AdapterResult(ok=False, message="work_dir is required for audio ingestion.", errors=["work_dir is required."])
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        warnings = extract_zip(path, work_dir)
        root = find_dataset_root(work_dir)

        audio_paths = collect_audio_files(root)
        corrupted = 0
        unsupported = 0
        all_files = [p for p in root.rglob("*") if p.is_file() and not any(part.startswith(".") or part == "__MACOSX" for part in p.parts)]
        for p in all_files:
            if p.suffix.lower() and p.suffix.lower() not in SUPPORTED_AUDIO_EXTS and p.suffix.lower() not in {".csv", ".json", ".jsonl", ".ndjson", ".txt", ".rttm", ".lab", ".textgrid", ".yaml", ".yml", ".md", ".srt", ".vtt"}:
                unsupported += 1

        transcript_map = _build_transcript_map(root)
        clean_pair_map = _build_clean_pair_map(root, audio_paths)

        samples: List[AudioSample] = []
        seen_names: Counter = Counter()
        format_counter: Counter = Counter()
        for audio_path in audio_paths:
            duration, sr, ch, status = quick_audio_props(audio_path)
            if status != "ok":
                corrupted += 1
                continue
            try:
                rel = audio_path.relative_to(root)
                parts = rel.parts
                label = parts[0] if len(parts) > 1 else ""
            except ValueError:
                rel = audio_path
                parts = (audio_path.name,)
                label = ""
            seen_names[audio_path.name] += 1
            sample_id = audio_path.stem if seen_names[audio_path.name] == 1 else f"{audio_path.stem}_{seen_names[audio_path.name]}"
            format_counter[audio_path.suffix.lower().lstrip(".")] += 1
            transcript = transcript_map.get((audio_path.parent, audio_path.stem), "")
            noisy_path = ""
            clean_path = ""
            pair_label = label
            paired_clean = clean_pair_map.get(audio_path)
            if paired_clean is not None:
                noisy_path = str(audio_path)
                clean_path = str(paired_clean)
                pair_label = "noisy"
            samples.append(
                AudioSample(
                    sample_id=sample_id,
                    audio_path=str(audio_path),
                    duration=duration,
                    sample_rate=sr,
                    channels=ch,
                    label=pair_label,
                    transcript=transcript,
                    noisy_path=noisy_path,
                    clean_path=clean_path,
                )
            )

        class_dirs = sorted({s.label for s in samples if s.label})
        label_mapping = {idx: name for idx, name in enumerate(class_dirs)}

        parsing_summary = {
            "input_format": self.format_key,
            "source_format": "audio_folder_zip",
            "conversion_strategy": "passthrough",
            "discovered_audio_files": len(audio_paths),
            "parsed_audio_count": len(samples),
            "corrupted_audio_count": corrupted,
            "unsupported_audio_count": unsupported,
            "duplicate_audio_count": sum(1 for c in seen_names.values() if c > 1),
            "extracted_root": str(root),
        }
        structure_profile = {
            "input_format": self.format_key,
            "structure_type": "class_folder" if class_dirs else "flat",
            "class_dirs": class_dirs,
            "depth": "two_level" if class_dirs else "flat",
        }

        ann_profile = annotation_or_reference_profile(samples)
        aud_profile = audio_profile_summary(samples, dict(format_counter))
        metadata_profile = {
            "input_format": self.format_key,
            "metadata_record_count": 0,
            "missing_audio_count": 0,
            "unused_audio_count": 0,
            "duplicate_metadata_row_count": 0,
            "metadata_field_count": 0,
            "metadata_missing_ratio": 0.0,
            "parsing_warnings": list(warnings),
            "original_audio_file_count": len(audio_paths),
            "parsed_audio_count": len(samples),
            "corrupted_audio_count": corrupted,
            "unsupported_audio_count": unsupported,
            "duplicate_audio_count": sum(1 for c in seen_names.values() if c > 1),
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
            "invalid_segment_count": 0,
            "invalid_pair_count": 0,
        }

        dataset = InternalAudioDataset(
            modality="audio",
            input_format=self.format_key,
            original_format="audio_folder_zip",
            samples=samples,
            label_mapping=label_mapping,
            speaker_mapping={},
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
