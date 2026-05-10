from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.ingestion.base_adapter import AdapterResult, BaseFormatAdapter
from src.utils.ingestion.text.internal import (
    InternalTextDataset,
    SUPPORTED_CONLL_EXTS,
    SUPPORTED_METADATA_EXTS,
    SUPPORTED_TXT_EXTS,
    collect_files,
    extract_zip,
    find_dataset_root,
    find_metadata_files,
    normalize_for_storage,
    parse_json_text_records,
    read_text_file,
    safe_relative_path,
)


def _parse_conll_file(path: Path) -> Tuple[List[List[str]], List[List[str]], Optional[str]]:
    tokens_per_sentence: List[List[str]] = []
    tags_per_sentence: List[List[str]] = []
    bio_seen = False
    pos_seen = False
    current_tokens: List[str] = []
    current_tags: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    if current_tokens:
                        tokens_per_sentence.append(current_tokens)
                        tags_per_sentence.append(current_tags)
                        current_tokens = []
                        current_tags = []
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                token = parts[0]
                tag = parts[-1]
                current_tokens.append(token)
                current_tags.append(tag)
                if any(tag.startswith(p) for p in ("B-", "I-", "O")) or tag == "O":
                    bio_seen = True
                if tag.isupper() and len(tag) <= 5 and not any(c in tag for c in "-_"):
                    pos_seen = True
    except Exception:
        return [], [], None
    if current_tokens:
        tokens_per_sentence.append(current_tokens)
        tags_per_sentence.append(current_tags)
    if not tokens_per_sentence:
        return [], [], None
    if bio_seen:
        kind = "ner"
    elif pos_seen:
        kind = "pos"
    else:
        kind = "ner"
    return tokens_per_sentence, tags_per_sentence, kind


def _bio_tags_to_entities(tokens: List[str], tags: List[str]) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    pos = 0
    i = 0
    starts: List[int] = []
    char_pos = 0
    for tok in tokens:
        starts.append(char_pos)
        char_pos += len(tok) + 1
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start = starts[i]
            end = starts[i] + len(tokens[i])
            j = i + 1
            while j < len(tokens) and tags[j] == f"I-{label}":
                end = starts[j] + len(tokens[j])
                j += 1
            entities.append({"start": start, "end": end, "label": label})
            i = j
        else:
            i += 1
    pos = pos
    return entities


class TxtZipTextAdapter(BaseFormatAdapter):
    modality = "Text"
    input_format = "TXT document folder / ZIP"
    is_implemented = True
    format_key = "txt_zip"

    def validate_input(self, path: Path) -> AdapterResult:
        errors: List[str] = []
        if not path.exists():
            errors.append(f"File does not exist: {path}")
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        if path.suffix.lower() != ".zip":
            errors.append("TXT document folder / ZIP requires a .zip archive.")
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
        has_text = any(Path(n).suffix.lower() in SUPPORTED_TXT_EXTS for n in names)
        has_metadata = any(Path(n).suffix.lower() in SUPPORTED_METADATA_EXTS for n in names)
        has_conll = any(Path(n).suffix.lower() in SUPPORTED_CONLL_EXTS for n in names)
        if not has_text and not has_metadata and not has_conll:
            errors.append(
                "Zip archive contains no supported text documents, metadata files, or CoNLL/BIO files. "
                f"Supported text extensions: {', '.join(sorted(SUPPORTED_TXT_EXTS))}; "
                f"CoNLL extensions: {', '.join(sorted(SUPPORTED_CONLL_EXTS))}."
            )
        if errors:
            return AdapterResult(ok=False, message=errors[0], errors=errors)
        return AdapterResult(ok=True)

    def to_internal_dataset(self, path: Path, work_dir: Optional[Path] = None, metadata_path: str = "", record_path: str = "", task_type: str = "", **kwargs) -> AdapterResult:
        validation = self.validate_input(path)
        if not validation.ok:
            return validation
        if work_dir is None:
            return AdapterResult(ok=False, message="work_dir is required for TXT ZIP ingestion.", errors=["work_dir is required."])
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            warnings = extract_zip(path, work_dir)
        except Exception as exc:
            return AdapterResult(ok=False, message=f"Failed to extract zip: {exc}", errors=[str(exc)])
        root = find_dataset_root(work_dir)

        text_files = collect_files(root, SUPPORTED_TXT_EXTS)
        metadata_files = find_metadata_files(root, SUPPORTED_METADATA_EXTS)
        conll_files = collect_files(root, SUPPORTED_CONLL_EXTS)

        chosen_metadata = self._choose_metadata(root, metadata_files, metadata_path)

        unsupported = 0
        for p in collect_files(root, None):
            ext = p.suffix.lower()
            if not ext:
                continue
            if ext in SUPPORTED_TXT_EXTS or ext in SUPPORTED_METADATA_EXTS:
                continue
            if ext in {".yaml", ".yml", ".md"}:
                continue
            unsupported += 1

        if isinstance(chosen_metadata, dict) and chosen_metadata.get("error"):
            return AdapterResult(ok=False, message=chosen_metadata["error"], errors=[chosen_metadata["error"]])

        if chosen_metadata is not None:
            df, mode_summary, parser_warnings, struct_extra = self._build_with_metadata(root, chosen_metadata, text_files, record_path, warnings)
            warnings = warnings + parser_warnings
        elif conll_files and (task_type or "").lower() in {"", "ner", "pos"}:
            df, mode_summary, parser_warnings, struct_extra = self._build_with_conll(root, conll_files, task_type)
            warnings = warnings + parser_warnings
        else:
            class_dirs = self._detect_class_dirs(root, text_files)
            if class_dirs:
                df, mode_summary, parser_warnings, struct_extra = self._build_class_folder(root, text_files, class_dirs)
            else:
                df, mode_summary, parser_warnings, struct_extra = self._build_plain_corpus(root, text_files)
            warnings = warnings + parser_warnings

        if df is None or df.empty:
            return AdapterResult(ok=False, message="No usable text documents were found inside the zip.", errors=["empty parsed dataset"])

        labels = df["label"] if "label" in df.columns else pd.Series(dtype=str)
        label_counter = Counter(str(v) for v in labels.fillna("").tolist() if str(v))

        unreadable = int(struct_extra.get("unreadable_document_count", 0))
        missing_doc = int(struct_extra.get("missing_document_count", 0))
        ext_dist = struct_extra.get("file_extension_distribution", {})

        structure_profile: Dict[str, Any] = {
            "input_format": self.format_key,
            "structure_type": mode_summary.get("structure_type", ""),
            "folder_label_mode": mode_summary.get("structure_type", "") == "class_folder",
            "metadata_file_used": str(chosen_metadata.relative_to(root)) if isinstance(chosen_metadata, Path) and chosen_metadata.exists() else "",
            "document_count": int(len(df)),
            "file_extension_distribution": dict(ext_dist),
            "class_dirs": sorted(label_counter.keys()),
            "columns": [str(c) for c in df.columns],
            "original_record_count": int(struct_extra.get("original_record_count", len(text_files))),
            "parsed_record_count": int(len(df)),
            "schema_variant_count": 1,
            "nested_field_count": 0,
            "flattened_metadata_fields": [],
            "max_nesting_depth": 1,
        }
        parsing_summary: Dict[str, Any] = {
            "input_format": self.format_key,
            "source_format": "txt_document_folder_zip",
            "conversion_strategy": mode_summary.get("conversion_strategy", "txt_to_dataframe"),
            "discovered_text_files": int(len(text_files)),
            "rows_read": int(len(df)),
            "columns_read": int(len(df.columns)),
            "unreadable_document_count": unreadable,
            "missing_document_count": missing_doc,
            "unsupported_document_count": int(unsupported),
            "metadata_file_used": str(chosen_metadata.relative_to(root)) if isinstance(chosen_metadata, Path) and chosen_metadata.exists() else "",
            "extracted_root": str(root),
        }

        dataset = InternalTextDataset(
            modality="text",
            input_format=self.format_key,
            original_format="text_txt_document_zip",
            dataframe=df,
            structure_profile=structure_profile,
            parsing_summary=parsing_summary,
            field_mapping={c: c for c in df.columns},
            warnings=list(warnings or []),
            dataset_root=root,
        )
        return AdapterResult(ok=True, data={"internal_dataset": dataset})

    def _choose_metadata(self, root: Path, metadata_files: List[Path], metadata_path: str):
        if metadata_path:
            safe = safe_relative_path(metadata_path)
            if safe is None:
                return {"error": f"Invalid metadata path: '{metadata_path}'."}
            candidate = root / safe
            if not candidate.exists():
                leaf = Path(safe).name
                matches = [m for m in metadata_files if m.name == leaf]
                if matches:
                    candidate = matches[0]
                else:
                    return {"error": f"Metadata file '{metadata_path}' not found inside the zip archive."}
            return candidate
        if not metadata_files:
            return None
        if len(metadata_files) == 1:
            return metadata_files[0]
        names = ", ".join(sorted({m.relative_to(root).as_posix() for m in metadata_files})[:5])
        return {"error": f"Multiple metadata files found ({names}). Provide the metadata file path/name to disambiguate."}

    def _detect_class_dirs(self, root: Path, text_files: List[Path]) -> List[str]:
        labels: Counter = Counter()
        any_nested = False
        for p in text_files:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) >= 2:
                labels[parts[0]] += 1
                any_nested = True
        if not any_nested:
            return []
        return sorted(labels.keys())

    def _build_with_conll(self, root: Path, conll_files: List[Path], task_type: str) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        ext_counter: Counter = Counter()
        warnings: List[str] = []
        detected_kinds: Counter = Counter()
        for path in conll_files:
            tokens_per_sent, tags_per_sent, kind = _parse_conll_file(path)
            ext_counter[path.suffix.lower().lstrip(".")] += 1
            if kind:
                detected_kinds[kind] += 1
            if not tokens_per_sent:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            for idx, (tokens, tags) in enumerate(zip(tokens_per_sent, tags_per_sent)):
                text = " ".join(tokens)
                if not text.strip():
                    continue
                row: Dict[str, Any] = {
                    "document_id": f"{rel}#s{idx}",
                    "document_path": rel,
                    "text": text,
                    "tokens": json.dumps(list(tokens)),
                }
                if kind == "ner":
                    row["entities"] = json.dumps(_bio_tags_to_entities(tokens, tags))
                else:
                    row["pos_tags"] = json.dumps(list(tags))
                rows.append(row)
        if not rows:
            warnings.append("CoNLL/BIO files found but no sentences were parsed.")
            return pd.DataFrame(), {"structure_type": "conll", "conversion_strategy": "conll_to_dataframe"}, warnings, {
                "original_record_count": len(conll_files), "unreadable_document_count": 0,
                "missing_document_count": 0, "file_extension_distribution": dict(ext_counter),
            }
        cols = ["document_id", "document_path", "text", "tokens"]
        if any("entities" in r for r in rows):
            cols.append("entities")
        if any("pos_tags" in r for r in rows):
            cols.append("pos_tags")
        df = pd.DataFrame(rows, columns=cols)
        majority = detected_kinds.most_common(1)[0][0] if detected_kinds else "ner"
        struct_extra = {
            "original_record_count": len(conll_files),
            "unreadable_document_count": 0,
            "missing_document_count": 0,
            "file_extension_distribution": dict(ext_counter),
            "conll_detected_kind": majority,
            "conll_sentence_count": len(rows),
        }
        mode_summary = {
            "structure_type": "conll",
            "conversion_strategy": f"conll_{majority}_to_dataframe",
        }
        return df, mode_summary, warnings, struct_extra


    def _build_class_folder(self, root: Path, text_files: List[Path], class_dirs: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        ext_counter: Counter = Counter()
        unreadable = 0
        warnings: List[str] = []
        for p in text_files:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            parts = rel.parts
            if len(parts) < 2:
                continue
            label = parts[0]
            text, status = read_text_file(p)
            ext_counter[p.suffix.lower().lstrip(".")] += 1
            if status != "ok":
                unreadable += 1
                if len(warnings) < 5:
                    warnings.append(f"unreadable text file: {rel.as_posix()}")
                continue
            text = normalize_for_storage(text)
            if not text.strip():
                continue
            rows.append({
                "document_id": rel.as_posix(),
                "document_path": rel.as_posix(),
                "text": text,
                "label": label,
            })
        df = pd.DataFrame(rows, columns=["document_id", "document_path", "text", "label"]) if rows else pd.DataFrame(columns=["document_id", "document_path", "text", "label"])
        struct_extra = {
            "original_record_count": len(text_files),
            "unreadable_document_count": unreadable,
            "missing_document_count": 0,
            "file_extension_distribution": dict(ext_counter),
        }
        return df, {"structure_type": "class_folder", "conversion_strategy": "folder_label_to_dataframe"}, warnings, struct_extra

    def _build_plain_corpus(self, root: Path, text_files: List[Path]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        ext_counter: Counter = Counter()
        unreadable = 0
        warnings: List[str] = []
        for p in text_files:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            text, status = read_text_file(p)
            ext_counter[p.suffix.lower().lstrip(".")] += 1
            if status != "ok":
                unreadable += 1
                if len(warnings) < 5:
                    warnings.append(f"unreadable text file: {rel.as_posix()}")
                continue
            text = normalize_for_storage(text)
            if not text.strip():
                continue
            rows.append({
                "document_id": rel.as_posix(),
                "document_path": rel.as_posix(),
                "text": text,
            })
        df = pd.DataFrame(rows, columns=["document_id", "document_path", "text"]) if rows else pd.DataFrame(columns=["document_id", "document_path", "text"])
        struct_extra = {
            "original_record_count": len(text_files),
            "unreadable_document_count": unreadable,
            "missing_document_count": 0,
            "file_extension_distribution": dict(ext_counter),
        }
        return df, {"structure_type": "plain_corpus", "conversion_strategy": "txt_to_dataframe"}, warnings, struct_extra

    def _build_with_metadata(self, root: Path, metadata_file: Path, text_files: List[Path], record_path: str, base_warnings: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any], List[str], Dict[str, Any]]:
        warnings: List[str] = []
        ext = metadata_file.suffix.lower()
        if ext == ".csv":
            try:
                meta_df = pd.read_csv(metadata_file)
            except Exception as exc:
                return None, {"structure_type": "metadata_csv"}, [f"Failed to read metadata CSV: {exc}"], {"original_record_count": 0, "unreadable_document_count": 0, "missing_document_count": 0, "file_extension_distribution": {}}
        elif ext in {".json", ".jsonl", ".ndjson"}:
            ok, err, meta_df, _structure, _ps, parser_warnings = parse_json_text_records(metadata_file, record_path=record_path)
            warnings.extend(parser_warnings or [])
            if not ok or meta_df is None:
                return None, {"structure_type": "metadata_json"}, [err or "Failed to parse metadata JSON."], {"original_record_count": 0, "unreadable_document_count": 0, "missing_document_count": 0, "file_extension_distribution": {}}
        else:
            return None, {"structure_type": "metadata_unknown"}, [f"Unsupported metadata file extension: {ext}"], {"original_record_count": 0, "unreadable_document_count": 0, "missing_document_count": 0, "file_extension_distribution": {}}

        if meta_df.empty:
            return None, {"structure_type": "metadata_empty"}, ["Metadata file contains no records."], {"original_record_count": 0, "unreadable_document_count": 0, "missing_document_count": 0, "file_extension_distribution": {}}

        path_col = self._detect_path_column(meta_df.columns)

        text_index = self._build_text_index(root, text_files)
        rows: List[Dict[str, Any]] = []
        unreadable = 0
        missing = 0
        ext_counter: Counter = Counter()

        if path_col is not None:
            for _, row in meta_df.iterrows():
                rel_raw = row.get(path_col)
                rel_safe = safe_relative_path(str(rel_raw) if rel_raw is not None else "")
                if not rel_safe:
                    missing += 1
                    continue
                p = self._resolve_doc_path(root, rel_safe, text_index)
                if p is None or not p.exists():
                    missing += 1
                    if len(warnings) < 5:
                        warnings.append(f"metadata references missing file: {rel_safe}")
                    continue
                text, status = read_text_file(p)
                ext_counter[p.suffix.lower().lstrip(".")] += 1
                if status != "ok":
                    unreadable += 1
                    if len(warnings) < 5:
                        warnings.append(f"unreadable text file: {rel_safe}")
                    continue
                text = normalize_for_storage(text)
                rec: Dict[str, Any] = {}
                for col in meta_df.columns:
                    if col == path_col:
                        continue
                    rec[str(col)] = row.get(col)
                try:
                    rel_doc = p.relative_to(root).as_posix()
                except ValueError:
                    rel_doc = p.name
                rec.setdefault("document_id", rel_doc)
                rec["document_path"] = rel_doc
                if "text" not in rec or rec.get("text") in (None, ""):
                    rec["text"] = text
                rows.append(rec)
            df = pd.DataFrame(rows)
            struct = {
                "original_record_count": int(len(meta_df)),
                "unreadable_document_count": unreadable,
                "missing_document_count": missing,
                "file_extension_distribution": dict(ext_counter),
            }
            return df, {"structure_type": "metadata_with_path", "conversion_strategy": "metadata_join_with_files"}, warnings, struct

        text_present = "text" in {str(c).strip().lower() for c in meta_df.columns}
        if text_present:
            df = meta_df.copy()
            df.columns = [str(c) for c in df.columns]
            struct = {
                "original_record_count": int(len(df)),
                "unreadable_document_count": 0,
                "missing_document_count": 0,
                "file_extension_distribution": {},
            }
            return df, {"structure_type": "metadata_inline_text", "conversion_strategy": "metadata_inline_text"}, warnings, struct

        warnings.append("Metadata file does not include a document path column or an inline text column. Falling back to plain corpus.")
        df_plain, mode_summary, parser_warnings, struct_extra = self._build_plain_corpus(root, text_files)
        warnings.extend(parser_warnings)
        return df_plain, {"structure_type": "metadata_no_path_fallback_corpus", "conversion_strategy": mode_summary.get("conversion_strategy", "txt_to_dataframe")}, warnings, struct_extra

    def _detect_path_column(self, columns) -> Optional[str]:
        candidates = ["document_path", "file_path", "filepath", "filename", "file", "path", "document"]
        norm = {str(c).strip().lower(): str(c) for c in columns}
        for name in candidates:
            if name in norm:
                return norm[name]
        return None

    def _build_text_index(self, root: Path, text_files: List[Path]) -> Dict[str, Path]:
        index: Dict[str, Path] = {}
        for p in text_files:
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.name
            index.setdefault(rel, p)
            index.setdefault(rel.lower(), p)
            index.setdefault(p.name, p)
            index.setdefault(p.name.lower(), p)
        return index

    def _resolve_doc_path(self, root: Path, rel_safe: str, text_index: Dict[str, Path]) -> Optional[Path]:
        candidate = root / rel_safe
        if candidate.exists() and candidate.is_file():
            return candidate
        leaf = Path(rel_safe).name
        return text_index.get(rel_safe) or text_index.get(rel_safe.lower()) or text_index.get(leaf) or text_index.get(leaf.lower())
