from __future__ import annotations

import customtkinter as ctk

from ui.constants import (
    BG_WIN, BG_BAR, BG_INPUT,
    ACCENT, ACCENT_H, BORDER, ERROR,
    TXT, TXT_MUTED,
    FONT_FAMILY,
)
from ui.helpers import _sec_label, _card, _hsep

_SENTINEL = "— select —"

_MODALITY_OPTS = [
    "CSV / Tabular",
    "Image",
    "Audio",
    "Text",
    "Semi-structured",
    "Unstructured",
]

_CSV_SUPERVISED_TASKS = [
    "Binary classification",
    "Multiclass classification",
    "Multi-label classification",
    "Regression",
    "Ordinal regression",
    "Ranking / learning-to-rank",
    "Time-series forecasting",
]
_CSV_UNSUPERVISED_TASKS = [
    "Clustering",
    "Anomaly / outlier detection",
    "Dimensionality reduction",
    "Association rule mining",
]
_CSV_TASK_OPTS = _CSV_SUPERVISED_TASKS + _CSV_UNSUPERVISED_TASKS
_CSV_SUPERVISED_SET = set(_CSV_SUPERVISED_TASKS)

_CSV_TASK_BACKEND = {
    "Binary classification":       "binary",
    "Multiclass classification":   "multiclass",
    "Multi-label classification":  "multiclass",
    "Regression":                  "regression",
    "Ordinal regression":          "regression",
    "Ranking / learning-to-rank":  "classification",
    "Time-series forecasting":     "regression",
    "Clustering":                  "other",
    "Anomaly / outlier detection": "other",
    "Dimensionality reduction":    "other",
    "Association rule mining":     "other",
}

_DOMAIN_OPTS = [
    _SENTINEL,
    "Finance / Banking",
    "Healthcare / Medical",
    "E-commerce / Retail",
    "Manufacturing / IoT",
    "HR / People Analytics",
    "Marketing / Advertising",
    "Transportation / Logistics",
    "Education / Academic",
    "Cybersecurity",
    "Energy / Utilities",
    "Legal / Compliance",
    "Science / Research",
    "Telecommunications",
    "Real Estate",
]

_CSV_FE_BUDGET_OPTS = [
    _SENTINEL,
    "Minimal (raw features only)",
    "Light (basic transforms)",
    "Moderate (interactions + encoding)",
    "Heavy (full feature engineering)",
]

_CSV_DATA_QUALITY_OPTS = [
    _SENTINEL,
    "Clean / well-curated",
    "Mostly clean (minor issues)",
    "Noisy / real-world collection",
    "Mixed quality",
    "Unknown",
]


_IMG_TASK_OPTS = [
    "Image classification (single-label)",
    "Image classification (multi-label)",
    "Object detection",
    "Semantic segmentation",
    "Instance segmentation",
    "Keypoint / pose estimation",
    "Image similarity / retrieval",
    "Anomaly / defect detection",
    "Optical character recognition",
    "Image generation / synthesis",
    "Depth estimation",
]

_IMG_FORMAT_OPTS = [
    _SENTINEL,
    "JPEG",
    "PNG",
    "BMP",
    "TIFF",
    "WebP",
    "GIF",
    "DICOM (medical)",
    "RAW (camera)",
    "Mixed",
]

_IMG_COLOR_OPTS = [
    _SENTINEL,
    "RGB",
    "Grayscale",
    "RGBA (with alpha)",
    "BGR (OpenCV default)",
    "HSV",
    "LAB / CIELAB",
    "Mixed",
]

_IMG_DOMAIN_OPTS = [
    _SENTINEL,
    "Medical imaging",
    "Satellite / aerial",
    "Autonomous driving",
    "Quality inspection",
    "Document / OCR",
    "Natural scenes",
    "Security / surveillance",
    "Agriculture",
    "Retail / product",
    "Art / creative",
    "Microscopy / scientific",
]

_AUD_TASK_OPTS = [
    "Audio classification",
    "Speech recognition (ASR)",
    "Speaker identification",
    "Speaker verification",
    "Speaker diarization",
    "Sound event detection",
    "Music genre / mood classification",
    "Emotion recognition from speech",
    "Voice activity detection",
    "Audio anomaly detection",
    "Noise suppression",
]

_AUD_FORMAT_OPTS = [
    _SENTINEL,
    "WAV (uncompressed)",
    "MP3",
    "FLAC (lossless)",
    "OGG",
    "M4A / AAC",
    "Mixed",
]

_AUD_CHANNEL_OPTS = [
    _SENTINEL,
    "Mono",
    "Stereo",
    "Multi-channel",
    "Mixed",
]

_AUD_SR_OPTS = [
    _SENTINEL,
    "8 kHz (telephony)",
    "16 kHz (speech standard)",
    "22.05 kHz",
    "44.1 kHz (CD quality)",
    "48 kHz (broadcast)",
    "Variable / mixed",
    "Keep native",
]

_AUD_DOMAIN_OPTS = [
    _SENTINEL,
    "Speech / conversational",
    "Music",
    "Environmental / ambient",
    "Medical / clinical",
    "Broadcast / media",
    "Industrial / machinery",
    "Security / surveillance",
    "Automotive",
]

_TXT_TASK_OPTS = [
    "Text classification (single-label)",
    "Text classification (multi-label)",
    "Sentiment analysis",
    "Named entity recognition",
    "Part-of-speech tagging",
    "Relation extraction",
    "Intent detection",
    "Semantic similarity / search",
    "Text summarization",
    "Machine translation",
    "Question answering",
    "Text generation",
    "Topic modeling",
    "Language detection",
]

_TXT_LANG_OPTS = [
    _SENTINEL,
    "English",
    "Arabic",
    "Chinese (Simplified)",
    "French",
    "German",
    "Japanese",
    "Portuguese",
    "Russian",
    "Spanish",
    "Multilingual",
]

_TXT_SOURCE_OPTS = [
    _SENTINEL,
    "Social media / user-generated",
    "News articles",
    "Scientific / academic papers",
    "Legal / regulatory documents",
    "Medical / clinical notes",
    "Customer support conversations",
    "Books / literature",
    "Web pages",
    "Email / messaging",
    "Code / technical",
    "Financial reports",
]

_TXT_LENGTH_OPTS = [
    _SENTINEL,
    "Micro texts (< 10 words)",
    "Short texts (10–100 words)",
    "Paragraphs (100–500 words)",
    "Documents (> 500 words)",
    "Variable / mixed",
]

_SEMI_FORMAT_OPTS = [
    _SENTINEL,
    "JSON",
    "JSON Lines (JSONL)",
    "XML",
    "YAML",
    "TOML",
    "Apache / NGINX log format",
    "CSV with nested fields",
    "Parquet / Avro / ORC",
    "HTML (structured)",
    "Mixed",
]

_SEMI_TASK_OPTS = [
    "Record / document classification",
    "Field / attribute extraction",
    "Schema inference",
    "Anomaly / schema violation detection",
    "Data normalization / harmonization",
    "Entity resolution / deduplication",
    "Event stream processing",
    "Relational extraction across records",
]

_SEMI_NESTING_OPTS = [
    _SENTINEL,
    "Flat (no nesting)",
    "Shallow (2–3 levels)",
    "Deep (4+ levels)",
    "Variable / mixed",
]

_UNSTRUCT_CONTENT_OPTS = [
    _SENTINEL,
    "PDF documents",
    "Scanned documents (OCR needed)",
    "HTML / web pages",
    "Email threads",
    "PowerPoint / presentations",
    "Word documents",
    "Plain text files",
    "Mixed document types",
]

_UNSTRUCT_TASK_OPTS = [
    "Document classification",
    "Information / entity extraction",
    "Topic discovery",
    "Semantic search / retrieval",
    "Document summarization",
    "Duplicate / near-duplicate detection",
    "Content moderation",
    "Document clustering",
]

_UNSTRUCT_LANG_OPTS = _TXT_LANG_OPTS

_CONSTRAINTS = {
    "CSV / Tabular": [
        ("no_smote",              "No SMOTE / oversampling"),
        ("no_power_transform",    "No power transform"),
        ("no_scaling",            "No feature scaling"),
        ("no_outlier_clip",       "No outlier clipping"),
        ("no_variance_filter",    "No variance filter"),
        ("no_feature_selection",  "No feature selection"),
        ("prefer_simple",         "Prefer simpler pipelines"),
        ("preserve_column_order", "Preserve column order"),
    ],
    "Image": [
        ("no_augmentation",   "No augmentation"),
        ("preserve_aspect",   "Preserve aspect ratio"),
        ("no_normalization",  "No pixel normalization"),
        ("no_resize",         "No resizing"),
        ("grayscale_only",    "Convert to grayscale"),
        ("keep_exif",         "Preserve EXIF metadata"),
        ("no_color_jitter",   "No color jitter"),
        ("no_crop",           "No cropping"),
    ],
    "Audio": [
        ("no_resampling",        "No resampling"),
        ("no_normalization",     "No amplitude normalization"),
        ("no_augmentation",      "No augmentation"),
        ("preserve_duration",    "Preserve original duration"),
        ("mono_only",            "Force mono channel"),
        ("no_noise_reduction",   "No noise reduction"),
        ("no_silence_removal",   "No silence trimming"),
        ("preserve_sample_rate", "Keep native sample rate"),
    ],
    "Text": [
        ("no_stemming",          "No stemming"),
        ("no_stopword_removal",  "No stopword removal"),
        ("preserve_case",        "Preserve case"),
        ("no_truncation",        "No truncation"),
        ("keep_punctuation",     "Preserve punctuation"),
        ("no_lemmatization",     "No lemmatization"),
        ("no_tokenization",      "Custom tokenizer only"),
        ("keep_whitespace",      "Preserve whitespace"),
    ],
    "Semi-structured": [
        ("preserve_structure",   "Preserve field structure"),
        ("no_type_coercion",     "No type coercion"),
        ("keep_null_fields",     "Keep null / missing fields"),
        ("no_schema_inference",  "No schema inference"),
        ("preserve_nesting",     "Preserve nesting depth"),
        ("no_field_rename",      "No field renaming"),
        ("preserve_array_order", "Preserve array ordering"),
        ("no_deduplication",     "No deduplication"),
    ],
    "Unstructured": [
        ("no_ocr",              "No OCR"),
        ("preserve_formatting", "Preserve text formatting"),
        ("keep_metadata",       "Preserve file metadata"),
        ("no_chunking",         "No text chunking"),
        ("no_deduplication",    "No deduplication"),
        ("keep_empty_sections", "Keep empty sections"),
        ("preserve_layout",     "Preserve page layout"),
        ("no_header_footer",    "Strip headers / footers"),
    ],
}

_RESULTS_CONTEXT_LABELS = {
    "task_type":          "Task",
    "domain":             "Domain",
    "fe_budget":          "FE budget",
    "data_quality":       "Data quality",
    "image_format":       "Format",
    "color_space":       "Color space",
    "audio_format":      "Format",
    "channel_layout":    "Channels",
    "sample_rate":       "Sample rate",
    "language":          "Language",
    "text_source":       "Text source",
    "text_length":       "Text length",
    "format_type":       "Format type",
    "nesting_depth":     "Nesting depth",
    "content_type":      "Content type",
}


class RunViewMixin:

    def _build_run_view(self) -> None:
        view = ctk.CTkFrame(self._content, fg_color=BG_WIN, corner_radius=0)
        self._views["run"] = view

        bar = ctk.CTkFrame(view, height=44, fg_color=BG_BAR, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Run  /  Assistant",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TXT).pack(side="left", padx=16)

        footer = ctk.CTkFrame(view, height=54, fg_color=BG_BAR, corner_radius=0)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)

        self._dot = ctk.CTkLabel(footer, text="●", font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                                  text_color=TXT_MUTED)
        self._dot.pack(side="left", padx=(16, 4))
        self._status_lbl = ctk.CTkLabel(footer, text="Ready",
                                         font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                                         text_color=TXT_MUTED)
        self._status_lbl.pack(side="left")
        self._step_lbl = ctk.CTkLabel(footer, text="",
                                       font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                                       text_color=TXT_MUTED)
        self._step_lbl.pack(side="left", padx=(10, 0))

        self._run_btn = ctk.CTkButton(
            footer, text="▶   Run MorphAI",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            height=36, corner_radius=8, width=160,
            fg_color=ACCENT, hover_color=ACCENT_H,
            command=self._on_run,
        )
        self._run_btn.pack(side="right", padx=16)

        scroll = ctk.CTkScrollableFrame(view, fg_color=BG_WIN,
                                        scrollbar_button_color=BORDER)
        scroll.pack(fill="both", expand=True)
        self._run_scroll = scroll

        def _req_label(parent, text, width=140):
            wrap = ctk.CTkFrame(parent, fg_color="transparent")
            wrap.pack(side="left")
            ctk.CTkLabel(wrap, text=text,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=TXT_MUTED, width=width, anchor="w").pack(side="left")
            ctk.CTkLabel(wrap, text=" *",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=ERROR).pack(side="left")

        def _opt_label(parent, text, width=140):
            wrap = ctk.CTkFrame(parent, fg_color="transparent")
            wrap.pack(side="left")
            ctk.CTkLabel(wrap, text=text,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=TXT_MUTED, width=width, anchor="w").pack(side="left")
            ctk.CTkLabel(wrap, text=" optional",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                         text_color=TXT_MUTED).pack(side="left")

        _sec_label(scroll, "DATASET CONFIGURATION", padx=20, pady=(18, 6))
        cfg_card = _card(scroll)
        cfg_card.pack(fill="x", padx=20)


        leg = ctk.CTkFrame(cfg_card, fg_color="transparent")
        leg.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(leg, text="*",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                     text_color=ERROR).pack(side="left")
        ctk.CTkLabel(leg, text="  Required field",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                     text_color=TXT_MUTED).pack(side="left")

        r0 = ctk.CTkFrame(cfg_card, fg_color="transparent")
        r0.pack(fill="x", padx=14, pady=(4, 6))
        _req_label(r0, "Data type")
        self._modality_var = ctk.StringVar(value="CSV / Tabular")
        self._modality_menu = ctk.CTkOptionMenu(
            r0, values=_MODALITY_OPTS,
            variable=self._modality_var,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=BG_BAR, text_color=TXT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=180,
            dynamic_resizing=False,
            command=self._on_modality_change,
        )
        self._modality_menu.pack(side="left", padx=(6, 0))

        r1 = ctk.CTkFrame(cfg_card, fg_color="transparent")
        r1.pack(fill="x", padx=14, pady=(4, 6))
        _req_label(r1, "Dataset")
        self._file_lbl = ctk.CTkLabel(
            r1, text="No file selected",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED, anchor="w",
        )
        self._file_lbl.pack(side="left", fill="x", expand=True)
        self._browse_btn = ctk.CTkButton(
            r1, text="Browse", width=80, height=30,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TXT,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=self._on_browse,
        )
        self._browse_btn.pack(side="right")

        self._csv_task_frame = ctk.CTkFrame(cfg_card, fg_color="transparent")
        self._csv_task_frame.pack(fill="x")

        r_task = ctk.CTkFrame(self._csv_task_frame, fg_color="transparent")
        r_task.pack(fill="x", padx=14, pady=(4, 6))
        _req_label(r_task, "Task type")
        self._csv_task_var = ctk.StringVar(value="Binary classification")
        self._csv_task_menu = ctk.CTkOptionMenu(
            r_task, values=_CSV_TASK_OPTS,
            variable=self._csv_task_var,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=BG_BAR, text_color=TXT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=260,
            dynamic_resizing=False,
            command=self._on_csv_task_change,
        )
        self._csv_task_menu.pack(side="left", padx=(6, 0))

        self._csv_fields_frame = ctk.CTkFrame(cfg_card, fg_color="transparent")
        self._csv_fields_frame.pack(fill="x", pady=(0, 12))

        r2 = ctk.CTkFrame(self._csv_fields_frame, fg_color="transparent")
        r2.pack(fill="x", padx=14, pady=6)
        _req_label(r2, "Target column")
        self._target = ctk.CTkEntry(
            r2, placeholder_text="e.g.  label  /  class  /  target",
            fg_color=BG_INPUT, border_color=BORDER,
            text_color=TXT, placeholder_text_color=TXT_MUTED,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
        )
        self._target.pack(side="left", fill="x", expand=True, padx=(6, 0))

        r3 = ctk.CTkFrame(self._csv_fields_frame, fg_color="transparent")
        r3.pack(fill="x", padx=14, pady=(6, 14))
        _opt_label(r3, "Priority metric")
        self._metric_var = ctk.StringVar(value="f1")
        self._metric_menu = ctk.CTkOptionMenu(
            r3, values=["f1", "accuracy", "precision", "recall"],
            variable=self._metric_var,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=BG_BAR, text_color=TXT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=140,
        )
        self._metric_menu.pack(side="left", padx=(6, 0))

        self._noncsv_spacer = ctk.CTkFrame(cfg_card, height=14, fg_color="transparent")

        self._modality_section = ctk.CTkFrame(scroll, fg_color="transparent")
        self._modality_section.pack(fill="x")

        self._modality_menus:        list = []
        self._modality_checks:       list = []
        self._modality_context_vars: dict = {}
        self._constraint_vars:       dict = {}

        self._build_modality_section("CSV / Tabular")

        _sec_label(scroll, "NOTES", padx=20, pady=(14, 6))
        notes_card = _card(scroll)
        notes_card.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkLabel(notes_card,
                     text="Optional — add a brief note about this run.",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(10, 4))
        self._notes = ctk.CTkTextbox(
            notes_card,
            height=50, fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        )
        self._notes.pack(fill="x", padx=14, pady=(0, 12))

        self._results_frame = ctk.CTkFrame(scroll, fg_color="transparent")

    def _on_modality_change(self, modality: str) -> None:
        self._build_modality_section(modality)
        self._noncsv_spacer.pack_forget()
        self._csv_task_frame.pack_forget()
        self._csv_fields_frame.pack_forget()

        if modality == "CSV / Tabular":
            if self._csv_task_var.get() in _CSV_SUPERVISED_SET:
                self._csv_task_frame.pack(fill="x")
                self._csv_fields_frame.pack(fill="x", pady=(0, 12))
            else:
                self._csv_task_frame.pack(fill="x", pady=(0, 12))
        else:
            self._noncsv_spacer.pack(fill="x", pady=(0, 12))

    def _on_csv_task_change(self, task: str) -> None:
        self._csv_task_frame.pack_forget()
        self._csv_fields_frame.pack_forget()

        if task in _CSV_SUPERVISED_SET:
            self._csv_task_frame.pack(fill="x")
            self._csv_fields_frame.pack(fill="x", pady=(0, 12))
        else:
            self._csv_task_frame.pack(fill="x", pady=(0, 12))

    def _build_modality_section(self, modality: str) -> None:
        for widget in self._modality_section.winfo_children():
            widget.destroy()

        self._modality_menus        = []
        self._modality_checks       = []
        self._modality_context_vars = {}
        self._constraint_vars       = {}

        _sec_label(self._modality_section, "TASK  &  CONTEXT", padx=20, pady=(16, 6))
        ctx_card = _card(self._modality_section)
        ctx_card.pack(fill="x", padx=20)

        ctk.CTkLabel(ctx_card,
                     text="All fields optional — fill in what you know to guide the agent.",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(10, 6))

        def _ctx_row(parent, label, var, opts, width=260, command=None):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=4)
            ctk.CTkLabel(row, text=label,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=TXT_MUTED, width=140, anchor="w").pack(side="left")
            menu = ctk.CTkOptionMenu(
                row, values=opts, variable=var,
                fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
                dropdown_fg_color=BG_BAR, text_color=TXT,
                font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=width,
                dynamic_resizing=False,
                command=command,
            )
            menu.pack(side="left", padx=(6, 0))
            self._modality_menus.append(menu)
            return var

        if modality == "CSV / Tabular":
            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

            fe_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Feature engineering budget", fe_var, _CSV_FE_BUDGET_OPTS, width=260)
            self._modality_context_vars["fe_budget"] = fe_var

            quality_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Data quality", quality_var, _CSV_DATA_QUALITY_OPTS, width=240)
            self._modality_context_vars["data_quality"] = quality_var

        elif modality == "Image":
            task_var = ctk.StringVar(value=_IMG_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _IMG_TASK_OPTS, width=280)
            self._modality_context_vars["task_type"] = task_var

            fmt_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Input format", fmt_var, _IMG_FORMAT_OPTS, width=200)
            self._modality_context_vars["image_format"] = fmt_var

            color_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Color space", color_var, _IMG_COLOR_OPTS, width=200)
            self._modality_context_vars["color_space"] = color_var

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _IMG_DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

        elif modality == "Audio":
            task_var = ctk.StringVar(value=_AUD_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _AUD_TASK_OPTS, width=280)
            self._modality_context_vars["task_type"] = task_var

            fmt_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Input format", fmt_var, _AUD_FORMAT_OPTS, width=200)
            self._modality_context_vars["audio_format"] = fmt_var

            channel_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Channel layout", channel_var, _AUD_CHANNEL_OPTS, width=180)
            self._modality_context_vars["channel_layout"] = channel_var

            sr_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Sample rate", sr_var, _AUD_SR_OPTS, width=240)
            self._modality_context_vars["sample_rate"] = sr_var

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _AUD_DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

        elif modality == "Text":
            task_var = ctk.StringVar(value=_TXT_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _TXT_TASK_OPTS, width=280)
            self._modality_context_vars["task_type"] = task_var

            lang_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Language", lang_var, _TXT_LANG_OPTS, width=200)
            self._modality_context_vars["language"] = lang_var

            source_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Text source", source_var, _TXT_SOURCE_OPTS, width=260)
            self._modality_context_vars["text_source"] = source_var

            len_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Text length", len_var, _TXT_LENGTH_OPTS, width=260)
            self._modality_context_vars["text_length"] = len_var

        elif modality == "Semi-structured":
            fmt_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Format type", fmt_var, _SEMI_FORMAT_OPTS, width=240)
            self._modality_context_vars["format_type"] = fmt_var

            task_var = ctk.StringVar(value=_SEMI_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _SEMI_TASK_OPTS, width=280)
            self._modality_context_vars["task_type"] = task_var

            nesting_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Nesting depth", nesting_var, _SEMI_NESTING_OPTS, width=200)
            self._modality_context_vars["nesting_depth"] = nesting_var

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

        elif modality == "Unstructured":
            content_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Content type", content_var, _UNSTRUCT_CONTENT_OPTS, width=240)
            self._modality_context_vars["content_type"] = content_var

            task_var = ctk.StringVar(value=_UNSTRUCT_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _UNSTRUCT_TASK_OPTS, width=280)
            self._modality_context_vars["task_type"] = task_var

            lang_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Language", lang_var, _UNSTRUCT_LANG_OPTS, width=200)
            self._modality_context_vars["language"] = lang_var

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

        ctk.CTkFrame(ctx_card, height=8, fg_color="transparent").pack()

        _sec_label(self._modality_section, "CONSTRAINTS  /  PREFERENCES", padx=20, pady=(14, 6))
        con_card = _card(self._modality_section)
        con_card.pack(fill="x", padx=20)

        ctk.CTkLabel(con_card,
                     text="Optional — select any constraints the agent must respect.",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(10, 6))

        grid = ctk.CTkFrame(con_card, fg_color="transparent")
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for idx, (key, label) in enumerate(_CONSTRAINTS.get(modality, [])):
            var = ctk.BooleanVar(value=False)
            self._constraint_vars[key] = var
            cb = ctk.CTkCheckBox(
                grid, text=label, variable=var,
                font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT,
                fg_color=ACCENT, hover_color=ACCENT_H,
                border_color=BORDER, corner_radius=4,
                checkmark_color="#ffffff",
            )
            cb.grid(row=idx // 2, column=idx % 2, sticky="w", pady=4, padx=(4, 0))
            self._modality_checks.append(cb)

    def _get_context_fields(self) -> dict:
        def _val(var: ctk.StringVar) -> str:
            v = var.get()
            return "" if v == _SENTINEL else v

        modality = self._modality_var.get()
        selected = [k for k, var in self._constraint_vars.items() if var.get()]

        result = {
            "modality":    modality,
            "constraints": ", ".join(selected),
            "notes":       self._notes.get("1.0", "end").strip(),
        }

        for key, var in self._modality_context_vars.items():
            result[key] = _val(var)

        if modality == "CSV / Tabular":
            raw_task = self._csv_task_var.get()
            result["task_type"] = _CSV_TASK_BACKEND.get(raw_task, "classification")

        return result

    def _clear_context_fields(self) -> None:
        self._target.delete(0, "end")
        self._notes.delete("1.0", "end")
        self._metric_var.set("f1")
        # To avoid visual flickering, do not reset modality or task type
        # which would trigger unnecessary UI layout repack operations.
        for var in self._modality_context_vars.values():
            var.set(_SENTINEL)
        for var in self._constraint_vars.values():
            if hasattr(var, "get"):
                var.set("off" if isinstance(var.get(), str) else False)

    def _show_run_results(self, msg: dict) -> None:
        rf = self._results_frame
        for w in rf.winfo_children():
            w.destroy()

        _hsep(rf, pady=(16, 4))
        _sec_label(rf, "RUN RESULTS", padx=20, pady=(8, 8))

        cards_row = ctk.CTkFrame(rf, fg_color="transparent")
        cards_row.pack(fill="x", padx=20)

        c1 = _card(cards_row)
        c1.pack(side="left", fill="both", expand=True, padx=(0, 6))
        _sec_label(c1, "DATASET", padx=14, pady=(10, 4))
        ctk.CTkLabel(c1, text=f"{msg['profile_rows']:,} rows",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"),
                     text_color=TXT).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=f"{msg['profile_cols']} features",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1,
                     text=f"{msg['num_cols']} numeric  /  {msg['cat_cols']} categorical",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ir    = msg.get("imbalance_ratio", 1.0)
        n_cls = msg.get("n_classes", "?")
        ctk.CTkLabel(c1, text=f"{n_cls} classes  (ratio {ir:.1f}x)",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(0, 12))

        c2 = _card(cards_row)
        c2.pack(side="left", fill="both", expand=True, padx=6)
        _sec_label(c2, "BEST PIPELINE", padx=14, pady=(10, 4))
        for part in msg["best_name"].split(" | "):
            ctk.CTkLabel(c2, text=part,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                         anchor="w").pack(anchor="w", padx=14)
        ctk.CTkLabel(c2, text=f"{msg['n_pipelines']} candidates tested",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(6, 2))
        ml_status = msg.get("meta_status", {})
        if ml_status.get("is_mature"):
            ml_txt = f"Meta-learner: active (w={ml_status.get('weight', 0):.2f})"
        else:
            n_tr   = ml_status.get("n_train", 0)
            n_need = ml_status.get("min_to_use", 5)
            ml_txt = f"Meta-learner: learning ({n_tr}/{n_need})"
        ctk.CTkLabel(c2, text=ml_txt,
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(0, 12))

        c3 = _card(cards_row)
        c3.pack(side="left", fill="both", expand=True, padx=(6, 0))
        _sec_label(c3, "METRICS", padx=14, pady=(10, 4))
        pmetric = msg["metric"]
        m       = msg["metrics"]
        for mk in ["f1", "accuracy", "precision", "recall"]:
            is_p = (mk == pmetric)
            mr   = ctk.CTkFrame(c3, fg_color="transparent")
            mr.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(mr,
                         text=mk.upper() if is_p else mk.capitalize(),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED,
                         width=74, anchor="w").pack(side="left")
            ctk.CTkLabel(mr, text=f"{m[mk]:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=TXT if is_p else TXT_MUTED).pack(side="left")
        ctk.CTkLabel(c3,
                     text=f"{msg['n_splits']} folds  x  {msg['n_models']} models",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(6, 12))

        ctx = msg.get("task_context", {})
        if any(v for v in ctx.values()):
            ctx_card = _card(rf)
            ctx_card.pack(fill="x", padx=20, pady=(8, 0))
            _sec_label(ctx_card, "TASK CONTEXT", padx=14, pady=(10, 4))

            modality_val = ctx.get("modality", "")
            if modality_val:
                row = ctk.CTkFrame(ctx_card, fg_color="transparent")
                row.pack(fill="x", padx=14, pady=1)
                ctk.CTkLabel(row, text="Modality:",
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                             text_color=TXT_MUTED, width=90, anchor="w").pack(side="left")
                ctk.CTkLabel(row, text=modality_val,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                             anchor="w").pack(side="left", fill="x")

            for field_key, label in _RESULTS_CONTEXT_LABELS.items():
                val = ctx.get(field_key, "")
                if not val:
                    continue
                row = ctk.CTkFrame(ctx_card, fg_color="transparent")
                row.pack(fill="x", padx=14, pady=1)
                ctk.CTkLabel(row, text=f"{label}:",
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                             text_color=TXT_MUTED, width=90, anchor="w").pack(side="left")
                short_val = val if len(val) <= 80 else val[:77] + "…"
                ctk.CTkLabel(row, text=short_val,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                             anchor="w", wraplength=500).pack(side="left", fill="x")

            for field_key, label in [("constraints", "Constraints"), ("notes", "Notes")]:
                val = ctx.get(field_key, "")
                if not val:
                    continue
                row = ctk.CTkFrame(ctx_card, fg_color="transparent")
                row.pack(fill="x", padx=14, pady=1)
                ctk.CTkLabel(row, text=f"{label}:",
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                             text_color=TXT_MUTED, width=90, anchor="w").pack(side="left")
                short_val = val if len(val) <= 80 else val[:77] + "…"
                ctk.CTkLabel(row, text=short_val,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                             anchor="w", wraplength=500).pack(side="left", fill="x")

            ctk.CTkFrame(ctx_card, height=10, fg_color="transparent").pack()

        act = ctk.CTkFrame(rf, fg_color="transparent")
        act.pack(fill="x", padx=20, pady=(10, 20))

        ctk.CTkButton(
            act, text="View Report", width=130, height=34,
            fg_color=ACCENT, hover_color=ACCENT_H,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=lambda: self._switch_view("report"),
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            act, text="Export PDF", width=110, height=34,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TXT,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=self._on_export_pdf,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            act, text="Open Cleaned CSV", width=150, height=34,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TXT,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=self._open_cleaned,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            act, text="Console", width=90, height=34,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER, text_color=TXT,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=lambda: self._switch_view("console"),
        ).pack(side="left")

        rf.pack(fill="x")
