from __future__ import annotations

import customtkinter as ctk
from src.tabular.config import default_metric_for_task, metric_label, valid_metrics_for_task
from src.image.config import (
    default_metric_for_task as image_default_metric_for_task,
    metric_label as image_metric_label,
    resolve_image_task,
    valid_metrics_for_task as image_valid_metrics_for_task,
)
from src.audio.config import (
    _AUD_TASK_BACKEND as _AUD_TASK_BACKEND_CFG,
    default_metric_for_task as audio_default_metric_for_task,
    metric_label as audio_metric_label,
    valid_metrics_for_task as audio_valid_metrics_for_task,
)
from src.text.config import (
    default_metric_for_task as text_default_metric_for_task,
    metric_label as text_metric_label,
    resolve_text_task,
    valid_metrics_for_task as text_valid_metrics_for_task,
)
from src.utils.ingestion import get_input_formats, get_input_format

from ui.constants import (
    BG_WIN, BG_BAR, BG_INPUT,
    ACCENT, ACCENT_H, BORDER, ERROR, WARN,
    TXT, TXT_MUTED,
    FONT_FAMILY,
)
from ui.helpers import _sec_label, _card, _hsep

_SENTINEL = "— select —"

_MODALITY_OPTS = [
    "Tabular",
    "Image",
    "Audio",
    "Text",
]


def _input_format_labels(modality: str) -> list:
    return [fmt.label for fmt in get_input_formats(modality)]


def _default_input_format_label(modality: str) -> str:
    for fmt in get_input_formats(modality):
        if fmt.implemented:
            return fmt.label
    formats = get_input_formats(modality)
    return formats[0].label if formats else ""

_CSV_SUPERVISED_TASKS = [
    "Binary classification",
    "Multiclass classification",
    "Regression",
    "Time-series forecasting",
]
_CSV_UNSUPERVISED_TASKS = [
    "Clustering",
    "Anomaly / outlier detection",
    "Association rule mining",
]
_CSV_TASK_OPTS = _CSV_SUPERVISED_TASKS + _CSV_UNSUPERVISED_TASKS
_CSV_SUPERVISED_SET = set(_CSV_SUPERVISED_TASKS)

_CSV_TASK_BACKEND = {
    "Binary classification":       "binary",
    "Multiclass classification":   "multiclass",
    "Regression":                  "regression",
    "Time-series forecasting":     "time_series",
    "Clustering":                  "clustering",
    "Anomaly / outlier detection": "anomaly",
    "Association rule mining":     "association_rules",
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
    "Image classification",
    "Object detection",
    "Semantic segmentation",
    "Image similarity / retrieval",
    "Anomaly / defect detection",
    "Optical character recognition",
]

_IMG_CLASSIFICATION_TASK = "Image classification"

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
    "Speaker recognition",
    "Sound event detection",
    "Voice activity detection",
    "Audio anomaly detection",
    "Noise suppression",
]

_AUD_TASK_BACKEND = dict(_AUD_TASK_BACKEND_CFG)

_AUD_FORMAT_OPTS = [
    _SENTINEL,
    "WAV (uncompressed)",
    "MP3",
    "FLAC (lossless)",
    "OGG",
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
    "Text classification",
    "Named entity recognition",
    "Semantic similarity / search",
    "Text summarization",
    "Question answering",
    "Topic modeling",
]

_TXT_CLASSIFICATION_TASK = "Text classification"
_LABEL_MODE_OPTS = ["single-label", "multi-label"]

_TXT_REQUIRED_COL_FIELDS = {
    "classification_single": [
        ("text", "Text column", "Enter text column name"),
        ("label", "Label column", "Enter label column name"),
    ],
    "classification_multi": [
        ("text", "Text column", "Enter text column name"),
    ],
    "ner": [
        ("text", "Text column", "Enter text column name"),
        ("entities", "Entities column (BIO tags or span JSON)", "Enter entities column name"),
    ],
    "semantic_similarity": [
        ("text_a", "Text A column", "Enter text A column name"),
        ("text_b", "Text B column", "Enter text B column name"),
        ("similarity", "Similarity score column", "Enter similarity score column name"),
    ],
    "summarization": [
        ("source_text", "Source text column", "Enter source text column name"),
        ("summary", "Reference summary column", "Enter reference summary column name"),
    ],
    "question_answering": [
        ("context", "Context column", "Enter context column name"),
        ("question", "Question column", "Enter question column name"),
        ("answer", "Answer column", "Enter answer column name"),
    ],
    "topic_modeling": [
        ("text", "Text column", "Enter text column name"),
    ],
}

_TXT_OPTIONAL_COL_FIELDS = {
    "question_answering": [
        ("answer_start", "answer_start column", "Enter answer_start column name (optional)"),
    ],
    "topic_modeling": [
        ("label", "External label column (validation only)", "Enter optional label column name"),
    ],
}

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

_CONSTRAINTS = {
    "Tabular": [
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
}

_RESULTS_CONTEXT_LABELS = {
    "task_name":          "Task",
    "label_mode":         "Label mode",
    "domain":             "Domain",
    "fe_budget":          "FE budget",
    "data_quality":       "Data quality",
    "input_format":       "Input format",
    "image_format":       "Format",
    "color_space":        "Color space",
    "audio_format":       "Format",
    "channel_layout":     "Channels",
    "sample_rate":        "Sample rate",
    "language":           "Language",
    "text_source":        "Text source",
    "text_length":        "Text length",
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
        _req_label(r0, "Modality")
        self._modality_var = ctk.StringVar(value="Tabular")
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

        r0b = ctk.CTkFrame(cfg_card, fg_color="transparent")
        r0b.pack(fill="x", padx=14, pady=(4, 2))
        _req_label(r0b, "Input format")
        initial_modality = self._modality_var.get()
        self._input_format_var = ctk.StringVar(value=_default_input_format_label(initial_modality))
        self._input_format_menu = ctk.CTkOptionMenu(
            r0b, values=_input_format_labels(initial_modality),
            variable=self._input_format_var,
            fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=BG_BAR, text_color=TXT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=280,
            dynamic_resizing=False,
            command=self._on_input_format_change,
        )
        self._input_format_menu.pack(side="left", padx=(6, 0))

        self._input_format_msg = ctk.CTkLabel(
            cfg_card, text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=WARN, anchor="w", justify="left", wraplength=720,
        )
        self._input_format_msg.pack(fill="x", padx=160, pady=(0, 6))

        self._input_format_help = ctk.CTkLabel(
            cfg_card, text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TXT_MUTED, anchor="w", justify="left", wraplength=720,
        )
        self._input_format_help.pack(fill="x", padx=160, pady=(0, 4))

        self._record_path_frame = ctk.CTkFrame(cfg_card, fg_color="transparent")
        rp_row = ctk.CTkFrame(self._record_path_frame, fg_color="transparent")
        rp_row.pack(fill="x", padx=14, pady=(2, 6))
        rp_lbl_wrap = ctk.CTkFrame(rp_row, fg_color="transparent")
        rp_lbl_wrap.pack(side="left")
        self._record_path_label = ctk.CTkLabel(
            rp_lbl_wrap, text="Record path",
            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
            text_color=TXT_MUTED, width=140, anchor="w",
        )
        self._record_path_label.pack(side="left")
        ctk.CTkLabel(rp_lbl_wrap, text=" optional",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                     text_color=TXT_MUTED).pack(side="left")
        self._record_path_var = ctk.StringVar(value="")
        self._record_path_entry = ctk.CTkEntry(
            rp_row, textvariable=self._record_path_var,
            placeholder_text="optional — e.g.  records  |  data.items  |  person",
            fg_color=BG_INPUT, border_color=BORDER,
            text_color=TXT, placeholder_text_color=TXT_MUTED,
            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
        )
        self._record_path_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))

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

        self._cfg_bottom_spacer = ctk.CTkFrame(cfg_card, height=12, fg_color="transparent")
        self._cfg_bottom_spacer.pack(fill="x")

        self._modality_section = ctk.CTkFrame(scroll, fg_color="transparent")
        self._modality_section.pack(fill="x")

        self._modality_menus:        list = []
        self._modality_checks:       list = []
        self._modality_context_vars: dict = {}
        self._constraint_vars:       dict = {}

        self._build_modality_section("Tabular")
        self._apply_input_format_state()

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

        labels = _input_format_labels(modality)
        self._input_format_menu.configure(values=labels or [""])
        self._input_format_var.set(_default_input_format_label(modality))
        self._apply_input_format_state()

    def _on_input_format_change(self, _label: str) -> None:
        self._apply_input_format_state()

    def _apply_input_format_state(self) -> None:
        modality = self._modality_var.get()
        label = self._input_format_var.get()
        fmt = get_input_format(modality, label)
        msg_widget = getattr(self, "_input_format_msg", None)
        run_btn = getattr(self, "_run_btn", None)
        browse_btn = getattr(self, "_browse_btn", None)
        help_widget = getattr(self, "_input_format_help", None)
        rp_frame = getattr(self, "_record_path_frame", None)

        if fmt is None:
            if msg_widget is not None:
                msg_widget.configure(
                    text="Select an input format to continue.",
                    text_color=ERROR,
                )
            if run_btn is not None:
                run_btn.configure(state="disabled")
            if browse_btn is not None:
                browse_btn.configure(state="disabled")
            if help_widget is not None:
                help_widget.configure(text="")
            if rp_frame is not None:
                rp_frame.pack_forget()
            return

        if not fmt.implemented:
            hint = fmt.coming_soon_hint or "Please choose an implemented format."
            text = (
                f"This input format is planned but not implemented yet. {hint}"
            )
            if msg_widget is not None:
                msg_widget.configure(text=text, text_color=WARN)
            if run_btn is not None:
                run_btn.configure(state="disabled")
            if browse_btn is not None:
                browse_btn.configure(state="disabled")
            self._csv_path = None
            file_lbl = getattr(self, "_file_lbl", None)
            if file_lbl is not None:
                file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
            if help_widget is not None:
                help_widget.configure(text="")
            if rp_frame is not None:
                rp_frame.pack_forget()
        else:
            if msg_widget is not None:
                msg_widget.configure(text="")
            if run_btn is not None:
                run_btn.configure(state="normal")
            if browse_btn is not None:
                browse_btn.configure(state="normal")

            help_text = ""
            show_rp = False
            if modality == "Tabular":
                if fmt.key == "csv_excel":
                    help_text = "Upload a CSV or Excel file where rows are samples and columns are features."
                elif fmt.key == "json_records":
                    help_text = (
                        "Upload JSON/JSONL records. The data should contain a list of objects "
                        "or an object containing a records list."
                    )
                    show_rp = True
                elif fmt.key == "xml_records":
                    help_text = (
                        "Upload XML containing repeated record elements. If multiple "
                        "repeated elements exist, enter the record path or element name."
                    )
                    show_rp = True
                elif fmt.key == "yaml_records":
                    help_text = (
                        "Upload YAML containing a list of records or an object containing a records list."
                    )
                    show_rp = True
            elif modality == "Audio":
                if fmt.key == "zip_folder":
                    help_text = "Upload a ZIP containing audio files. For classification, use one folder per class."
                elif fmt.key == "metadata_csv":
                    help_text = (
                        "Upload a ZIP containing audio files and a CSV metadata file. The metadata must "
                        "include audio paths and the required labels/references for the selected task."
                    )
                    show_rp = True
                elif fmt.key == "metadata_json":
                    help_text = (
                        "Upload a ZIP containing audio files and JSON / JSONL metadata records. Each record "
                        "should reference an audio file and include the required labels/references for the "
                        "selected task. Use the record path for nested record lists."
                    )
                    show_rp = True
            elif modality == "Text":
                if fmt.key == "csv_excel":
                    help_text = (
                        "Upload a CSV or Excel file with one sample per row. "
                        "Enter the exact text column names and required target/reference/annotation column names."
                    )
                elif fmt.key == "json_text_records":
                    help_text = (
                        "Upload JSON/JSONL text records. The data should contain a list of objects "
                        "or an object containing a records list."
                    )
                    show_rp = True
                elif fmt.key == "txt_zip":
                    help_text = (
                        "Upload a ZIP containing text documents. For classification, use one folder per "
                        "class or include a metadata file (CSV/JSON) with labels."
                    )
                    show_rp = True
            elif modality == "Image":
                if fmt.key == "zip_folder":
                    help_text = "Upload a ZIP containing image files. For classification, use one folder per class."
                elif fmt.key == "coco":
                    help_text = (
                        "Upload a ZIP containing image files and a COCO annotation JSON file with images, "
                        "annotations, and categories."
                    )
                elif fmt.key == "pascal_voc":
                    help_text = (
                        "Upload a ZIP containing images and Pascal VOC XML annotation files. Each XML "
                        "file should describe objects and bounding boxes for an image."
                    )
                elif fmt.key == "yolo":
                    help_text = (
                        "Upload a ZIP containing images, YOLO label .txt files, and a class config file "
                        "such as data.yaml or classes.txt."
                    )
            if help_widget is not None:
                help_widget.configure(text=help_text)
            if rp_frame is not None:
                if show_rp:
                    rp_frame.pack(fill="x", before=self._cfg_bottom_spacer)
                else:
                    rp_frame.pack_forget()
            rp_label = getattr(self, "_record_path_label", None)
            rp_entry = getattr(self, "_record_path_entry", None)
            if modality == "Text" and fmt.key == "txt_zip":
                if rp_label is not None:
                    rp_label.configure(text="Metadata file")
                if rp_entry is not None:
                    rp_entry.configure(placeholder_text="optional — e.g.  metadata.csv  |  labels/metadata.json")
            else:
                if rp_label is not None:
                    rp_label.configure(text="Record path")
                if rp_entry is not None:
                    rp_entry.configure(placeholder_text="optional — e.g.  records  |  data.items  |  person")

    def _build_modality_section(self, modality: str) -> None:
        for widget in self._modality_section.winfo_children():
            widget.destroy()

        self._modality_menus        = []
        self._modality_checks       = []
        self._modality_context_vars = {}
        self._constraint_vars       = {}
        self._txt_col_entries       = {}
        self._txt_binary_entry      = None
        self._txt_aux_entry         = None
        for _stale in ("_target", "_csv_target_row", "_csv_domain_row", "_metric_menu",
                       "_metric_var", "_csv_task_var", "_image_metric_menu",
                       "_audio_metric_menu", "_text_metric_menu"):
            if hasattr(self, _stale):
                delattr(self, _stale)

        _sec_label(self._modality_section, "TASK  &  CONTEXT", padx=20, pady=(16, 6))
        ctx_card = _card(self._modality_section)
        ctx_card.pack(fill="x", padx=20)

        if modality == "Text":
            _hdr_text = ("Task type and the task-specific column names are required. "
                         "Optional fields can be left blank.")
        elif modality == "Tabular":
            _hdr_text = ("Task type is required; the target column is required for supervised tasks. "
                         "Optional fields can be left blank.")
        else:
            _hdr_text = "Task type is required. Optional fields can be left blank."
        ctk.CTkLabel(ctx_card,
                     text=_hdr_text,
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED,
                     wraplength=780, justify="left").pack(
            anchor="w", padx=14, pady=(10, 6))

        def _ctx_row(parent, label, var, opts, width=260, command=None, required=False):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=4)
            lbl_wrap = ctk.CTkFrame(row, fg_color="transparent")
            lbl_wrap.pack(side="left")
            ctk.CTkLabel(lbl_wrap, text=label,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=TXT_MUTED, width=140, anchor="w").pack(side="left")
            ctk.CTkLabel(lbl_wrap, text="*" if required else "",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=ERROR, width=12, anchor="w").pack(side="left")
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
            return row

        if modality == "Tabular":
            def _on_csv_task_change(task: str) -> None:
                backend_task = _CSV_TASK_BACKEND.get(task, "")
                values = valid_metrics_for_task(backend_task)
                metric_menu = getattr(self, "_metric_menu", None)
                if metric_menu is not None and values:
                    metric_menu.configure(values=values)
                    self._metric_var.set(default_metric_for_task(backend_task))
                target_row = getattr(self, "_csv_target_row", None)
                domain_row = getattr(self, "_csv_domain_row", None)
                if target_row is not None:
                    if task in _CSV_SUPERVISED_SET:
                        if domain_row is not None:
                            target_row.pack(fill="x", padx=14, pady=4, before=domain_row)
                        else:
                            target_row.pack(fill="x", padx=14, pady=4)
                    else:
                        target_row.pack_forget()

            self._csv_task_var = ctk.StringVar(value="Binary classification")
            _ctx_row(ctx_card, "Task type", self._csv_task_var, _CSV_TASK_OPTS,
                     width=260, command=_on_csv_task_change, required=True)

            _csv_backend = _CSV_TASK_BACKEND.get(self._csv_task_var.get(), "")
            self._metric_var = ctk.StringVar(value=default_metric_for_task(_csv_backend))
            _ctx_row(ctx_card, "Priority metric", self._metric_var,
                     valid_metrics_for_task(_csv_backend), width=180)
            self._metric_menu = self._modality_menus[-1]

            target_row = ctk.CTkFrame(ctx_card, fg_color="transparent")
            target_row.pack(fill="x", padx=14, pady=4)
            target_lbl_wrap = ctk.CTkFrame(target_row, fg_color="transparent")
            target_lbl_wrap.pack(side="left")
            ctk.CTkLabel(target_lbl_wrap, text="Target column",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=TXT_MUTED, width=140, anchor="w").pack(side="left")
            ctk.CTkLabel(target_lbl_wrap, text="*",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=ERROR, width=12, anchor="w").pack(side="left")
            self._target = ctk.CTkEntry(
                target_row, placeholder_text="e.g.  label  /  class  /  target",
                fg_color=BG_INPUT, border_color=BORDER,
                text_color=TXT, placeholder_text_color=TXT_MUTED,
                corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
            )
            self._target.pack(side="left", fill="x", expand=True, padx=(6, 0))
            self._csv_target_row = target_row

            domain_var = ctk.StringVar(value=_SENTINEL)
            self._csv_domain_row = _ctx_row(ctx_card, "Domain / use case", domain_var,
                                            _DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

            fe_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Feature engineering budget", fe_var, _CSV_FE_BUDGET_OPTS, width=260)
            self._modality_context_vars["fe_budget"] = fe_var

            quality_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Data quality", quality_var, _CSV_DATA_QUALITY_OPTS, width=240)
            self._modality_context_vars["data_quality"] = quality_var

            if self._csv_task_var.get() not in _CSV_SUPERVISED_SET:
                target_row.pack_forget()

        elif modality == "Image":
            task_var = ctk.StringVar(value=_IMG_TASK_OPTS[0])
            label_mode_var = ctk.StringVar(value=_LABEL_MODE_OPTS[0])

            def _refresh_image_metrics() -> None:
                backend_task = resolve_image_task(task_var.get(), label_mode_var.get())
                metric_menu = getattr(self, "_image_metric_menu", None)
                metric_var = self._modality_context_vars.get("metric")
                values = image_valid_metrics_for_task(backend_task)
                if metric_menu is not None and metric_var is not None and values:
                    metric_menu.configure(values=values)
                    metric_var.set(image_default_metric_for_task(backend_task))

            def _on_image_task_change(task: str) -> None:
                lm_row = getattr(self, "_img_label_mode_row", None)
                task_row = getattr(self, "_img_task_row", None)
                if lm_row is not None:
                    if task == _IMG_CLASSIFICATION_TASK:
                        if task_row is not None:
                            lm_row.pack(fill="x", padx=14, pady=4, after=task_row)
                        else:
                            lm_row.pack(fill="x", padx=14, pady=4)
                    else:
                        lm_row.pack_forget()
                _refresh_image_metrics()

            def _on_image_label_mode_change(_v: str) -> None:
                _refresh_image_metrics()

            self._img_task_row = _ctx_row(ctx_card, "Task type", task_var, _IMG_TASK_OPTS,
                                          width=280, command=_on_image_task_change, required=True)
            self._modality_context_vars["task_type"] = task_var

            self._img_label_mode_row = _ctx_row(ctx_card, "Label mode", label_mode_var, _LABEL_MODE_OPTS,
                                                width=180, command=_on_image_label_mode_change, required=True)
            self._modality_context_vars["label_mode"] = label_mode_var
            if task_var.get() != _IMG_CLASSIFICATION_TASK:
                self._img_label_mode_row.pack_forget()

            initial_task = resolve_image_task(task_var.get(), label_mode_var.get())
            metric_values = image_valid_metrics_for_task(initial_task)
            metric_var = ctk.StringVar(value=image_default_metric_for_task(initial_task))
            _ctx_row(ctx_card, "Priority metric", metric_var, metric_values, width=180)
            self._modality_context_vars["metric"] = metric_var
            self._image_metric_menu = self._modality_menus[-1]

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _IMG_DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

            fmt_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Format", fmt_var, _IMG_FORMAT_OPTS, width=200)
            self._modality_context_vars["image_format"] = fmt_var

            color_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Color space", color_var, _IMG_COLOR_OPTS, width=200)
            self._modality_context_vars["color_space"] = color_var

        elif modality == "Audio":
            def _on_audio_task_change(task: str) -> None:
                backend_task = _AUD_TASK_BACKEND.get(task, "classification")
                metric_menu = getattr(self, "_audio_metric_menu", None)
                metric_var = self._modality_context_vars.get("metric")
                values = audio_valid_metrics_for_task(backend_task)
                if metric_menu is not None and metric_var is not None and values:
                    metric_menu.configure(values=values)
                    metric_var.set(audio_default_metric_for_task(backend_task))

            task_var = ctk.StringVar(value=_AUD_TASK_OPTS[0])
            _ctx_row(ctx_card, "Task type", task_var, _AUD_TASK_OPTS, width=280, command=_on_audio_task_change, required=True)
            self._modality_context_vars["task_type"] = task_var

            initial_task = _AUD_TASK_BACKEND.get(task_var.get(), "classification")
            metric_values = audio_valid_metrics_for_task(initial_task)
            metric_var = ctk.StringVar(value=audio_default_metric_for_task(initial_task))
            _ctx_row(ctx_card, "Priority metric", metric_var, metric_values, width=220)
            self._modality_context_vars["metric"] = metric_var
            self._audio_metric_menu = self._modality_menus[-1]

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _AUD_DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

            ctk.CTkLabel(ctx_card,
                         text="Upload a ZIP containing class folders or manifests plus .wav/.mp3/.flac/.ogg files. WAV works by default; other formats require optional decoder support.",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED,
                         wraplength=780, justify="left").pack(anchor="w", padx=154, pady=(0, 6))

            fmt_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Format", fmt_var, _AUD_FORMAT_OPTS, width=200)
            self._modality_context_vars["audio_format"] = fmt_var

            channel_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Channel layout", channel_var, _AUD_CHANNEL_OPTS, width=180)
            self._modality_context_vars["channel_layout"] = channel_var

            sr_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Sample rate", sr_var, _AUD_SR_OPTS, width=240)
            self._modality_context_vars["sample_rate"] = sr_var

        elif modality == "Text":
            def _rebuild_col_fields(backend_task: str) -> None:
                frame = getattr(self, "_txt_col_frame", None)
                hdr = getattr(self, "_txt_col_hdr", None)
                anchor = getattr(self, "_txt_col_anchor", None)
                if frame is None:
                    return
                try:
                    for w in frame.winfo_children():
                        w.destroy()
                except Exception:
                    return
                self._txt_col_entries = {}
                self._txt_binary_entry = None

                required_fields = _TXT_REQUIRED_COL_FIELDS.get(backend_task, [])
                optional_fields = _TXT_OPTIONAL_COL_FIELDS.get(backend_task, [])

                ref = anchor if anchor else None
                if hdr:
                    hdr_kwargs = {"fill": "x", "padx": 14, "pady": (8, 0)}
                    if ref:
                        hdr_kwargs["before"] = ref
                    hdr.pack(**hdr_kwargs)
                frame_kwargs = {"fill": "x"}
                if ref:
                    frame_kwargs["before"] = ref
                frame.pack(**frame_kwargs)

                def _build_field_row(parent, col_key, col_label, placeholder, required):
                    row = ctk.CTkFrame(parent, fg_color="transparent")
                    row.pack(fill="x", padx=14, pady=(2, 2))
                    lbl_wrap = ctk.CTkFrame(row, fg_color="transparent")
                    lbl_wrap.pack(side="left")
                    ctk.CTkLabel(
                        lbl_wrap, text=col_label,
                        font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                        text_color=TXT_MUTED, width=220, anchor="w",
                    ).pack(side="left")
                    if required:
                        ctk.CTkLabel(
                            lbl_wrap, text=" *",
                            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                            text_color=ERROR,
                        ).pack(side="left")
                    else:
                        ctk.CTkLabel(
                            lbl_wrap, text=" optional",
                            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                            text_color=TXT_MUTED,
                        ).pack(side="left")
                    entry = ctk.CTkEntry(
                        row,
                        placeholder_text=placeholder,
                        fg_color=BG_INPUT, border_color=BORDER,
                        text_color=TXT, placeholder_text_color=TXT_MUTED,
                        corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
                    )
                    entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
                    return entry

                if backend_task == "classification_multi":
                    fmt_row = ctk.CTkFrame(frame, fg_color="transparent")
                    fmt_row.pack(fill="x", padx=14, pady=(2, 2))
                    fmt_lbl_wrap = ctk.CTkFrame(fmt_row, fg_color="transparent")
                    fmt_lbl_wrap.pack(side="left")
                    ctk.CTkLabel(
                        fmt_lbl_wrap, text="Label format",
                        font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                        text_color=TXT_MUTED, width=220, anchor="w",
                    ).pack(side="left")
                    ctk.CTkLabel(
                        fmt_lbl_wrap, text=" *",
                        font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                        text_color=ERROR,
                    ).pack(side="left")
                    fmt_var = self._txt_multilabel_format_var
                    ctk.CTkOptionMenu(
                        fmt_row,
                        values=["Single multi-label column", "Multiple binary label columns"],
                        variable=fmt_var,
                        fg_color=BG_INPUT, button_color=ACCENT, button_hover_color=ACCENT_H,
                        dropdown_fg_color=BG_BAR, text_color=TXT,
                        font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30, width=260,
                        dynamic_resizing=False,
                        command=lambda _v: _rebuild_col_fields(backend_task),
                    ).pack(side="left", padx=(6, 0))

                for col_key, col_label, placeholder in required_fields:
                    entry = _build_field_row(frame, col_key, col_label, placeholder, required=True)
                    self._txt_col_entries[col_key] = entry

                if backend_task == "classification_multi":
                    if self._txt_multilabel_format_var.get().startswith("Single"):
                        entry = _build_field_row(frame, "labels", "Labels column", "Enter labels column name", required=True)
                        self._txt_col_entries["labels"] = entry
                    else:
                        binary_row = ctk.CTkFrame(frame, fg_color="transparent")
                        binary_row.pack(fill="x", padx=14, pady=(2, 2))
                        binary_lbl_wrap = ctk.CTkFrame(binary_row, fg_color="transparent")
                        binary_lbl_wrap.pack(side="left")
                        ctk.CTkLabel(
                            binary_lbl_wrap, text="Binary label columns",
                            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                            text_color=TXT_MUTED, width=220, anchor="w",
                        ).pack(side="left")
                        ctk.CTkLabel(
                            binary_lbl_wrap, text=" *",
                            font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                            text_color=ERROR,
                        ).pack(side="left")
                        binary_entry = ctk.CTkEntry(
                            binary_row,
                            placeholder_text="Enter comma-separated binary label column names",
                            fg_color=BG_INPUT, border_color=BORDER,
                            text_color=TXT, placeholder_text_color=TXT_MUTED,
                            corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
                        )
                        binary_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
                        self._txt_binary_entry = binary_entry

                for col_key, col_label, placeholder in optional_fields:
                    entry = _build_field_row(frame, col_key, col_label, placeholder, required=False)
                    self._txt_col_entries[col_key] = entry

                aux_row = ctk.CTkFrame(frame, fg_color="transparent")
                aux_row.pack(fill="x", padx=14, pady=(8, 2))
                aux_lbl_wrap = ctk.CTkFrame(aux_row, fg_color="transparent")
                aux_lbl_wrap.pack(side="left")
                ctk.CTkLabel(
                    aux_lbl_wrap, text="Auxiliary feature columns",
                    font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                    text_color=TXT_MUTED, width=220, anchor="w",
                ).pack(side="left")
                ctk.CTkLabel(
                    aux_lbl_wrap, text=" optional",
                    font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                    text_color=TXT_MUTED,
                ).pack(side="left")
                aux_entry = ctk.CTkEntry(
                    aux_row,
                    placeholder_text="Enter comma-separated auxiliary numeric/categorical column names (leave blank to auto-detect)",
                    fg_color=BG_INPUT, border_color=BORDER,
                    text_color=TXT, placeholder_text_color=TXT_MUTED,
                    corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), height=30,
                )
                aux_entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
                self._txt_aux_entry = aux_entry

            self._txt_multilabel_format_var = ctk.StringVar(value="Single multi-label column")
            self._txt_binary_entry = None
            self._txt_aux_entry = None

            task_var = ctk.StringVar(value=_TXT_TASK_OPTS[0])
            label_mode_var = ctk.StringVar(value=_LABEL_MODE_OPTS[0])

            def _apply_text_task() -> None:
                backend_task = resolve_text_task(task_var.get(), label_mode_var.get())
                metric_menu = getattr(self, "_text_metric_menu", None)
                metric_var = self._modality_context_vars.get("metric")
                values = text_valid_metrics_for_task(backend_task)
                if metric_menu is not None and metric_var is not None and values:
                    metric_menu.configure(values=values)
                    metric_var.set(text_default_metric_for_task(backend_task))
                _rebuild_col_fields(backend_task)

            def _on_text_task_change(task: str) -> None:
                lm_row = getattr(self, "_txt_label_mode_row", None)
                task_row = getattr(self, "_txt_task_row", None)
                if lm_row is not None:
                    if task == _TXT_CLASSIFICATION_TASK:
                        if task_row is not None:
                            lm_row.pack(fill="x", padx=14, pady=4, after=task_row)
                        else:
                            lm_row.pack(fill="x", padx=14, pady=4)
                    else:
                        lm_row.pack_forget()
                _apply_text_task()

            def _on_text_label_mode_change(_v: str) -> None:
                _apply_text_task()

            self._txt_task_row = _ctx_row(ctx_card, "Task type", task_var, _TXT_TASK_OPTS,
                                          width=280, command=_on_text_task_change, required=True)
            self._modality_context_vars["task_type"] = task_var

            self._txt_label_mode_row = _ctx_row(ctx_card, "Label mode", label_mode_var, _LABEL_MODE_OPTS,
                                                width=180, command=_on_text_label_mode_change, required=True)
            self._modality_context_vars["label_mode"] = label_mode_var
            if task_var.get() != _TXT_CLASSIFICATION_TASK:
                self._txt_label_mode_row.pack_forget()

            initial_task = resolve_text_task(task_var.get(), label_mode_var.get())
            metric_values = text_valid_metrics_for_task(initial_task)
            metric_var = ctk.StringVar(value=text_default_metric_for_task(initial_task))
            _ctx_row(ctx_card, "Priority metric", metric_var, metric_values, width=220)
            self._modality_context_vars["metric"] = metric_var
            self._text_metric_menu = self._modality_menus[-1]

            domain_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Domain / use case", domain_var, _DOMAIN_OPTS, width=240)
            self._modality_context_vars["domain"] = domain_var

            ctk.CTkLabel(
                ctx_card,
                text="English text only — non-English rows are removed automatically before evaluation. "
                     "Upload a CSV or Excel file with one sample per row.",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED,
                wraplength=780, justify="left",
            ).pack(anchor="w", padx=154, pady=(0, 4))

            col_hdr = ctk.CTkFrame(ctx_card, fg_color="transparent")
            ctk.CTkLabel(
                col_hdr, text="Required columns for this task",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                text_color=TXT, anchor="w",
            ).pack(side="left", padx=(0, 6))
            ctk.CTkLabel(
                col_hdr, text="Type the exact column name from the uploaded file. * = required.",
                font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED,
            ).pack(side="left")
            self._txt_col_hdr = col_hdr

            txt_col_frame = ctk.CTkFrame(ctx_card, fg_color="transparent")
            self._txt_col_frame = txt_col_frame
            self._txt_col_entries = {}

            txt_col_anchor = ctk.CTkFrame(ctx_card, height=0, fg_color="transparent")
            txt_col_anchor.pack(fill="x")
            self._txt_col_anchor = txt_col_anchor

            _rebuild_col_fields(initial_task)

            ctk.CTkFrame(ctx_card, height=6, fg_color="transparent").pack()
            ctk.CTkLabel(
                ctx_card, text="Optional context  /  preferences",
                font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                text_color=TXT_MUTED, anchor="w",
            ).pack(anchor="w", padx=14, pady=(2, 2))

            source_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Text source", source_var, _TXT_SOURCE_OPTS, width=260)
            self._modality_context_vars["text_source"] = source_var

            len_var = ctk.StringVar(value=_SENTINEL)
            _ctx_row(ctx_card, "Text length", len_var, _TXT_LENGTH_OPTS, width=260)
            self._modality_context_vars["text_length"] = len_var

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
        input_format_label = self._input_format_var.get() if hasattr(self, "_input_format_var") else ""
        fmt = get_input_format(modality, input_format_label)
        selected = [k for k, var in self._constraint_vars.items() if var.get()]

        record_path = ""
        rp_var = getattr(self, "_record_path_var", None)
        if rp_var is not None:
            try:
                record_path = rp_var.get().strip()
            except Exception:
                record_path = ""

        result = {
            "modality":     modality,
            "input_format": input_format_label,
            "input_format_key": fmt.key if fmt else "",
            "record_path":  record_path,
            "constraints":  ", ".join(selected),
            "notes":        self._notes.get("1.0", "end").strip(),
        }

        for key, var in self._modality_context_vars.items():
            result[key] = _val(var)

        if modality == "Tabular":
            raw_task = self._csv_task_var.get()
            result["task_type"] = _CSV_TASK_BACKEND.get(raw_task, "")

        if modality == "Text":
            col_overrides = {}
            for key, entry in getattr(self, "_txt_col_entries", {}).items():
                try:
                    val = entry.get().strip()
                    if val:
                        col_overrides[key] = val
                except Exception:
                    pass

            binary_cols: list = []
            binary_entry = getattr(self, "_txt_binary_entry", None)
            if binary_entry is not None:
                try:
                    raw = binary_entry.get().strip()
                    if raw:
                        binary_cols = [c.strip() for c in raw.split(",") if c.strip()]
                except Exception:
                    pass
            if binary_cols:
                col_overrides["binary_label_columns"] = binary_cols

            aux_cols: list = []
            aux_entry = getattr(self, "_txt_aux_entry", None)
            if aux_entry is not None:
                try:
                    raw = aux_entry.get().strip()
                    if raw:
                        aux_cols = [c.strip() for c in raw.split(",") if c.strip()]
                except Exception:
                    pass

            fmt_var = getattr(self, "_txt_multilabel_format_var", None)
            fmt_choice = fmt_var.get() if fmt_var is not None else "Single multi-label column"
            multilabel_format = "binary_columns" if fmt_choice.startswith("Multiple") else "single_column"

            result["col_overrides"] = col_overrides
            result["auxiliary_feature_columns"] = aux_cols
            result["binary_label_columns"] = binary_cols
            result["multilabel_format"] = multilabel_format

            fmt_key = result.get("input_format_key", "")
            if fmt_key == "txt_zip":
                result["metadata_path"] = record_path
                result["record_path"] = ""
            elif fmt_key == "json_text_records":
                result["metadata_path"] = ""
            else:
                result["metadata_path"] = ""

        return result

    def _clear_context_fields(self) -> None:
        target = getattr(self, "_target", None)
        if target is not None:
            try:
                target.delete(0, "end")
            except Exception:
                pass
        rp_entry = getattr(self, "_record_path_entry", None)
        if rp_entry is not None:
            try:
                rp_entry.delete(0, "end")
            except Exception:
                pass
        for entry in getattr(self, "_txt_col_entries", {}).values():
            try:
                entry.delete(0, "end")
            except Exception:
                pass
        for attr in ("_txt_binary_entry", "_txt_aux_entry"):
            entry = getattr(self, attr, None)
            if entry is not None:
                try:
                    entry.delete(0, "end")
                except Exception:
                    pass
        self._notes.delete("1.0", "end")
        for key, var in self._modality_context_vars.items():
            if key in ("task_type", "metric", "label_mode"):
                continue
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
        task_type = msg.get("task_context", {}).get("task_type", "")
        supervision = msg.get("task_context", {}).get("supervision", "")
        if task_type in {"regression", "time_series"}:
            ctk.CTkLabel(c1, text="Continuous target",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
                anchor="w", padx=14, pady=(0, 12))
        elif supervision == "unsupervised":
            ctk.CTkLabel(c1, text="Unsupervised tabular evaluation",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
                anchor="w", padx=14, pady=(0, 12))
        else:
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
        m       = msg.get("raw_metrics", msg["metrics"])
        metric_names = valid_metrics_for_task(task_type) or list(m.keys())
        for mk in metric_names:
            if mk not in m:
                continue
            is_p = (mk == pmetric)
            mr   = ctk.CTkFrame(c3, fg_color="transparent")
            mr.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(mr,
                         text=metric_label(mk),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED,
                         width=74, anchor="w").pack(side="left")
            ctk.CTkLabel(mr, text=f"{m[mk]:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=TXT if is_p else TXT_MUTED).pack(side="left")
        ctk.CTkLabel(c3, text=f"Normalized score {msg['best_score']:.4f}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(4, 0))
        eval_mode = msg.get("evaluation_mode", "")
        if eval_mode:
            ctk.CTkLabel(c3, text=f"Evaluation mode: {eval_mode}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
                anchor="w", padx=14, pady=(2, 0))
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

    def _show_audio_run_results(self, msg: dict) -> None:
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
        ctk.CTkLabel(c1, text=f"{msg['n_audio_files']:,} audio files",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"),
                     text_color=TXT).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=f"{msg['n_classes']} labels",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=f"avg {msg.get('avg_duration_sec', 0):.2f}s  /  rates {msg.get('sample_rates', 'unknown')}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=msg.get("quality_info", ""),
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(0, 12))

        c2 = _card(cards_row)
        c2.pack(side="left", fill="both", expand=True, padx=6)
        _sec_label(c2, "BEST PIPELINE", padx=14, pady=(10, 4))
        for part in msg["best_name"].split(" | "):
            ctk.CTkLabel(c2, text=part,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                         anchor="w").pack(anchor="w", padx=14)
        ctk.CTkLabel(c2, text=f"{msg['n_pipelines']} candidates tested",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(6, 12))

        c3 = _card(cards_row)
        c3.pack(side="left", fill="both", expand=True, padx=(6, 0))
        _sec_label(c3, "METRICS", padx=14, pady=(10, 4))
        pmetric = msg["metric"]
        m = msg.get("raw_metrics", msg["metrics"])
        task_type = msg.get("task_context", {}).get("task_type", "")
        metric_names = audio_valid_metrics_for_task(task_type) or list(m.keys())
        for mk in metric_names:
            if mk not in m:
                continue
            is_p = (mk == pmetric)
            mr = ctk.CTkFrame(c3, fg_color="transparent")
            mr.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(mr, text=audio_metric_label(mk),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED, width=150, anchor="w").pack(side="left")
            ctk.CTkLabel(mr, text=f"{m[mk]:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=TXT if is_p else TXT_MUTED).pack(side="left")
        ctk.CTkLabel(c3, text=f"Normalized score {msg['best_score']:.4f}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(4, 0))
        if msg.get("evaluation_mode"):
            ctk.CTkLabel(c3, text=f"Evaluation mode: {msg['evaluation_mode']}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(2, 12))

        act = ctk.CTkFrame(rf, fg_color="transparent")
        act.pack(fill="x", padx=20, pady=(10, 20))
        ctk.CTkButton(act, text="View Report", width=130, height=34, fg_color=ACCENT, hover_color=ACCENT_H, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=lambda: self._switch_view("report")).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Export PDF", width=110, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._on_export_pdf).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Open Processed ZIP", width=150, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._open_cleaned).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Console", width=90, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=lambda: self._switch_view("console")).pack(side="left")
        rf.pack(fill="x")

    def _show_text_run_results(self, msg: dict) -> None:
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
        ctk.CTkLabel(c1, text=f"{msg['n_samples']:,} samples",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"),
                     text_color=TXT).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=f"{msg.get('n_classes', 0)} labels  /  vocab {msg.get('vocabulary_size_estimate', 0):,}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=f"avg {msg.get('avg_token_length', 0):.1f} tokens",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1, text=msg.get("quality_info", ""),
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(0, 12))

        c2 = _card(cards_row)
        c2.pack(side="left", fill="both", expand=True, padx=6)
        _sec_label(c2, "BEST PIPELINE", padx=14, pady=(10, 4))
        for part in msg["best_name"].split(" | "):
            ctk.CTkLabel(c2, text=part,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT,
                         anchor="w").pack(anchor="w", padx=14)
        ctk.CTkLabel(c2, text=f"{msg['n_pipelines']} candidates tested",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(6, 12))

        c3 = _card(cards_row)
        c3.pack(side="left", fill="both", expand=True, padx=(6, 0))
        _sec_label(c3, "METRICS", padx=14, pady=(10, 4))
        pmetric = msg["metric"]
        m = msg.get("raw_metrics", msg["metrics"])
        task_type = msg.get("task_context", {}).get("task_type", "")
        metric_names = text_valid_metrics_for_task(task_type) or list(m.keys())
        for mk in metric_names:
            if mk not in m:
                continue
            is_p = (mk == pmetric)
            mr = ctk.CTkFrame(c3, fg_color="transparent")
            mr.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(mr, text=text_metric_label(mk),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED, width=150, anchor="w").pack(side="left")
            ctk.CTkLabel(mr, text=f"{m[mk]:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold" if is_p else "normal"),
                         text_color=TXT if is_p else TXT_MUTED).pack(side="left")
        ctk.CTkLabel(c3, text=f"Normalized score {msg['best_score']:.4f}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(4, 0))
        if msg.get("evaluation_mode"):
            ctk.CTkLabel(c3, text=f"Evaluation mode: {msg['evaluation_mode']}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(2, 12))

        act = ctk.CTkFrame(rf, fg_color="transparent")
        act.pack(fill="x", padx=20, pady=(10, 20))
        ctk.CTkButton(act, text="View Report", width=130, height=34, fg_color=ACCENT, hover_color=ACCENT_H, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=lambda: self._switch_view("report")).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Export PDF", width=110, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._on_export_pdf).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Open Cleaned Dataset", width=170, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=self._open_cleaned).pack(side="left", padx=(0, 8))
        ctk.CTkButton(act, text="Console", width=90, height=34, fg_color=BG_INPUT, hover_color=BORDER, border_width=1, border_color=BORDER, text_color=TXT, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=lambda: self._switch_view("console")).pack(side="left")
        rf.pack(fill="x")

    def _show_image_run_results(self, msg: dict) -> None:
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
        ctk.CTkLabel(c1, text=f"{msg['n_images']:,} images",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=22, weight="bold"),
                     text_color=TXT).pack(anchor="w", padx=14)
        task_type = msg.get("task_context", {}).get("task_type", "")
        supervision = msg.get("task_context", {}).get("supervision", "")
        ctk.CTkLabel(c1, text=f"{msg['n_classes']} classes",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        ctk.CTkLabel(c1,
                     text=f"avg {msg['avg_height']}x{msg['avg_width']}  /  {msg.get('color_info', 'RGB')}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(anchor="w", padx=14)
        if task_type in {"generation", "depth", "ocr", "retrieval"} or supervision == "unsupervised":
            ctk.CTkLabel(c1, text=f"task: {task_type.replace('_', ' ')}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12), text_color=TXT_MUTED).pack(
                anchor="w", padx=14, pady=(0, 12))
        else:
            ir = msg.get("imbalance_ratio", 1.0)
            ctk.CTkLabel(c1, text=f"imbalance ratio {ir:.1f}x",
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
        m = msg.get("raw_metrics", msg["metrics"])
        metric_names = image_valid_metrics_for_task(task_type) or list(m.keys())
        for mk in metric_names:
            if mk not in m:
                continue
            is_p = (mk == pmetric)
            mr   = ctk.CTkFrame(c3, fg_color="transparent")
            mr.pack(fill="x", padx=14, pady=1)
            ctk.CTkLabel(mr,
                         text=image_metric_label(mk),
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=ACCENT if is_p else TXT_MUTED,
                         width=110, anchor="w").pack(side="left")
            ctk.CTkLabel(mr, text=f"{m[mk]:.4f}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12,
                                          weight="bold" if is_p else "normal"),
                         text_color=TXT if is_p else TXT_MUTED).pack(side="left")
        ctk.CTkLabel(c3, text=f"Normalized score {msg['best_score']:.4f}",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
            anchor="w", padx=14, pady=(4, 0))
        eval_mode = msg.get("evaluation_mode", "")
        if eval_mode:
            ctk.CTkLabel(c3, text=f"Evaluation mode: {eval_mode}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11), text_color=TXT_MUTED).pack(
                anchor="w", padx=14, pady=(2, 0))
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
            act, text="Open Processed ZIP", width=150, height=34,
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
