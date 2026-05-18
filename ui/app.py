"""
ui/app.py — Desktop assistant frontend v2.

Four views accessible via left navigation:
  Run      — primary: inputs, run, status, result cards, action buttons
  Report   — structured in-app report viewer (polished, section-based)
  Console  — full technical log with timestamps, color-coded levels
  History  — browse past runs, reopen reports and cleaned datasets

Run via:
    python start_ui.py
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import sys
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, Optional

# ── Project root ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_ROOT)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import customtkinter as ctk
except ImportError:
    print("[ERROR] customtkinter not installed.  Run:  pip install customtkinter")
    sys.exit(1)

try:
    from PIL import Image as PILImage
    _PIL_OK = True
except ImportError:
    PILImage = None
    _PIL_OK = False

# ── Appearance ─────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

from ui.constants import (
    BG_WIN, BG_SIDEBAR, BG_BAR,
    ACCENT, SUCCESS, ERROR, WARN,
    TXT, TXT_MUTED, FONT_FAMILY,
    NAV_W, POLL_MS,
)
from ui.helpers import _open_file, _hsep
from ui.worker_runner import run_worker
from src.tabular.config import default_metric_for_task
from src.utils.ingestion import get_input_format, is_supported
from ui.views.run_view import RunViewMixin
from ui.views.report_view import ReportViewMixin
from ui.views.console_view import ConsoleViewMixin
from ui.views.history_view import HistoryViewMixin
from ui.views.settings_view import SettingsViewMixin


class App(
    RunViewMixin,
    ReportViewMixin,
    ConsoleViewMixin,
    HistoryViewMixin,
    SettingsViewMixin,
    ctk.CTk,
):
    def __init__(self) -> None:
        super().__init__()
        self._q                                        = multiprocessing.Queue()
        self._proc:         Optional[multiprocessing.Process] = None
        self._csv_path:     Optional[Path]             = None
        self._cleaned_path: Optional[Path]             = None
        self._report_data:  Optional[dict]             = None
        self._pulse_on:     bool                       = False
        self._nav_btns:     Dict[str, ctk.CTkButton]  = {}
        self._views:        Dict[str, ctk.CTkFrame]   = {}
        self._theme_mode:   str                        = "dark"

        self._setup_window()
        self._build_nav()

        # Content area — all views live inside this frame
        self._content = ctk.CTkFrame(self, fg_color=BG_WIN, corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew", padx=(1, 0))

        self._build_run_view()
        self._build_report_view()
        self._build_console_view()
        self._build_history_view()
        self._build_settings_view()

        self._switch_view("run")
        self._poll()

    # ── Window ─────────────────────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.title("MorphAI")
        self.geometry("1200x740")
        self.minsize(960, 580)
        self.configure(fg_color=BG_WIN)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Window icon
        if _PIL_OK:
            try:
                import tempfile as _tf
                
                def _set_icon(is_dark: bool) -> None:
                    p = Path(__file__).parent / "logo" / ("Morph_AI_DarkMode_Logo.png" if is_dark else "Morph_AI_LightMode_Logo.png")
                    if not p.exists(): return
                    _raw = PILImage.open(str(p)).convert("RGBA")
                    _s32 = _raw.resize((32, 32), PILImage.LANCZOS)
                    if sys.platform == "win32":
                        _ico = _tf.NamedTemporaryFile(delete=False, suffix=".ico")
                        _ico.close()
                        _s32.save(_ico.name, format="ICO", sizes=[(32, 32), (16, 16)])
                        self.iconbitmap(_ico.name)
                    else:
                        from PIL import ImageTk
                        self._icon_tk = ImageTk.PhotoImage(_s32)
                        self.iconphoto(True, self._icon_tk)
                
                self._set_os_icon = _set_icon
                self.after(100, lambda: self._set_os_icon(ctk.get_appearance_mode().lower() == "dark"))
            except Exception:
                pass

    # ── Left navigation ────────────────────────────────────────────────────────

    def _build_nav(self) -> None:
        nav = ctk.CTkFrame(self, width=NAV_W, fg_color=BG_SIDEBAR, corner_radius=0)
        nav.grid(row=0, column=0, sticky="nsew")
        nav.grid_propagate(False)

        # Title — logo image + "MorphAI" text
        title_row = ctk.CTkFrame(nav, fg_color="transparent")
        title_row.pack(fill="x", padx=14, pady=(20, 6))

        _logo_dark = Path(__file__).parent / "logo" / "Morph_AI_DarkMode_Logo.png"
        _logo_light = Path(__file__).parent / "logo" / "Morph_AI_LightMode_Logo.png"
        if _PIL_OK and _logo_dark.exists() and _logo_light.exists():
            try:
                _img_dark = PILImage.open(_logo_dark).convert("RGBA").resize((28, 28), PILImage.LANCZOS)
                _img_light = PILImage.open(_logo_light).convert("RGBA").resize((28, 28), PILImage.LANCZOS)
                self._logo_img = ctk.CTkImage(light_image=_img_light, dark_image=_img_dark, size=(28, 28))
                ctk.CTkLabel(title_row, image=self._logo_img, text="").pack(side="left")
            except Exception:
                ctk.CTkLabel(title_row, text="◆", font=ctk.CTkFont(family=FONT_FAMILY, size=20),
                             text_color=ACCENT).pack(side="left")
        else:
            ctk.CTkLabel(title_row, text="◆", font=ctk.CTkFont(family=FONT_FAMILY, size=20),
                         text_color=ACCENT).pack(side="left")

        ctk.CTkLabel(title_row, text="  MorphAI",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
                     text_color=TXT).pack(side="left")
        ctk.CTkLabel(nav, text="Adaptive Preprocessing Agent",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                     text_color=TXT_MUTED).pack(anchor="w", padx=14, pady=(0, 14))

        _hsep(nav, pady=(0, 10))

        for key, icon, label in [
            ("run",     "▶", "Run  /  Assistant"),
            ("report",  "■", "Report"),
            ("console", "≡", "Console"),
            ("history", "◷", "History"),
        ]:
            btn = ctk.CTkButton(
                nav,
                text=f"  {icon}   {label}",
                anchor="w", height=42, corner_radius=8,
                fg_color="transparent", hover_color=BG_BAR,
                text_color=TXT_MUTED, font=ctk.CTkFont(family=FONT_FAMILY, size=15),
                command=lambda k=key: self._switch_view(k),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self._nav_btns[key] = btn

        # Push settings to bottom
        ctk.CTkFrame(nav, fg_color="transparent").pack(fill="both", expand=True)
        _hsep(nav, pady=(4, 4))

        settings_btn = ctk.CTkButton(
            nav,
            text="  ⚙   Settings",
            anchor="w", height=42, corner_radius=8,
            fg_color="transparent", hover_color=BG_BAR,
            text_color=TXT_MUTED, font=ctk.CTkFont(family=FONT_FAMILY, size=15),
            command=lambda: self._switch_view("settings"),
        )
        settings_btn.pack(fill="x", padx=10, pady=(2, 10))
        self._nav_btns["settings"] = settings_btn

    def _switch_view(self, name: str) -> None:
        for k, btn in self._nav_btns.items():
            if k == name:
                btn.configure(fg_color=ACCENT, text_color=("#ffffff", "#ffffff"),
                              font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"))
            else:
                btn.configure(fg_color="transparent", text_color=TXT_MUTED,
                              font=ctk.CTkFont(family=FONT_FAMILY, size=15))
        for frame in self._views.values():
            frame.pack_forget()
        self._views[name].pack(fill="both", expand=True)
        if name == "history":
            self._refresh_history()

    # ── Queue polling ──────────────────────────────────────────────────────────

    def _poll(self) -> None:
        try:
            while True:
                msg  = self._q.get_nowait()
                kind = msg["kind"]
                if kind == "log":
                    self._clog(msg["text"], msg.get("level", "INFO"))
                    if msg.get("level") == "STEP":
                        self._step_lbl.configure(text=msg["text"])
                elif kind == "done":
                    self._on_done(msg)
                elif kind == "fail":
                    self._on_fail(msg.get("text", "Unknown error"))
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_browse(self) -> None:
        modality = self._modality_var.get()
        input_format_label = self._input_format_var.get() if hasattr(self, "_input_format_var") else ""
        fmt = get_input_format(modality, input_format_label)

        if fmt is None or not fmt.implemented:
            self._clog(
                f"'{input_format_label}' for {modality} is not implemented yet.",
                "WARN",
            )
            return

        title = fmt.file_dialog_title
        filetypes = list(fmt.file_dialog_filters)
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not path:
            return
        self._csv_path = Path(path)
        name = self._csv_path.name
        self._file_lbl.configure(
            text=(name if len(name) <= 40 else name[:37] + "…"),
            text_color=TXT,
        )

    def _start_worker(self, modality: str, args: tuple, kwargs: dict) -> None:
        self._proc = multiprocessing.Process(
            target=run_worker,
            args=(modality, self._q, args, kwargs),
            daemon=True,
        )
        self._proc.start()

    def _on_run(self) -> None:
        if self._proc and self._proc.is_alive():
            return

        modality = self._modality_var.get()
        input_format_label = self._input_format_var.get() if hasattr(self, "_input_format_var") else ""

        if not is_supported(modality, input_format_label):
            fmt = get_input_format(modality, input_format_label)
            hint = (fmt.coming_soon_hint if fmt else "") or "Please choose an implemented format."
            self._clog(
                f"This input format is planned but not implemented yet. {hint}",
                "WARN",
            )
            self._switch_view("console")
            return

        if not self._csv_path or not self._csv_path.exists():
            self._clog("Please select a valid input file.", "ERROR")
            self._switch_view("console")
            return

        if modality == "Image":
            ctx = self._get_context_fields()
            img_metric = ctx.get("metric", "f1")
            if not img_metric or img_metric == "— select —":
                img_metric = "f1"

            from src.image.config import resolve_image_task, SUPPORTED_TASK_TYPES as _IMG_SUPPORTED
            raw_task = ctx.get("task_type", "")
            label_mode = ctx.get("label_mode", "")
            backend_task = resolve_image_task(raw_task, label_mode)
            if not backend_task or backend_task not in _IMG_SUPPORTED:
                self._clog(
                    f"Image task type '{raw_task}' is not supported or has been deprecated. "
                    "Please select a supported image task type.",
                    "ERROR",
                )
                self._switch_view("console")
                return

            self._cleaned_path = None
            self._report_data  = None
            self._results_frame.pack_forget()
            self._step_lbl.configure(text="")
            self._set_status("Running", WARN)

            self._start_worker(
                "Image",
                (self._csv_path, img_metric),
                dict(
                    task_type=backend_task,
                    label_mode=label_mode,
                    domain=ctx.get("domain", ""),
                    constraints=ctx.get("constraints", ""),
                    notes=ctx.get("notes", ""),
                    image_format=ctx.get("image_format", ""),
                    color_space=ctx.get("color_space", ""),
                    input_format=ctx.get("input_format", ""),
                    input_format_key=ctx.get("input_format_key", "zip_folder"),
                    annotation_path=ctx.get("annotation_path", ""),
                    image_dir=ctx.get("image_dir", ""),
                    annotation_dir=ctx.get("annotation_dir", ""),
                    label_dir=ctx.get("label_dir", ""),
                    class_config=ctx.get("class_config", ""),
                    split_selection=ctx.get("split_selection", ""),
                ),
            )

            self._clear_context_fields()
            self._csv_path = None
            self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
            self._set_controls(False)
            return

        if modality == "Audio":
            ctx = self._get_context_fields()
            from src.audio.config import _AUD_TASK_BACKEND, default_metric_for_task as audio_default_metric_for_task
            raw_task = ctx.get("task_type", "")
            backend_task = _AUD_TASK_BACKEND.get(raw_task, "")
            audio_metric = ctx.get("metric", "") or audio_default_metric_for_task(backend_task)
            if not backend_task:
                self._clog("Please select a supported audio task type.", "ERROR")
                self._switch_view("console")
                return
            self._cleaned_path = None
            self._report_data = None
            self._results_frame.pack_forget()
            self._step_lbl.configure(text="")
            self._set_status("Running", WARN)
            self._start_worker(
                "Audio",
                (self._csv_path, audio_metric),
                dict(
                    task_type=backend_task,
                    domain=ctx.get("domain", ""),
                    constraints=ctx.get("constraints", ""),
                    notes=ctx.get("notes", ""),
                    audio_format=ctx.get("audio_format", ""),
                    channel_layout=ctx.get("channel_layout", ""),
                    sample_rate=ctx.get("sample_rate", ""),
                    input_format=ctx.get("input_format", ""),
                    input_format_key=ctx.get("input_format_key", "zip_folder"),
                    metadata_path=ctx.get("metadata_path", ""),
                    record_path=ctx.get("record_path", ""),
                    field_overrides=ctx.get("audio_field_overrides", {}) or {},
                ),
            )
            self._clear_context_fields()
            self._csv_path = None
            self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
            self._set_controls(False)
            return

        if modality == "Text":
            ctx = self._get_context_fields()
            from src.text.config import (
                resolve_text_task,
                SUPPORTED_TASK_TYPES as _TXT_SUPPORTED,
                default_metric_for_task as text_default_metric_for_task,
            )
            raw_task = ctx.get("task_type", "")
            label_mode = ctx.get("label_mode", "")
            backend_task = resolve_text_task(raw_task, label_mode)
            text_metric = ctx.get("metric", "") or text_default_metric_for_task(backend_task)
            if not backend_task or backend_task not in _TXT_SUPPORTED:
                self._clog(
                    f"Text task type '{raw_task}' is not supported or has been deprecated. "
                    "Please select a supported text task type.",
                    "ERROR",
                )
                self._switch_view("console")
                return
            self._cleaned_path = None
            self._report_data = None
            self._results_frame.pack_forget()
            self._step_lbl.configure(text="")
            self._set_status("Running", WARN)
            self._start_worker(
                "Text",
                (self._csv_path, text_metric),
                dict(
                    task_type=backend_task,
                    label_mode=label_mode,
                    domain=ctx.get("domain", ""),
                    constraints=ctx.get("constraints", ""),
                    notes=ctx.get("notes", ""),
                    language=ctx.get("language", ""),
                    text_source=ctx.get("text_source", ""),
                    text_length=ctx.get("text_length", ""),
                    col_overrides=ctx.get("col_overrides") or {},
                    auxiliary_feature_columns=ctx.get("auxiliary_feature_columns") or [],
                    multilabel_format=ctx.get("multilabel_format", "single_column"),
                    binary_label_columns=ctx.get("binary_label_columns") or [],
                    input_format=ctx.get("input_format", ""),
                    input_format_key=ctx.get("input_format_key", "csv_excel"),
                    record_path=ctx.get("record_path", ""),
                    metadata_path=ctx.get("metadata_path", ""),
                ),
            )
            self._clear_context_fields()
            self._csv_path = None
            self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
            self._set_controls(False)
            return

        ctx = self._get_context_fields()

        if self._csv_target_row.winfo_ismapped():
            target = self._target.get().strip()
            if not target:
                self._clog("Please enter the target column name.", "ERROR")
                self._switch_view("console")
                return
            metric = self._metric_var.get()
        else:
            target = ""
            metric = self._metric_var.get() or default_metric_for_task(ctx.get("task_type", ""))

        self._cleaned_path = None
        self._report_data  = None
        self._results_frame.pack_forget()
        self._step_lbl.configure(text="")
        self._set_status("Running", WARN)

        self._start_worker(
            "Tabular",
            (self._csv_path, target, metric),
            dict(
                task_type=ctx.get("task_type", ""),
                domain=ctx.get("domain", ""),
                constraints=ctx.get("constraints", ""),
                notes=ctx.get("notes", ""),
                modality=ctx.get("modality", "Tabular"),
                input_format=ctx.get("input_format", ""),
                input_format_key=ctx.get("input_format_key", ""),
                record_path=ctx.get("record_path", ""),
                fe_budget=ctx.get("fe_budget", ""),
                data_quality=ctx.get("data_quality", ""),
            ),
        )

        self._clear_context_fields()
        self._csv_path = None
        self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
        self._set_controls(False)

    def _on_done(self, msg: dict) -> None:
        self._set_controls(True)
        self._cleaned_path = msg["cleaned_path"]
        self._report_data  = msg["report"]
        self._set_status("Done", SUCCESS)
        self._step_lbl.configure(text="All steps complete.")
        if msg.get("modality") == "Image":
            self._show_image_run_results(msg)
        elif msg.get("modality") == "Audio":
            self._show_audio_run_results(msg)
        elif msg.get("modality") == "Text":
            self._show_text_run_results(msg)
        else:
            self._show_run_results(msg)
        self._render_report(msg["report"])
        self._refresh_history()

    def _on_fail(self, text: str) -> None:
        self._set_controls(True)
        self._set_status("Error", ERROR)
        short = text[:80] + "..." if len(text) > 80 else text
        self._step_lbl.configure(text=f"Failed: {short}")
        self._switch_view("console")

    # ── PDF export ─────────────────────────────────────────────────────────────

    def _on_export_pdf(self) -> None:
        self._export_pdf_for(self._report_data)

    def _export_pdf_for(self, report: Optional[dict]) -> None:
        if not report:
            self._clog("No report available. Run the agent first.", "WARN")
            return
        try:
            from ui.pdf_exporter import export_report_pdf
        except ImportError as e:
            messagebox.showerror(
                "PDF Export Error",
                f"PDF export requires fpdf2.\n\npip install fpdf2\n\n{e}",
            )
            return

        cfg          = report.get("config", {})
        ds_stem      = Path(cfg.get("data_path", "report")).stem
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{ds_stem}_report_{ts}.pdf"

        path = filedialog.asksaveasfilename(
            title="Export Report as PDF",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            out = export_report_pdf(report, Path(path))
            self._clog(f"PDF exported: {out}", "OK")
            _open_file(out)
        except Exception as exc:
            self._clog(f"PDF export failed: {exc}", "ERROR")
            messagebox.showerror("PDF Export Failed", str(exc))

    # ── File helpers ───────────────────────────────────────────────────────────

    def _open_cleaned(self) -> None:
        if self._cleaned_path and Path(self._cleaned_path).exists():
            _open_file(Path(self._cleaned_path))
        else:
            self._clog("Cleaned dataset not available yet.", "WARN")

    # ── Status / control helpers ───────────────────────────────────────────────

    def _set_status(self, text: str, color) -> None:
        self._pulse_on = (text == "Running")
        self._dot.configure(text_color=color)
        self._status_lbl.configure(text=text, text_color=color)
        if self._pulse_on:
            self._pulse()

    def _pulse(self) -> None:
        if not self._pulse_on:
            return
        cur = self._dot.cget("text_color")
        nxt = ACCENT if cur == TXT_MUTED else TXT_MUTED
        self._dot.configure(text_color=nxt)
        self.after(500, self._pulse)

    def _set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._run_btn.configure(state=state)
        self._browse_btn.configure(state=state)
        self._modality_menu.configure(state=state)
        if hasattr(self, "_input_format_menu"):
            self._input_format_menu.configure(state=state)
        for menu in self._modality_menus:
            menu.configure(state=state)
        for cb in self._modality_checks:
            cb.configure(state=state)
        self._notes.configure(state=state)

        for entry in getattr(self, "_txt_col_entries", {}).values():
            try:
                entry.configure(state=state)
            except Exception:
                pass
        for attr in ("_txt_binary_entry", "_txt_aux_entry", "_record_path_entry", "_target"):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.configure(state=state)
                except Exception:
                    pass

        if enabled:
            self._apply_input_format_state()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
