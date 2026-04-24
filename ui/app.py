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

import os
import queue
import sys
import threading
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
from ui.worker import AgentWorker, ImageAgentWorker
from src.tabular.config import default_metric_for_task
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
        self._q:            queue.Queue                = queue.Queue()
        self._thread:       Optional[threading.Thread] = None
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

        if modality == "Image":
            path = filedialog.askopenfilename(
                title="Select image dataset zip archive",
                filetypes=[("Zip archives", "*.zip"), ("All files", "*.*")],
            )
            if not path:
                return
            self._csv_path = Path(path)
            name = self._csv_path.name
            self._file_lbl.configure(
                text=(name if len(name) <= 40 else name[:37] + "…"),
                text_color=TXT,
            )
            return

        _filetypes_map = {
            "CSV / Tabular":   [("CSV files", "*.csv"), ("All files", "*.*")],
            "Audio":           [("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All files", "*.*")],
            "Text":            [("Text files", "*.txt *.csv *.json *.jsonl"), ("All files", "*.*")],
            "Semi-structured": [("Structured files", "*.json *.xml *.yaml *.yml *.log"), ("All files", "*.*")],
            "Unstructured":    [("All files", "*.*")],
        }
        _titles_map = {
            "CSV / Tabular":   "Select CSV dataset",
            "Audio":           "Select audio file or dataset",
            "Text":            "Select text dataset",
            "Semi-structured": "Select semi-structured dataset",
            "Unstructured":    "Select dataset",
        }
        path = filedialog.askopenfilename(
            title=_titles_map.get(modality, "Select dataset"),
            filetypes=_filetypes_map.get(modality, [("All files", "*.*")]),
        )
        if not path:
            return
        self._csv_path = Path(path)
        name = self._csv_path.name
        self._file_lbl.configure(
            text=(name if len(name) <= 40 else name[:37] + "…"),
            text_color=TXT,
        )

    def _on_run(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        modality = self._modality_var.get()

        if not self._csv_path or not self._csv_path.exists():
            self._clog("Please select a valid input file.", "ERROR")
            self._switch_view("console")
            return

        if modality == "Image":
            ctx = self._get_context_fields()
            img_metric = ctx.get("metric", "f1")
            if not img_metric or img_metric == "— select —":
                img_metric = "f1"

            self._cleaned_path = None
            self._report_data  = None
            self._results_frame.pack_forget()
            self._step_lbl.configure(text="")
            self._set_status("Running", WARN)

            from src.image.config import _IMG_TASK_BACKEND
            raw_task = ctx.get("task_type", "")
            backend_task = _IMG_TASK_BACKEND.get(raw_task, "classification")

            worker = ImageAgentWorker(
                self._q, self._csv_path, img_metric,
                task_type=backend_task,
                domain=ctx.get("domain", ""),
                constraints=ctx.get("constraints", ""),
                notes=ctx.get("notes", ""),
                image_format=ctx.get("image_format", ""),
                color_space=ctx.get("color_space", ""),
            )
            self._thread = threading.Thread(target=worker.run, daemon=True)
            self._thread.start()

            self._clear_context_fields()
            self._csv_path = None
            self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)
            return

        if modality not in ("CSV / Tabular", "Image"):
            ctx = self._get_context_fields()
            self._clog(f"[{modality}]  Run context captured.", "INFO")
            for k, v in ctx.items():
                if v and k not in ("modality", "notes"):
                    self._clog(f"  {k}: {v}", "INFO")
            if ctx.get("notes"):
                self._clog(f"  notes: {ctx['notes']}", "INFO")
            self._clog(f"Backend pipeline for '{modality}' is not yet implemented.", "WARN")
            self._switch_view("console")
            return

        ctx = self._get_context_fields()

        if self._csv_target_frame.winfo_ismapped():
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

        worker = AgentWorker(
            self._q, self._csv_path, target, metric,
            task_type=ctx.get("task_type", ""),
            domain=ctx.get("domain", ""),
            constraints=ctx.get("constraints", ""),
            notes=ctx.get("notes", ""),
            modality=ctx.get("modality", "CSV / Tabular"),
            fe_budget=ctx.get("fe_budget", ""),
            data_quality=ctx.get("data_quality", ""),
        )
        self._thread = threading.Thread(target=worker.run, daemon=True)
        self._thread.start()

        self._clear_context_fields()
        self._csv_path = None
        self._file_lbl.configure(text="No file selected", text_color=TXT_MUTED)

    def _on_done(self, msg: dict) -> None:
        self._cleaned_path = msg["cleaned_path"]
        self._report_data  = msg["report"]
        self._set_status("Done", SUCCESS)
        self._step_lbl.configure(text="All steps complete.")
        if msg.get("modality") == "Image":
            self._show_image_run_results(msg)
        else:
            self._show_run_results(msg)
        self._render_report(msg["report"])
        self._refresh_history()

    def _on_fail(self, text: str) -> None:
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
        if self._modality_var.get() == "CSV / Tabular":
            self._csv_task_menu.configure(state=state)
            self._target.configure(state=state)
            self._metric_menu.configure(state=state)
        for menu in self._modality_menus:
            menu.configure(state=state)
        for cb in self._modality_checks:
            cb.configure(state=state)
        self._notes.configure(state=state)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
