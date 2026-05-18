"""
ui/views/console_view.py — Console view mixin.

Provides _build_console_view and all console helpers for the App class.
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

from ui.constants import (
    BG_WIN, BG_BAR, BG_INPUT, BG_CONSOLE,
    BORDER, TXT, TXT_MUTED, TXT_CONSOLE,
    _TK_TXT, _TK_MUTED, _TK_ACCENT, _TK_SIDEBAR,
    CTAG,
    FONT_FAMILY,
)
from ui.helpers import _ts

_CONSOLE_MAX_LINES = 4000


class ConsoleViewMixin:
    """Mixin that adds the Console view to App."""

    def _build_console_view(self) -> None:
        view = ctk.CTkFrame(self._content, fg_color=BG_WIN, corner_radius=0)
        self._views["console"] = view

        # Top bar
        bar = ctk.CTkFrame(view, height=44, fg_color=BG_BAR, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="MorphAI Console",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TXT).pack(side="left", padx=16)

        for label, cmd in [
            ("Save Log", self._save_log),
            ("Copy",     self._copy_log),
            ("Clear",    self._clear_console),
        ]:
            ctk.CTkButton(
                bar, text=label, width=72, height=26,
                fg_color=BG_INPUT, border_width=1, border_color=BORDER,
                text_color=TXT, hover_color=BORDER,
                corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13), command=cmd,
            ).pack(side="right", padx=(4, 6), pady=9)

        # tk.Text console
        cf = tk.Frame(view, bg=BG_CONSOLE)
        cf.pack(fill="both", expand=True)
        cf.rowconfigure(0, weight=1)
        cf.columnconfigure(0, weight=1)

        self._console = tk.Text(
            cf,
            bg=BG_CONSOLE, fg=TXT_CONSOLE,
            font=("Consolas", 11),
            bd=0, highlightthickness=0, relief="flat",
            wrap="word", state="disabled",
            padx=14, pady=10, spacing1=2, spacing3=2,
            insertbackground=_TK_TXT, selectbackground=_TK_ACCENT,
        )
        self._console.grid(row=0, column=0, sticky="nsew")

        sb = tk.Scrollbar(cf, command=self._console.yview,
                          bg=_TK_SIDEBAR, troughcolor=BG_CONSOLE,
                          bd=0, highlightthickness=0, width=10)
        sb.grid(row=0, column=1, sticky="ns")
        self._console.configure(yscrollcommand=sb.set)

        # Colour tags
        for key, fg in CTAG.items():
            w = "bold" if key in ("STEP", "BEST", "OK") else "normal"
            self._console.tag_configure(key, foreground=fg,
                                        font=("Consolas", 11, w))
        self._console.tag_configure("TS", foreground=_TK_MUTED,
                                    font=("Consolas", 10))

    def _clog(self, text: str, level: str = "INFO") -> None:
        self._console.configure(state="normal")
        self._console.insert("end", f"[{_ts()}] ", "TS")
        self._console.insert("end", text + "\n", level)
        line_count = int(self._console.index("end-1c").split(".")[0])
        if line_count > _CONSOLE_MAX_LINES:
            self._console.delete("1.0", f"{line_count - _CONSOLE_MAX_LINES}.0")
        self._console.configure(state="disabled")
        self._console.see("end")

    def _clear_console(self) -> None:
        self._console.configure(state="normal")
        self._console.delete("1.0", "end")
        self._console.configure(state="disabled")

    def _copy_log(self) -> None:
        self._console.configure(state="normal")
        text = self._console.get("1.0", "end")
        self._console.configure(state="disabled")
        self.clipboard_clear()
        self.clipboard_append(text)

    def _save_log(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self._console.configure(state="normal")
            text = self._console.get("1.0", "end")
            self._console.configure(state="disabled")
            Path(path).write_text(text, encoding="utf-8")
