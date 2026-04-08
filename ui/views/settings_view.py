"""
ui/views/settings_view.py — Settings view mixin.

Provides _build_settings_view and _toggle_theme for the App class.
"""
from __future__ import annotations

import customtkinter as ctk

from ui.constants import (
    BG_WIN, BG_BAR, BG_INPUT,
    BORDER, TXT, TXT_MUTED,
    FONT_FAMILY,
)
from ui.helpers import _sec_label


class SettingsViewMixin:
    """Mixin that adds the Settings view to App."""

    def _build_settings_view(self) -> None:
        view = ctk.CTkFrame(self._content, fg_color=BG_WIN, corner_radius=0)
        self._views["settings"] = view

        bar = ctk.CTkFrame(view, height=44, fg_color=BG_BAR, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        ctk.CTkLabel(bar, text="Settings",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TXT).pack(side="left", padx=16)

        scroll = ctk.CTkScrollableFrame(view, fg_color=BG_WIN,
                                        scrollbar_button_color=BORDER)
        scroll.pack(fill="both", expand=True)

        _sec_label(scroll, "APPEARANCE", padx=20, pady=(22, 8))
        theme_card = ctk.CTkFrame(scroll, fg_color=("#f4f6fb", "#334155"),
                                  corner_radius=8, border_width=1,
                                  border_color=BORDER)
        theme_card.pack(fill="x", padx=20)

        row = ctk.CTkFrame(theme_card, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(row, text="Theme",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
                     text_color=TXT, width=120, anchor="w").pack(side="left")
        ctk.CTkLabel(row, text="Switch between dark and light interface mode.",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                     text_color=TXT_MUTED).pack(side="left", padx=(0, 16))

        self._theme_btn = ctk.CTkButton(
            row, text="Switch to Light Mode",
            width=160, height=32, corner_radius=6,
            fg_color=BG_INPUT, hover_color=BORDER,
            border_width=1, border_color=BORDER,
            text_color=TXT, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            command=self._toggle_theme,
        )
        self._theme_btn.pack(side="right")

        _sec_label(scroll, "ABOUT", padx=20, pady=(22, 8))
        about_card = ctk.CTkFrame(scroll, fg_color=("#f4f6fb", "#334155"),
                                  corner_radius=8, border_width=1,
                                  border_color=BORDER)
        about_card.pack(fill="x", padx=20, pady=(0, 20))

        for line, muted in [
            ("MorphAI — Adaptive Preprocessing Agent", False),
            ("Automatically profiles, cleans, and evaluates ML preprocessing pipelines.", True),
        ]:
            ctk.CTkLabel(
                about_card, text=line,
                font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold" if not muted else "normal"),
                text_color=TXT if not muted else TXT_MUTED,
                anchor="w",
            ).pack(anchor="w", padx=16, pady=(10 if not muted else 2, 2))
        ctk.CTkFrame(about_card, height=10, fg_color="transparent").pack()

    def _toggle_theme(self) -> None:
        if self._theme_mode == "dark":
            self._theme_mode = "light"
            ctk.set_appearance_mode("light")
            self._theme_btn.configure(text="Switch to Dark Mode")
            if hasattr(self, "_set_os_icon"): self._set_os_icon(False)
        else:
            self._theme_mode = "dark"
            ctk.set_appearance_mode("dark")
            self._theme_btn.configure(text="Switch to Light Mode")
            if hasattr(self, "_set_os_icon"): self._set_os_icon(True)
