"""
ui/helpers.py — Shared UI utility functions used across all views.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from ui.constants import BG_CARD, BORDER, TXT_MUTED, FONT_FAMILY


def _open_file(path: Path) -> None:
    try:
        if hasattr(os, "startfile"):            # Windows
            os.startfile(str(path))             # type: ignore[attr-defined]
        else:                                   # macOS / Linux
            cmd = "open" if sys.platform == "darwin" else "xdg-open"
            os.system(f'{cmd} "{path}"')
    except Exception as exc:
        print(f"[UI] Cannot open: {exc}")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _card(parent, **kw) -> ctk.CTkFrame:
    defaults = dict(fg_color=BG_CARD, corner_radius=12,
                    border_width=1, border_color=BORDER)
    defaults.update(kw)
    return ctk.CTkFrame(parent, **defaults)


def _hsep(parent, pady=(4, 4)) -> None:
    ctk.CTkFrame(parent, height=1, fg_color=BORDER, corner_radius=0).pack(
        fill="x", pady=pady if isinstance(pady, tuple) else (pady, pady)
    )


def _sec_label(parent, text: str, padx: int = 0, pady=(12, 6)) -> None:
    ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
        text_color=TXT_MUTED,
    ).pack(anchor="w", padx=padx, pady=pady)


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
