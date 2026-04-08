"""
start_ui.py — Launch the desktop assistant UI for the Preprocessing Agent.

Usage:
    python start_ui.py

Requirements:
    pip install customtkinter
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of where Python is invoked.
sys.path.insert(0, str(Path(__file__).parent))

from ui.app import main  # noqa: E402

if __name__ == "__main__":
    main()