from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def run_worker(modality: str, q, args: tuple, kwargs: dict) -> None:
    root = Path(__file__).resolve().parent.parent
    try:
        os.chdir(root)
    except Exception:
        pass
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        if modality == "Image":
            from ui.worker import ImageAgentWorker as Worker
        elif modality == "Audio":
            from ui.audio_worker import AudioAgentWorker as Worker
        elif modality == "Text":
            from ui.text_worker import TextAgentWorker as Worker
        else:
            from ui.worker import AgentWorker as Worker
        Worker(q, *args, **kwargs).run()
    except Exception as exc:
        try:
            q.put({"kind": "fail", "text": f"{exc}\n{traceback.format_exc()}"})
        except Exception:
            pass
