from __future__ import annotations

import os
from typing import List

import requests


def glosses_to_sentence(glosses: List[str]) -> str:
    """
    Uses Ollama if OLLAMA_MODEL is set, otherwise returns a simple join.
    """
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        return " ".join(glosses)

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    gloss_str = ", ".join(glosses)
    prompt = (
        "Task: Convert ASL gloss keywords (may be out of order) into ONE natural English sentence.\n"
        "Rules: reorder if needed; add missing function words; don't invent facts; if WH-word present output a WH-question; "
        "if ends with '?' output a question; output ONLY the sentence.\n"
        "Example: AUNT, YESTERDAY, GO, WHERE? -> Where did your aunt go yesterday?\n\n"
        f"Glosses: {gloss_str}\n"
        "Sentence:"
    )
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip() or " ".join(glosses)
    except Exception:
        return " ".join(glosses)

