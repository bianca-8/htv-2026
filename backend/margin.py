"""Interpret a note written in the margin.

The matcher is deliberately narrow. A sentence it does not understand is saved
and answered with what the desk can act on, rather than guessed.
"""

from __future__ import annotations

import re


def interpret(text: str) -> list[dict]:
    raw = " ".join(text.split())
    if not raw:
        return []
    hay = raw.lower()
    actions: list[dict] = []

    if re.search(r"\bpause\b|\bhold on\b|\bstop\b|\bwait\b", hay):
        actions.append({"type": "pause"})
    if re.search(r"\bresume\b|\bcontinue\b|\bcarry on\b", hay):
        actions.append({"type": "resume"})

    if re.search(r"no synthetic|don'?t invent|do not invent|without smote|no smote|no over-?sampl|stop invent", hay):
        actions.append({"type": "rule", "rule": "no_synthetic"})
    if re.search(r"no under-?sampl|don'?t discard|do not discard|keep (all )?(the )?majority", hay):
        actions.append({"type": "rule", "rule": "no_undersample"})
    if re.search(r"no class weight|don'?t reweight|do not reweight|no cost-sensitive", hay):
        actions.append({"type": "rule", "rule": "no_weight"})

    if re.search(r"no threshold|no cutoff tun|default cutoff only", hay):
        actions.append({"type": "rule", "rule": "no_threshold"})

    floor = re.search(
        r"(precision|false alarms?).{0,48}?(?:≥|>=|at or above|at least|above|over|of)\s*(0\.\d+)",
        hay,
    )
    if floor is None:
        floor = re.search(r"(0\.\d+).{0,24}?precision", hay)
    if floor and re.search(r"recall|precision|false alarm|floor", hay):
        value = float(floor.group(floor.lastindex))
        if 0 < value < 1:
            actions.append({"type": "precision_floor", "value": round(value, 2)})

    if not any(action["type"] == "precision_floor" for action in actions):
        if re.search(r"average precision|pr-?auc|auprc|precision-recall", hay):
            actions.append({"type": "metric", "metric": "average_precision"})
        elif re.search(r"\bf1\b|f-measure|f1-score", hay):
            actions.append({"type": "metric", "metric": "f1"})
        elif re.search(r"rank by accuracy|use accuracy|score by accuracy|optimi[sz]e accuracy", hay):
            actions.append({"type": "metric", "metric": "accuracy"})
        elif re.search(r"rank by recall|maximi[sz]e recall|optimi[sz]e recall|care about recall", hay):
            actions.append({"type": "metric", "metric": "recall"})

    drop = re.search(r"\b(?:drop|remove|strike|ignore)\b(?:\s+\w+){0,3}\s+(\d{4}\.\d{4,5})\b", hay)
    if drop:
        actions.append({"type": "drop_paper", "paper_id": drop.group(1)})

    if raw.lower().startswith("search ") or raw.lower().startswith("look for ") or "instead search" in hay:
        direction = re.sub(r"^(search|look for)\s+", "", raw, flags=re.I)
        direction = re.sub(r"^instead search\s+", "", direction, flags=re.I)
        actions.append({"type": "redirect", "direction": direction})

    return actions


def unhandled_reply() -> str:
    return (
        "I saved that note, and I do not yet know how to act on it. "
        "I can drop a paper, forbid synthetic samples, forbid undersampling, "
        "rank by average precision, F1, recall, or accuracy, "
        "hold a precision floor such as 0.40, pause, or start a new search."
    )
