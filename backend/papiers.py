"""Search the person's Papiers library. Nothing here invents a paper.

The CLI is read-only. A missing install or a missing token does not stop the
arXiv search; the desk says Papiers was not used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .sources import Paper, Quote

_ROOT = Path(__file__).resolve().parents[1]
_UNAVAILABLE = ""


def search_library(direction: str, *, limit: int = 6) -> tuple[list[Paper], list[str]]:
    """Return saved documents for this direction, plus notes about the attempt."""
    global _UNAVAILABLE
    if _UNAVAILABLE:
        return [], []
    binary = shutil.which("papiers")
    if binary is None:
        _UNAVAILABLE = "missing"
        return [], [
            "Papiers is not installed, so this search used the public index only. "
            "Install the CLI with npm install -g papiers and set PAPIERS_TOKEN to include your library."
        ]
    query = " ".join(direction.split())[:300].strip()
    if not query:
        return [], []
    payload, error = _cli(binary, ["search", query, "--limit", str(max(1, min(limit, 8)))])
    if error:
        return [], [error]
    if not isinstance(payload, dict):
        return [], ["Papiers did not return a library search. Nothing was invented."]
    signal = ((payload.get("quality") or {}).get("signal") or "").lower()
    results = [item for item in payload.get("results") or [] if isinstance(item, dict)]
    documents = [item for item in results if item.get("kind") == "document"][:limit]
    if signal in {"empty", "low"} and not documents:
        return [], [f"Papiers searched your library for “{query}” and had no document to add."]
    reads = _read_documents(binary, [str(item.get("id") or "") for item in documents])
    papers = [_paper(item, reads.get(str(item.get("id") or ""))) for item in documents]
    papers = [paper for paper in papers if paper is not None]
    notes = [f"I searched your Papiers library for “{query}”."]
    if not papers:
        notes.append("Papiers returned no saved document with a title. I did not invent one.")
    return papers, notes


def _read_documents(binary: str, ids: list[str]) -> dict[str, dict]:
    ids = [item for item in ids if item.startswith("doc_")][:4]
    if not ids:
        return {}
    payload, error = _cli(
        binary,
        ["read", *ids, "--format", "plain", "--max-chars-per-item", "3500"],
        timeout=45,
    )
    if error or not isinstance(payload, dict):
        return {}
    found: dict[str, dict] = {}
    for item in payload.get("items") or []:
        if isinstance(item, dict) and item.get("ok") is True and item.get("id"):
            found[str(item["id"])] = item
    return found


def _paper(result: dict, read: dict | None) -> Paper | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    if read and isinstance(read.get("metadata"), dict):
        metadata = {**metadata, **read["metadata"]}
    title = _text(result.get("title")) or _text(metadata.get("title"))
    if not title:
        return None
    content = _text((read or {}).get("content"))
    snippets = []
    for evidence in result.get("evidence") or []:
        if isinstance(evidence, dict):
            snippet = _text(evidence.get("snippet"))
            if snippet:
                snippets.append((str(evidence.get("kind") or "passage"), snippet))
    abstract = content or "\n\n".join(snippet for _, snippet in snippets)
    if not abstract:
        return None
    identifiers = metadata.get("identifiers") if isinstance(metadata.get("identifiers"), dict) else {}
    arxiv_id = _text(identifiers.get("arxivId")).removeprefix("arXiv:")
    source = result.get("source") if isinstance(result.get("source"), dict) else {}
    if read and isinstance(read.get("source"), dict):
        source = read["source"]
    link = _text(source.get("url"))
    if not link and arxiv_id:
        link = f"https://arxiv.org/abs/{arxiv_id}"
    doi = _text(identifiers.get("doi"))
    if not link and doi:
        link = f"https://doi.org/{doi}"
    authors = metadata.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    authors = [_text(author) for author in authors if _text(author)]
    published = _text(metadata.get("publishedDate")) or _text(metadata.get("publishedYear"))
    quotes = []
    collapsed = " ".join(abstract.split())
    for kind, snippet in snippets:
        if " ".join(snippet.split()) not in collapsed:
            continue
        quotes.append(Quote(text=snippet, source="papiers", section=kind))
    return Paper(
        paper_id=arxiv_id or str(result.get("id") or title),
        title=title,
        authors=authors,
        published=published[:10],
        abstract=abstract,
        link=link,
        pdf_url=link if _text(source.get("family")) == "pdf" else None,
        categories=["Papiers"],
        venue=_text(metadata.get("venue")) or None,
        origin="papiers",
        quotes=quotes[:3],
    )


def _cli(binary: str, args: list[str], *, timeout: int = 30) -> tuple[dict | None, str]:
    try:
        completed = subprocess.run(
            [binary, *args, "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, "Papiers did not answer. This search continued with the public index. Nothing was invented."
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        message = detail[-1] if detail else f"Papiers exited {completed.returncode}."
        if _auth_problem(message):
            return None, (
                "Papiers is not signed in, so this search used the public index only. "
                "Create a token in Papiers settings and set PAPIERS_TOKEN."
            )
        return None, f"Papiers could not search the library ({message}). Nothing was invented."
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        return None, "Papiers returned text that was not a library record. Nothing was invented."
    return payload if isinstance(payload, dict) else None, ""


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("PAPIERS_TOKEN", "").strip():
        return env
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return env
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        if key.strip() == "PAPIERS_TOKEN":
            token = raw.strip().strip('"').strip("'")
            if token:
                env["PAPIERS_TOKEN"] = token
            break
    return env


def _auth_problem(message: str) -> bool:
    lowered = message.lower()
    return any(word in lowered for word in ("token", "auth", "unauthorized", "sign in", "logged"))


def _text(value: object) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    return " ".join(str(value).split()).strip() if isinstance(value, str) else ""
