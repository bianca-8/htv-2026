"""Ask Gemini to write one experiment for a branch, then run that file.

The model sees the person's direction and the approved abstracts. The script
it returns is what gets executed. This module does not invent a result if the
model is unavailable, and it will not run code that is an attack.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi

from .data_ground import ground_papers
from .sources import Paper

_ROOT = Path(__file__).resolve().parents[1]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

_SYSTEM = """You are the experiment-writing half of a research desk.
The person already chose papers. You write one Python script that follows a method those abstracts describe.
The script is the experiment. It is not a paper search.

Data, in this order:
- If the prompt names a public dataset, load that dataset with sklearn.datasets (load_breast_cancer or fetch_openml). Do not substitute a different dataset.
- If the prompt gives a synthesis procedure, use only those numbers. Both classes are standard normal unless the prompt states a separation.
- If the prompt lists a GitHub repository, treat it as the paper's code. Reimplement the method it describes. Do not clone, download, or run that repository.
- If the prompt says no table was named, do not invent a benchmark. Implement the training and test procedure the papers describe, and run it. A small input is allowed when the method needs one and the paper names no table.
- Do not invent a sample size, a feature count, a class balance, or a mean shift that the prompt did not state.

Other rules:
- numpy and scikit-learn are allowed. No urllib, no requests, no subprocess, no sockets.
- Assign EVIDENCE to the paper sentence given in the prompt, copied exactly.
- Print one JSON object as the last line, with keys "summary", "measurements", and "data".
- "data" must say whether the rows were the named dataset or the stated synthesis.
- The summary must describe only what the script computed.
- Return only a python code block.
- don't have ``` at the beginning or end of the code block
"""

_BLOCKED = re.compile(
    r"jailbreak|exploit|reverse shell|payload|os\.system|subprocess|socket\.|urllib|requests\.|prompt injection",
    re.I,
)


def api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            if key.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                return raw.strip().strip('"').strip("'")
    return ""


def papers_brief(papers: list[Paper], *, limit: int = 8) -> str:
    blocks = []
    for paper in papers[:limit]:
        abstract = " ".join(paper.abstract.split())
        blocks.append(f"ID: {paper.paper_id}\nTitle: {paper.title}\nAbstract: {abstract}\nExcerpts: {json.dumps([q.as_dict() for q in paper.quotes])}")
    return "\n\n".join(blocks)


def extract_script(text: str, *, evidence: str = "") -> str:
    cleaned = text.strip()
    if cleaned.upper() == "REFUSE":
        raise RuntimeError("The model refused to write that experiment.")
    if cleaned.upper() == "INSUFFICIENT":
        raise RuntimeError(
            "The papers do not state a dataset or a complete synthesis procedure. No table was invented."
        )
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    script = (match.group(1) if match else text).strip()
    if not script or "print" not in script:
        preview = " ".join(text.split())[:160]
        detail = f" It returned: {preview}" if preview else ""
        raise RuntimeError(f"The model did not return a runnable script.{detail}")
    if evidence and " ".join(evidence.split()) not in " ".join(script.split()):
        raise RuntimeError("The script did not keep the paper sentence that states the data, so it was not run.")
    if _BLOCKED.search(script):
        raise RuntimeError("The script was not run. It asked for network access or an attack, which this desk does not execute.")
    return script + ("\n" if not script.endswith("\n") else "")


def writer_name() -> str:
    choice = os.environ.get("EXPERIMENT_WRITER", "ollama").strip().lower()
    if choice == "gemini":
        return MODEL
    return OLLAMA_MODEL


def _prompt(
    direction: str,
    papers: list[Paper],
    table_path: str | None = None,
    repos: list[dict] | None = None,
) -> tuple[str, str]:
    grounding = ground_papers(papers)
    repos = repos or []
    repo_lines = "\n".join(
        f"- {item['url']} (from {item['paper_title']})" for item in repos
    )
    if grounding is not None and grounding.kind == "public":
        data_lines = (
            f"Load the public dataset loader id {grounding.loader!r} ({grounding.dataset_name}).\n"
            "Use fetch_openml(name=that id, version=1, as_frame=False, parser='liac-arff') unless the id is breast_cancer, "
            "in which case use load_breast_cancer.\n"
            "Keep the smaller class as the positive label only when the table is binary. "
            "If it is not binary, use the smallest class as the positive label and say so."
        )
        evidence = grounding.evidence
    elif grounding is not None:
        data_lines = (
            f"Synthesize {grounding.n_rows} rows and {grounding.n_features} features. "
            f"The smaller class is {grounding.positive_rate:.4f} of the rows. "
            "Draw every row from a standard normal. Do not shift one class."
        )
        evidence = grounding.evidence
    else:
        data_lines = (
            "No public table was named. Do not invent a benchmark dataset.\n"
            "Implement the training and test procedure described in the papers and run it. "
            "A method can be a model plus a test, with no external dataset. "
            "If the method needs input and the paper does not name a table, use a small input and say that the paper did not name one."
        )
        sentence = _method_sentence(papers)
        if sentence:
            data_lines += f"\nA sentence from the papers, for context: {sentence}"
        evidence = ""
    if repo_lines:
        data_lines += (
            "\nThe papers cite these GitHub repositories. Follow that method. "
            "Do not clone or execute the repository.\n"
            + repo_lines
        )
    prompt = (
        f"Direction:\n{direction.strip()}\n\n"
        "Approved papers:\n"
        f"{papers_brief(papers)}\n\n"
        "What to implement:\n"
        f"{data_lines}\n"
        + (
            f"EVIDENCE = {evidence!r}\n"
            "Copy that sentence into the script as EVIDENCE.\n\n"
            if evidence else ""
        )
        + (
            f"If you create rows, write them to {table_path} as CSV. "
            "Columns are f0, f1, and so on, then label, then split. "
            "split is fit, validation, or test. Create the parent directory. Do not print the rows.\n\n"
            if table_path else ""
        )
        + "Write the experiment script now."
    )
    return prompt, evidence


def _method_sentence(papers: list[Paper]) -> str:
    for paper in papers:
        for sentence in re.split(r"(?<=[.])\s+", " ".join(paper.abstract.split())):
            if len(sentence) >= 40:
                return sentence.strip()
    if papers and papers[0].abstract.strip():
        return " ".join(papers[0].abstract.split())[:240]
    return papers[0].title if papers else ""


def write_experiment(
    direction: str,
    papers: list[Paper],
    table_path: str | None = None,
    repos: list[dict] | None = None,
    on_delta=None,
) -> str:
    choice = os.environ.get("EXPERIMENT_WRITER", "ollama").strip().lower()
    if choice == "gemini":
        return _write_gemini(direction, papers, table_path, repos, on_delta)
    return _write_ollama(direction, papers, table_path, repos, on_delta)


def _write_ollama(
    direction: str,
    papers: list[Paper],
    table_path: str | None = None,
    repos: list[dict] | None = None,
    on_delta=None,
    *,
    again: bool = False,
) -> str:
    prompt, evidence = _prompt(direction, papers, table_path, repos)
    prompt += (
        "\n\nReturn one python code block and nothing else. "
        "No comments. No docstrings. Use as few words as possible."
    )
    if again:
        prompt += (
            "\n\nYour previous reply was not a Python script. "
            "Return one python code block only. The script must call print."
        )
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "stream": True,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0,
            "num_predict": 3500,
            "num_ctx": 16384,
            "num_thread": 4,
            "stop": ["\n```"],
        },
    }).encode()
    request = urllib.request.Request(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                piece = _ollama_delta(line)
                if not piece:
                    continue
                parts.append(piece)
    except TimeoutError as exc:
        raise RuntimeError(
            f"{OLLAMA_MODEL} did not finish the script within 10 minutes. No result was invented."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {OLLAMA_URL}. Start it and pull {OLLAMA_MODEL}. No result was invented."
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Ollama returned {exc.code} for {OLLAMA_MODEL}. {detail}") from exc
    text = "".join(parts).strip()
    if text.count("```") % 2 == 1:
        text += "\n```"
    if not text:
        raise RuntimeError(f"{OLLAMA_MODEL} returned no text.")
    try:
        script = extract_script(text, evidence=evidence)
    except RuntimeError as exc:
        if again or "runnable script" not in str(exc):
            raise
        return _write_ollama(direction, papers, table_path, repos, on_delta, again=True)
    script = _without_comments(script)
    if evidence and " ".join(evidence.split()) not in " ".join(script.split()):
        script = f"EVIDENCE = {evidence!r}\n{script}"
    if on_delta:
        on_delta(script)
    return script


def _without_comments(script: str) -> str:
    """Drop comments and one-line docstrings. Leave strings that contain a hash."""
    kept = []
    for line in script.splitlines():
        code = _drop_hash_comment(line).rstrip()
        if re.fullmatch(r'\s*(?:"""[\s\S]*"""|\'\'\'[\s\S]*\'\'\')\s*', code):
            continue
        if code.strip():
            kept.append(code)
    return "\n".join(kept) + ("\n" if kept else "")


def _drop_hash_comment(line: str) -> str:
    quote = ""
    triple = False
    i = 0
    while i < len(line):
        if quote:
            if triple and line.startswith(quote * 3, i):
                i += 3
                quote = ""
                triple = False
                continue
            if not triple and line[i] == "\\":
                i += 2
                continue
            if not triple and line[i] == quote:
                quote = ""
            i += 1
            continue
        if line.startswith('"""', i) or line.startswith("'''", i):
            quote = line[i]
            triple = True
            i += 3
            continue
        if line[i] in "'\"":
            quote = line[i]
            i += 1
            continue
        if line[i] == "#":
            return line[:i]
        i += 1
    return line


def _ollama_delta(line: bytes) -> str:
    raw = line.decode("utf-8", errors="replace").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return ((payload.get("message") or {}).get("content") or "")


def _write_gemini(
    direction: str,
    papers: list[Paper],
    table_path: str | None = None,
    repos: list[dict] | None = None,
    on_delta=None,
) -> str:
    key = api_key()
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY to let the agent write the experiment. No result was invented.")
    prompt, evidence = _prompt(direction, papers, table_path, repos)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:streamGenerateContent?alt=sse"
    body = json.dumps({
        "system_instruction": {"parts": [{"text": _SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=600, context=_ssl()) as response:
            while True:
                line = response.readline()
                if not line:
                    break
                piece = _gemini_delta(line)
                if not piece:
                    continue
                parts.append(piece)
                if on_delta:
                    on_delta("".join(parts))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"Gemini returned {exc.code}. {detail}") from exc
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("Gemini returned no text.")
    return extract_script(text, evidence=evidence)


def run_model_experiment(
    direction: str,
    papers: list[Paper],
    table_path: str | None = None,
    repos: list[dict] | None = None,
    on_delta=None,
) -> dict:
    script = write_experiment(direction, papers, table_path, repos, on_delta)
    import asyncio
    import uuid
    from .runner import execute
    folder = _ROOT / "experiments" / ("legacy-" + uuid.uuid4().hex)
    payload = asyncio.run(execute(script, folder, lambda: False))
    stdout = (folder / "stdout.txt").read_text().strip()
    summary = stdout
    measurements: dict | list | None = None
    last = stdout.splitlines()[-1] if stdout else ""
    try:
        parsed = json.loads(last)
        if isinstance(parsed, dict):
            summary = str(parsed.get("summary") or stdout)
            measurements = parsed.get("measurements")
    except json.JSONDecodeError:
        parsed = None
    return {
        "reading": summary,
        "measurements": measurements,
        "stdout": stdout,
        "code": script,
        "rows": [],
    }


def _gemini_delta(line: bytes) -> str:
    raw = line.decode("utf-8", errors="replace").strip()
    if raw.startswith("data:"):
        raw = raw[5:].strip()
    if not raw or raw == "[DONE]":
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not payload.get("candidates"):
        return ""
    parts = ((payload["candidates"][0].get("content") or {}).get("parts") or [])
    return "".join(part.get("text", "") for part in parts)


def _ssl():
    import ssl
    return ssl.create_default_context(cafile=certifi.where())
