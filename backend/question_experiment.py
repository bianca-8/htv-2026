"""Write and run a script for a direction the classifier bench does not fit.

The script is built from the approved papers and the search words. It only
counts which of those papers use the words. It does not invent a result.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .sources import Paper

_ROOT = Path(__file__).resolve().parents[1]


def question_source(terms: list[str], papers: list[Paper]) -> str:
    records = [
        {"title": paper.title, "abstract": paper.abstract, "link": paper.link}
        for paper in papers
    ]
    return (
        "import json\n"
        "import re\n"
        f"TERMS = {list(terms)!r}\n"
        f"PAPERS = json.loads({json.dumps(records)!r})\n"
        "\n"
        "def mentions(text, term):\n"
        "    stem = term\n"
        "    for suffix in ('ing', 'ers', 'er', 'es', 's'):\n"
        "        if term.endswith(suffix) and len(term) - len(suffix) >= 5:\n"
        "            stem = term[: -len(suffix)]\n"
        "            break\n"
        "    return re.search(r'\\b' + re.escape(stem) + r'\\w*', text, flags=re.I) is not None\n"
        "\n"
        "rows = []\n"
        "for paper in PAPERS:\n"
        "    blob = paper['title'] + '\\n' + paper['abstract']\n"
        "    hits = [term for term in TERMS if mentions(blob, term)]\n"
        "    rows.append({\n"
        "        'title': paper['title'],\n"
        "        'link': paper['link'],\n"
        "        'hits': hits,\n"
        "        'all': len(TERMS) > 0 and len(hits) == len(TERMS),\n"
        "    })\n"
        "matched = [row for row in rows if row['all']]\n"
        "print(json.dumps({'matched': len(matched), 'total': len(rows), 'terms': TERMS, 'rows': rows}))\n"
    )


def run_question_script(terms: list[str], papers: list[Paper]) -> tuple[dict, str]:
    source = question_source(terms, papers)
    folder = _ROOT / "experiments"
    folder.mkdir(exist_ok=True)
    script = folder / "question.py"
    script.write_text(source)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no output").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"exit {completed.returncode}")
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    matched = int(payload["matched"])
    total = int(payload["total"])
    shown = " and ".join(terms) if terms else "the question"
    if total == 0:
        reading = "No approved paper was left to check."
    elif matched == 0:
        reading = f"None of the {total} approved papers use {shown} together in the title or abstract."
    elif matched == total:
        reading = f"All {total} approved papers use {shown} in the title or abstract."
    else:
        reading = f"{matched} of {total} approved papers use {shown} together in the title or abstract."
    payload["reading"] = reading
    payload["code"] = source
    return payload, source
