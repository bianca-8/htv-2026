"""Branch by combining approaches the kept papers actually name.

A branch exists only when each piece has a verbatim sentence in a kept paper.
The desk then searches that pair. It does not invent a paper, and it does not
carry out the method. An empty index result is a gap in the search. A hit means
the combination is already published.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sources import (
    CS_CATEGORIES,
    Paper,
    SearchPlan,
    _mentions,
    _sentence_in,
    _stem,
    _unique,
    abstract_quotes,
)


@dataclass(frozen=True)
class Approach:
    approach_id: str
    name: str
    pattern: re.Pattern
    query_term: str
    side: str  # "attack" or "defense"; used only to order pairs

    def matches(self, paper: Paper) -> bool:
        return self.pattern.search(f"{paper.title}\n{paper.abstract}") is not None


# High-level names only. A paper has to use the word. The branch never turns
# the name into a procedure.
APPROACHES: tuple[Approach, ...] = (
    Approach("defense", "Defense", re.compile(r"\bdefen[cs]", re.I), "defense", "defense"),
    Approach("alignment", "Safety alignment", re.compile(r"\balignment\b|safety-aligned|safety aligned", re.I), "alignment", "defense"),
    Approach("refusal", "Refusal", re.compile(r"\brefusals?\b", re.I), "refusal", "defense"),
    Approach("proactive", "Proactive defense", re.compile(r"\bproactive\b", re.I), "proactive", "defense"),
    Approach("persona", "Persona prompts", re.compile(r"\bpersonas?\b", re.I), "persona", "attack"),
    Approach("adaptive", "Adaptive attacks", re.compile(r"\badaptive\b", re.I), "adaptive", "attack"),
    Approach("multilingual", "Multilingual prompts", re.compile(r"\bmultilingual\b", re.I), "multilingual", "attack"),
    Approach("optimization", "Optimization", re.compile(r"\boptimi[sz]", re.I), "optimization", "attack"),
    Approach("multiturn", "Multi-turn attacks", re.compile(r"multi-turn|multiturn", re.I), "multi-turn", "attack"),
    Approach("reinforcement", "Reinforcement learning", re.compile(r"\breinforcement learning\b|\brl-jailbreak", re.I), "reinforcement", "attack"),
    Approach("genetic", "Genetic search", re.compile(r"\bgenetic\b", re.I), "genetic", "attack"),
    Approach("indirect", "Indirect prompts", re.compile(r"\bindirect\b", re.I), "indirect", "attack"),
)

BY_APPROACH = {item.approach_id: item for item in APPROACHES}


@dataclass
class Citation:
    approach_id: str
    approach_name: str
    paper_id: str
    paper_title: str
    link: str
    quote: str
    source: str

    def as_dict(self) -> dict:
        return {
            "approach_id": self.approach_id,
            "approach_name": self.approach_name,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "link": self.link,
            "quote": self.quote,
            "source": self.source,
        }


@dataclass
class Branch:
    branch_id: str
    title: str
    kind: str  # "split" needs a search; "joined" is already in one kept paper
    approach_ids: list[str]
    citations: list[Citation]
    plan: SearchPlan | None = None
    status: str = "queued"
    reading: str = ""
    query: str = ""
    found: list[Paper] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.branch_id,
            "title": self.title,
            "kind": self.kind,
            "status": self.status,
            "reading": self.reading,
            "query": self.query or (self.plan.query if self.plan else ""),
            "citations": [citation.as_dict() for citation in self.citations],
            "papers": [paper.as_dict() for paper in self.found],
        }


def plan_branches(papers: list[Paper], plan: SearchPlan, *, limit: int = 6) -> list[Branch]:
    """Pairs of approaches named in the abstracts.

    Pairs that never share a paper are searched. One pair that already shares
    a paper is kept as a contrast, with no extra search. The list is capped.
    """
    available = []
    for approach in APPROACHES:
        if _stem(approach.query_term) in {_stem(word) for word in plan.excluded}:
            continue
        citations = [_cite(paper, approach) for paper in papers]
        citations = [citation for citation in citations if citation is not None]
        citations.sort(key=lambda citation: 0 if approach.pattern.search(citation.paper_title) else 1)
        if citations:
            available.append((approach, citations))
    if len(available) < 2:
        return []

    pairs: list[tuple[Approach, Approach, list[Citation], list[Citation], list[Paper]]] = []
    for index, (left, left_cites) in enumerate(available):
        for right, right_cites in available[index + 1 :]:
            together = [paper for paper in papers if left.matches(paper) and right.matches(paper)]
            pairs.append((left, right, left_cites, right_cites, together))
    pairs.sort(
        key=lambda item: (
            0 if not item[4] else 1,
            0 if item[0].side != item[1].side else 1,
            min(len(item[2]), len(item[3])),
            item[0].name,
            item[1].name,
        )
    )

    chosen: list[Branch] = []
    splits = [item for item in pairs if not item[4]]
    joined = [item for item in pairs if item[4]]
    for item in splits:
        if len(chosen) >= max(limit - (1 if joined else 0), 1):
            break
        chosen.append(_make_branch(item, plan, kind="split"))
    if joined and len(chosen) < limit:
        chosen.append(_make_branch(joined[0], plan, kind="joined"))
    return chosen


def still_grounded(branch: Branch, dropped: set[str]) -> bool:
    for approach_id in branch.approach_ids:
        if not any(citation.approach_id == approach_id and citation.paper_id not in dropped for citation in branch.citations):
            return False
    return True


def judge_split(branch: Branch, papers: list[Paper], *, query: str, origin: str) -> Branch:
    """Keep a found paper only when its own abstract uses every required word."""
    required = list(branch.plan.terms) if branch.plan else []
    excluded = list(branch.plan.excluded) if branch.plan else []
    matched: list[Paper] = []
    for paper in papers:
        if any(_mentions(paper.title, term) for term in excluded):
            continue
        blob = f"{paper.title}\n{paper.abstract}"
        if required and not all(_mentions(blob, term) for term in required):
            continue
        quotes = abstract_quotes(paper, required, limit=2)
        paper.quotes = [quote for quote in quotes if any(_mentions(quote.text, term) for term in required)][:1]
        matched.append(paper)
    names = _names(branch)
    branch.query = branch.plan.query if branch.plan else query
    branch.found = matched[:3]
    where = "arXiv" if origin == "arxiv" else "The fallback index"
    if matched:
        branch.status = "published"
        branch.reading = (
            f"{where} returned {len(matched)} paper{'s' if len(matched) != 1 else ''} whose title or abstract uses "
            f"{names}. This combination is already published. I am not calling it a gap."
        )
    else:
        branch.status = "gap"
        branch.reading = (
            f"{where} returned no paper whose title or abstract uses {names}. "
            "That empty result is the gap in this search. I did not invent a paper to fill it."
        )
    return branch


def judge_joined(branch: Branch) -> Branch:
    names = _names(branch)
    titles = list(dict.fromkeys(citation.paper_title for citation in branch.citations))
    where = f"“{titles[0]}”" if len(titles) == 1 else "the papers above"
    branch.status = "joined"
    branch.reading = (
        f"{names.capitalize()} already appear together in {where}. "
        "I did not search again, and I am not calling this pair a gap."
    )
    return branch


def _names(branch: Branch) -> str:
    labels = []
    for approach_id in branch.approach_ids:
        approach = BY_APPROACH[approach_id]
        if approach.name not in labels:
            labels.append(approach.name.lower())
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return " and ".join(labels)


def _make_branch(
    item: tuple[Approach, Approach, list[Citation], list[Citation], list[Paper]],
    plan: SearchPlan,
    *,
    kind: str,
) -> Branch:
    left, right, left_cites, right_cites, together = item
    if kind == "joined" and together:
        host = together[0]
        left_cites = [_cite(host, left) or left_cites[0]]
        right_cites = [_cite(host, right) or right_cites[0]]
    terms = _unique([*plan.terms, left.query_term, right.query_term])
    parts = [f'all:"{term}"' if " " in term else f"all:{term}" for term in terms]
    query = f"{CS_CATEGORIES} AND " + " AND ".join(parts)
    search = SearchPlan(
        query=query,
        notes=[],
        raw_direction=plan.raw_direction,
        terms=terms,
        excluded=list(plan.excluded),
    )
    return Branch(
        branch_id=f"{left.approach_id}__{right.approach_id}",
        title=f"{left.name} combined with {right.name}",
        kind=kind,
        approach_ids=[left.approach_id, right.approach_id],
        citations=[left_cites[0], right_cites[0]],
        plan=search if kind == "split" else None,
        query="" if kind == "joined" else query,
    )


def _cite(paper: Paper, approach: Approach) -> Citation | None:
    sentence, source = _verbatim_sentence(paper, approach.pattern)
    if sentence is None:
        return None
    return Citation(
        approach_id=approach.approach_id,
        approach_name=approach.name,
        paper_id=paper.paper_id,
        paper_title=paper.title,
        link=paper.link,
        quote=sentence,
        source=source,
    )


def _verbatim_sentence(paper: Paper, pattern: re.Pattern) -> tuple[str, str] | tuple[None, None]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", paper.abstract).strip())
    for part in parts:
        part = part.strip()
        if len(part) < 12 or not pattern.search(part):
            continue
        if _sentence_in(part, paper.abstract):
            return part, "abstract"
    title = re.sub(r"\s+", " ", paper.title).strip()
    if pattern.search(title) and _sentence_in(title, paper.title):
        return title, "title"
    return None, None
