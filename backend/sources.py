"""Retrieve real papers. Nothing in this module invents a bibliographic record.

Search order:
1. The arXiv Atom API (title, authors, abstract, categories, canonical link).
2. If arXiv is rate-limiting, OpenAlex works that have an abstract and a
   computer-science topic. The link is the arXiv abstract when OpenAlex
   recorded an arXiv id, otherwise the DOI landing page.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .http_cache import FetchError, fetch

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

CS_CATEGORIES = "(cat:cs.LG OR cat:cs.AI OR cat:cs.CL OR cat:cs.CR OR cat:cs.CY OR cat:stat.ML OR cat:cs.CV)"

# Words that describe the user's intent but make a poor arXiv conjunction.
STOPWORDS = {
    "about", "after", "again", "against", "alarm", "alarms", "also", "among",
    "approach", "approaches", "because", "before", "being", "between", "both",
    "catch", "catching", "classifier", "classifiers", "could", "data", "detect",
    "detecting", "does", "drowning", "each", "else", "false", "find", "finding",
    "from", "general", "have", "help", "helps", "into", "just", "literature",
    "looking", "method", "methods", "model", "models", "more", "most", "only",
    "other", "over", "paper", "papers", "people", "possible", "research",
    "should", "some", "such", "than", "that", "their", "them", "then", "there",
    "these", "this", "those", "through", "training", "using", "want", "what",
    "when", "where", "which", "while", "with", "without", "work", "works",
    "would", "your",
    # Instructions about the search itself, not the subject of a paper.
    "pivot", "regular", "explore", "explored", "exploring", "exists", "exist",
    "gap", "please", "instead", "toward", "towards", "topic", "area", "field",
    "focus", "specifically", "especially", "particularly", "narrow", "narrower",
    "look", "search", "searching", "relevant", "related", "something",
    "anything", "everything", "really", "like", "make", "makes", "need",
    "needs", "keep", "reading", "change", "redirect", "bench", "desk",
}

# Short tokens that are real technical terms and must not be dropped for length.
SHORT_TERMS = {"ai", "cv", "gan", "gpt", "llm", "llms", "mlm", "nlp", "rl", "svm", "vlm"}

# Used only to repair a nearby misspelling. A word is replaced when it is one
# edit from one of these and at least six letters long.
SPELLINGS = (
    "adversarial",
    "agentic",
    "agents",
    "alignment",
    "injection",
    "jailbreak",
    "jailbreaking",
    "language",
    "multimodal",
    "prompt",
    "safety",
)


@dataclass
class Quote:
    text: str
    source: str  # "abstract" or "paper-html" or "title"
    section: str | None = None

    def as_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "section": self.section}


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    link: str
    pdf_url: str | None
    categories: list[str]
    venue: str | None
    origin: str  # "arxiv" or "openalex"
    quotes: list[Quote] = field(default_factory=list)

    def blob(self) -> str:
        extra = " ".join(q.text for q in self.quotes)
        return f"{self.title}\n{self.abstract}\n{extra}"

    def as_dict(self) -> dict:
        authors = self.authors[:4]
        author_line = ", ".join(authors)
        if len(self.authors) > 4:
            author_line += ", et al."
        return {
            "id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "author_line": author_line or "Author list unavailable",
            "published": self.published,
            "year": self.published[:4] if self.published else "",
            "abstract": self.abstract,
            "link": self.link,
            "pdf_url": self.pdf_url,
            "categories": self.categories,
            "venue": self.venue,
            "origin": self.origin,
            "quotes": [q.as_dict() for q in self.quotes],
            "kept": True,
        }


@dataclass
class SearchPlan:
    query: str
    notes: list[str]
    raw_direction: str
    terms: list[str]
    excluded: list[str] = field(default_factory=list)
    # Words added by a pivot. A paper must use every one of them, on top of the standing search.
    required: list[str] = field(default_factory=list)


def _excluded_terms(text: str) -> tuple[list[str], str]:
    """Pull words the person negated, and return the text with those spans removed."""
    pattern = re.compile(r"\b(?:not|without|except)\s+([a-z][a-z0-9\-]{2,})|\bnon-([a-z][a-z0-9\-]{2,})")
    excluded: list[str] = []
    cleaned = text
    for match in pattern.finditer(text):
        word = match.group(1) or match.group(2)
        fixed = _correct(word)
        if fixed not in excluded:
            excluded.append(fixed)
        cleaned = cleaned.replace(match.group(0), " ")
    return excluded, cleaned


def _positive_terms(text: str, excluded: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    corrections: list[tuple[str, str]] = []
    terms: list[str] = []
    excluded_stems = {_stem(word) for word in excluded}
    for word in re.findall(r"[a-z][a-z0-9\-]{1,}", text):
        if word in STOPWORDS or (len(word) < 4 and word not in SHORT_TERMS):
            continue
        fixed = _correct(word)
        if fixed != word:
            corrections.append((word, fixed))
        if fixed in STOPWORDS or _stem(fixed) in excluded_stems:
            continue
        if fixed not in terms:
            terms.append(fixed)
    return terms[:6], corrections


def _correct(word: str) -> str:
    if len(word) < 6 or word in SPELLINGS or word in {"infection", "infections", "infected"}:
        return word
    best = word
    best_distance = 2
    for known in SPELLINGS:
        if abs(len(known) - len(word)) > 1:
            continue
        distance = _levenshtein(word, known)
        if distance < best_distance:
            best = known
            best_distance = distance
    return best


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char in enumerate(left, start=1):
        current = [i]
        for j, other in enumerate(right, start=1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char != other))
            )
        previous = current
    return previous[-1]


def _stem(word: str) -> str:
    for suffix in ("ing", "ers", "er", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 5:
            return word[: -len(suffix)]
    return word


def _mentions(text: str, term: str) -> bool:
    stem = _stem(term)
    return re.search(rf"\b{re.escape(stem)}\w*", text, flags=re.I) is not None


def select_relevant(papers: list[Paper], plan: SearchPlan, *, limit: int = 8) -> list[Paper]:
    """Keep papers whose own title or abstract uses the search words.

    A title that uses a ruled-out word is dropped. Papers that use more of the
    specific words rank first. Nothing is added to fill the list.
    """
    if not plan.terms:
        return papers[:limit]
    specific = [term for term in plan.terms if len(_stem(term)) >= 6]
    ranked: list[tuple[int, int, Paper]] = []
    for paper in papers:
        title = paper.title.lower()
        blob = f"{paper.title}\n{paper.abstract}".lower()
        if any(_mentions(title, term) for term in plan.excluded):
            continue
        if plan.required and not all(_mentions(blob, term) for term in plan.required):
            continue
        if specific and not any(_mentions(blob, term) for term in specific):
            continue
        if not any(_mentions(blob, term) for term in plan.terms):
            continue
        specific_hits = sum(1 for term in specific if _mentions(blob, term))
        score = 0
        for term in plan.terms:
            if _mentions(title, term):
                score += 3
            elif _mentions(blob, term):
                score += 1
        ranked.append((specific_hits, score, paper))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [paper for _, _, paper in ranked[:limit]]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def compose_direction(previous: str, pivot: str) -> str:
    """Keep the standing direction and append a pivot that builds on it.

    A pivot that already contains the standing text is used as written. A
    repeated pivot is not stacked a second time.
    """
    standing = " ".join(previous.split()).strip()
    added = " ".join(pivot.split()).strip()
    if not added:
        return standing
    if not standing or standing.lower() in added.lower():
        return added
    if added.lower() in standing.lower():
        return standing
    return f"{standing}\n\nFocus: {added}"


def plan_query(direction: str) -> SearchPlan:
    """Turn a person's direction into an auditable arXiv query.

    A “Focus:” section is a pivot. Its words are required on top of the
    standing search, so a narrower note does not throw away the earlier topic.
    """
    pieces = re.split(r"\n+\s*Focus:\s*", direction.strip(), flags=re.I)
    if len(pieces) > 1 and pieces[0].strip():
        plan = plan_query(pieces[0].strip())
        focus_terms: list[str] = []
        for piece in pieces[1:]:
            excluded, cleaned = _excluded_terms(" ".join(piece.lower().split()))
            positive, _corrections = _positive_terms(cleaned, plan.excluded + excluded)
            for term in positive:
                if term not in plan.terms and term not in focus_terms:
                    focus_terms.append(term)
        focus_terms = focus_terms[:4]
        plan.raw_direction = direction
        if not focus_terms:
            plan.notes.append("The pivot did not add a subject word, so the earlier search stays as it was.")
            return plan
        extra = [f'all:"{term}"' if " " in term else f"all:{term}" for term in focus_terms]
        plan.query = f"{plan.query} AND " + " AND ".join(extra)
        plan.terms = _unique(plan.terms + focus_terms)
        plan.required = focus_terms
        plan.notes.append(
            "I kept the earlier direction. A paper also has to use "
            + " and ".join(focus_terms)
            + ", which is the part you asked to focus on."
        )
        return plan

    text = " ".join(direction.lower().split())
    if direction.strip().startswith("cat:") or "all:" in direction:
        return SearchPlan(
            query=direction.strip(),
            notes=["I sent this to arXiv as written."],
            raw_direction=direction,
            terms=[],
        )

    notes: list[str] = []
    rare = any(
        phrase in text
        for phrase in (
            "rare event",
            "rare events",
            "rare-event",
            "minority class",
            "class imbalance",
            "imbalanced",
            "imbalance",
            "false alarm",
        )
    )
    if rare:
        terms = ["imbalanced"]
        if "imbalance" not in text and "imbalanced" not in text:
            notes.append(
                "You described rare events. I am searching under “imbalanced classification”, "
                "which is the name this problem usually has in the computer-science index. "
                "Change the search if that is the wrong expansion."
            )
        if any(phrase in text for phrase in ("classif", "false alarm", "fraud", "churn", "detect")):
            terms.append("classification")
        for token in ("fraud", "calibration", "smote", "cost-sensitive", "resampling"):
            if token in text and token not in terms:
                terms.append(token)
        parts = [f'all:"{term}"' if " " in term else f"all:{term}" for term in terms]
        query = f"{CS_CATEGORIES} AND " + " AND ".join(parts)
        return SearchPlan(query=query, notes=notes, raw_direction=direction, terms=terms)

    excluded, cleaned = _excluded_terms(text)
    positive, corrections = _positive_terms(cleaned, excluded)
    if corrections:
        notes.append("I read " + ", ".join(f"“{src}” as {dst}" for src, dst in corrections) + ".")
    if not positive:
        positive = ["machine learning"]
        notes.append("I could not find a subject word in that note. I searched “machine learning”.")
    else:
        shown = " or ".join(positive)
        notes.append(
            f"I am searching for {shown}. "
            "A paper only stays if its title or abstract uses those words."
        )
    if excluded:
        notes.append(
            "You ruled out "
            + " and ".join(excluded)
            + ". I drop a paper when its title uses those words."
        )
    query_terms: list[str] = []
    for term in positive:
        query_terms.append(term)
        stem = _stem(term)
        if stem != term:
            query_terms.append(stem)
    query_terms = _unique(query_terms)
    parts = [f'all:"{term}"' if " " in term else f"all:{term}" for term in query_terms]
    joiner = " AND " if len(parts) == 1 else " OR "
    query = f"{CS_CATEGORIES} AND (" + joiner.join(parts) + ")" if len(parts) > 1 else f"{CS_CATEGORIES} AND {parts[0]}"
    return SearchPlan(
        query=query,
        notes=notes,
        raw_direction=direction,
        terms=positive,
        excluded=excluded,
    )


def search_papers(
    direction: str,
    *,
    limit: int = 8,
    query_override: str | None = None,
) -> tuple[str, list[str], list[Paper], str]:
    """Return query, notes, papers, and which index answered.

    A pasted arXiv query is sent as written. OpenAlex, used only after a rate
    limit, searches the same scientific terms rather than the full sentence.
    """
    plan = plan_query(direction)
    override = (query_override or "").strip()
    if override:
        plan = SearchPlan(
            query=override,
            notes=["You edited the query. I sent your version to arXiv."],
            raw_direction=direction,
            terms=plan.terms,
            excluded=plan.excluded,
        )
    return execute_search(plan, limit=limit)


def execute_search(plan: SearchPlan, *, limit: int = 8) -> tuple[str, list[str], list[Paper], str]:
    """Run one planned query. Notes are copied so the plan can be reused."""
    from .papiers import search_library

    query = plan.query
    notes = list(plan.notes)
    library, library_notes = search_library(plan.raw_direction or query, limit=min(limit, 6))
    notes.extend(library_notes)
    if library:
        matched = select_relevant(library, plan, limit=limit)
        if matched:
            library = matched
        else:
            library = []
            notes.append(
                "Papiers returned saved documents, and none of their text used the search words. I did not keep them."
            )

    last: FetchError | None = None
    arxiv_papers: list[Paper] = []
    for attempt in range(2):
        try:
            arxiv_papers = search_arxiv(query, limit=max(limit * 2, 16))
            break
        except FetchError as exc:
            last = exc
            if not exc.rate_limited:
                raise
            if time.time() < _arxiv_blocked_until:
                break
            if attempt == 0:
                time.sleep(4)
    if arxiv_papers:
        kept = select_relevant(arxiv_papers, plan, limit=limit)
        if kept:
            merged = _merge_papers(library, kept, limit)
            origin = "papiers" if any(paper.origin == "papiers" for paper in merged) else "arxiv"
            return query, notes, merged, origin
        if plan.excluded:
            notes.append(
                f"arXiv returned {len(arxiv_papers)} papers, and every title used a word you ruled out. I did not keep them."
            )
        else:
            notes.append(
                f"arXiv returned {len(arxiv_papers)} papers, and none of their titles or abstracts used the search words. I did not keep them."
            )
    oa_query = " ".join(plan.terms) if plan.terms else plan.raw_direction
    try:
        oa_papers = search_openalex(oa_query, limit=max(limit * 2, 16))
    except FetchError as exc:
        if library:
            return query, notes, library, "papiers"
        if last and last.rate_limited:
            raise FetchError(
                "arXiv rate-limited the search, and the fallback index failed too. Nothing was invented. Try again in a minute."
            ) from exc
        raise
    kept = select_relevant(oa_papers, plan, limit=limit)
    if last and last.rate_limited:
        notes = [
            *notes,
            f"arXiv was tried with {query} and rate-limited the request. OpenAlex was then searched for “{oa_query}”.",
        ]
    elif not arxiv_papers:
        notes = [*notes, f"arXiv returned nothing for {query}. OpenAlex was then searched for “{oa_query}”."]
    if not kept and oa_papers:
        notes.append(
            f"OpenAlex returned {len(oa_papers)} papers. After reading the titles and abstracts, none matched the words I kept."
        )
    merged = _merge_papers(library, kept, limit)
    if any(paper.origin == "papiers" for paper in merged) and not kept:
        return query, notes, merged, "papiers"
    origin = "papiers" if any(paper.origin == "papiers" for paper in merged) else "openalex"
    return oa_query, notes, merged, origin


def _merge_papers(library: list[Paper], indexed: list[Paper], limit: int) -> list[Paper]:
    """Library papers first. A public record with the same arXiv id is not listed twice."""
    chosen: list[Paper] = []
    seen: set[str] = set()
    for paper in [*library, *indexed]:
        key = paper.paper_id.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(paper)
        if len(chosen) >= limit:
            break
    return chosen


_arxiv_blocked_until = 0.0


def search_arxiv(query: str, *, limit: int = 8) -> list[Paper]:
    global _arxiv_blocked_until
    if time.time() < _arxiv_blocked_until:
        raise FetchError("arXiv is still rate-limiting this machine.", rate_limited=True)
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        body = fetch(url, timeout=20, polite_arxiv=True)
    except FetchError as exc:
        if exc.rate_limited:
            _arxiv_blocked_until = time.time() + 90
        raise
    return parse_arxiv_feed(body)


def parse_arxiv_feed(body: bytes) -> list[Paper]:
    if body.strip()[:20].lower().startswith(b"rate exceeded"):
        raise FetchError("arXiv returned 'Rate exceeded'.", rate_limited=True)
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise FetchError("arXiv did not return a paper feed.") from exc

    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = _text(entry, f"{ATOM}id")
        if not raw_id or "/abs/" not in raw_id:
            # An error feed has a single entry without an abs id.
            continue
        arxiv_id = raw_id.split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        title = _clean(_text(entry, f"{ATOM}title"))
        abstract = _clean(_text(entry, f"{ATOM}summary"))
        if not title or not abstract:
            continue
        authors = [_clean(_text(author, f"{ATOM}name")) for author in entry.findall(f"{ATOM}author")]
        authors = [a for a in authors if a]
        published = _text(entry, f"{ATOM}published")[:10]
        categories = []
        primary = entry.find(f"{ARXIV_NS}primary_category")
        if primary is not None and primary.attrib.get("term"):
            categories.append(primary.attrib["term"])
        for cat in entry.findall(f"{ATOM}category"):
            term = cat.attrib.get("term")
            if term and term not in categories:
                categories.append(term)
        pdf = None
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
                pdf = link.attrib.get("href")
        venue = _clean(_text(entry, f"{ARXIV_NS}journal_ref")) or None
        papers.append(
            Paper(
                paper_id=arxiv_id,
                title=title,
                authors=authors,
                published=published,
                abstract=abstract,
                link=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf or f"https://arxiv.org/pdf/{arxiv_id}",
                categories=categories,
                venue=venue,
                origin="arxiv",
            )
        )
    return papers


def search_openalex(query: str, *, limit: int = 8) -> list[Paper]:
    """Fallback when arXiv is rate-limiting. Records still come from the API."""
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "has_abstract:true,language:en",
            "per-page": str(max(limit * 2, 12)),
            "select": ",".join(
                [
                    "id",
                    "display_name",
                    "publication_year",
                    "authorships",
                    "primary_location",
                    "ids",
                    "open_access",
                    "abstract_inverted_index",
                    "primary_topic",
                    "type",
                ]
            ),
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    body = fetch(url, timeout=30, polite_arxiv=False)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FetchError("OpenAlex did not return paper records.") from exc

    papers: list[Paper] = []
    for work in payload.get("results") or []:
        paper = _openalex_work(work)
        if paper is None:
            continue
        papers.append(paper)
        if len(papers) >= limit:
            break
    return papers


def _openalex_work(work: dict) -> Paper | None:
    title = _clean(work.get("display_name") or "")
    abstract = reconstruct_inverted_index(work.get("abstract_inverted_index") or {})
    if not title or len(abstract) < 80:
        return None
    topic = work.get("primary_topic") or {}
    field = ((topic.get("field") or {}).get("display_name") or "").lower()
    domain = ((topic.get("domain") or {}).get("display_name") or "").lower()
    if field and "computer" not in field and "math" not in field and "computer" not in domain:
        # Keep the fallback inside the fields the desk claims to search.
        return None
    ids = work.get("ids") or {}
    arxiv_url = ids.get("arxiv")
    doi = ids.get("doi")
    location = work.get("primary_location") or {}
    source = (location.get("source") or {}).get("display_name")
    landing = location.get("landing_page_url")
    if isinstance(arxiv_url, str) and arxiv_url:
        match = re.search(r"(\d{4}\.\d{4,5})", arxiv_url)
        paper_id = match.group(1) if match else arxiv_url.rstrip("/").split("/")[-1]
        link = f"https://arxiv.org/abs/{paper_id}"
        origin = "arxiv"
        pdf = f"https://arxiv.org/pdf/{paper_id}"
    else:
        doi_str = doi if isinstance(doi, str) else ""
        if doi_str:
            paper_id = doi_str.removeprefix("https://doi.org/")
            link = doi_str if doi_str.startswith("http") else f"https://doi.org/{doi_str}"
        else:
            raw_id = str(work.get("id") or "")
            paper_id = raw_id.rstrip("/").split("/")[-1] or title[:48]
            link = landing or raw_id
        origin = "openalex"
        oa = (work.get("open_access") or {}).get("oa_url")
        pdf = oa if isinstance(oa, str) else None
    if not link:
        return None
    authors = []
    for authorship in work.get("authorships") or []:
        name = ((authorship.get("author") or {}).get("display_name") or "").strip()
        if name:
            authors.append(name)
    year = str(work.get("publication_year") or "")
    return Paper(
        paper_id=str(paper_id),
        title=title,
        authors=authors,
        published=year,
        abstract=abstract,
        link=link,
        pdf_url=pdf,
        categories=[field] if field else [],
        venue=source,
        origin=origin,
    )


def reconstruct_inverted_index(inverted: dict) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in inverted.items():
        for index in indexes:
            positions.append((int(index), str(token)))
    positions.sort()
    return _clean(" ".join(token for _, token in positions))


_GITHUB = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)


def github_urls(text: str) -> list[str]:
    """GitHub repository links copied out of text. Fragments and .git are stripped."""
    found: list[str] = []
    for match in _GITHUB.finditer(text or ""):
        url = match.group(0).rstrip(").,;]>'\"")
        url = re.sub(r"\.git$", "", url, flags=re.I)
        if url not in found:
            found.append(url)
    return found


def collect_githubs(papers: list[Paper], *, limit: int = 4) -> list[dict]:
    """Look in the abstract and, for a few arXiv papers, the HTML for a GitHub link."""
    found: list[dict] = []
    seen: set[str] = set()
    fetched = 0
    for paper in papers:
        blobs = [paper.title, paper.abstract, *(quote.text for quote in paper.quotes)]
        if paper.origin == "arxiv" and fetched < limit:
            fetched += 1
            bare = re.sub(r"v\d+$", "", paper.paper_id)
            try:
                html = fetch(f"https://arxiv.org/html/{bare}", timeout=8, polite_arxiv=False)
            except FetchError:
                html = b""
            if html:
                blobs.append(html.decode("utf-8", errors="replace"))
        for blob in blobs:
            for url in github_urls(blob):
                if url in seen:
                    continue
                seen.add(url)
                found.append({"url": url, "paper_title": paper.title, "link": paper.link})
    return found


def enrich_with_html(paper: Paper, *, timeout: float = 8) -> None:
    """Pull verbatim sentences from the arXiv HTML, when that HTML exists.

    Failure leaves the abstract quotes in place. A missing HTML version is
    normal for older papers.
    """
    if paper.origin != "arxiv":
        return
    bare = re.sub(r"v\d+$", "", paper.paper_id)
    url = f"https://arxiv.org/html/{bare}"
    try:
        body = fetch(url, timeout=timeout, polite_arxiv=False)
    except FetchError:
        return
    html = body.decode("utf-8", errors="replace")
    if "No HTML for" in html or "HTML is not available" in html:
        return
    paper.quotes.extend(html_quotes(html, limit=12))


def html_quotes(html: str, *, limit: int = 2) -> list[Quote]:
    soup = BeautifulSoup(html, "html.parser")
    found: list[Quote] = []
    sections = soup.select("section.ltx_section, section.ltx_subsection")
    for section in sections:
        heading = section.find(["h2", "h3", "h4"])
        if heading is None:
            continue
        title = _clean(heading.get_text(" ", strip=True))
        if not re.search(
            r"method|approach|algorithm|proposed|model|imbalance|resample|experiment|framework",
            title,
            re.I,
        ):
            continue
        paragraphs = [_clean(p.get_text(" ", strip=True)) for p in section.find_all("p")[:6]]
        text = " ".join(p for p in paragraphs if p)
        if len(text) < 80:
            continue
        # Keep one substantial sentence from the section, copied exactly.
        for sentence in split_sentences(text):
            if 80 <= len(sentence) <= 420 and _sentence_in(sentence, text):
                found.append(Quote(text=sentence, source="paper-html", section=title))
                break
        if len(found) >= limit:
            break
    return found


def split_sentences(text: str) -> list[str]:
    cleaned = _clean(text)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if len(part.strip()) >= 24]


def abstract_quotes(paper: Paper, needles: list[str], *, limit: int = 3) -> list[Quote]:
    """Choose abstract sentences that actually contain the needles.

    Every returned sentence is checked to be a whitespace-normalized
    substring of the abstract. Sentences that fail the check are dropped.
    """
    sentences = split_sentences(paper.abstract)
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        hay = sentence.lower()
        score = sum(2 if needle.lower() in hay else 0 for needle in needles)
        if score:
            ranked.append((score, sentence))
    ranked.sort(key=lambda item: (-item[0], -len(item[1])))
    chosen: list[Quote] = []
    seen: set[str] = set()
    for _, sentence in ranked:
        key = _normalize(sentence)
        if key in seen:
            continue
        if not _sentence_in(sentence, paper.abstract):
            continue
        seen.add(key)
        chosen.append(Quote(text=sentence, source="abstract", section=None))
        if len(chosen) >= limit:
            break
    if not chosen and sentences and _sentence_in(sentences[0], paper.abstract):
        chosen.append(Quote(text=sentences[0], source="abstract", section=None))
    return chosen


def _sentence_in(sentence: str, source: str) -> bool:
    return _normalize(sentence) in _normalize(source)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path)
    if found is None or found.text is None:
        return ""
    return found.text


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def arxiv_id_from_link(url: str) -> str:
    """Accept paper links only; never fetch a user-controlled host."""
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Enter an arXiv paper link.")
    parsed = urllib.parse.urlsplit(url.strip())
    if (parsed.scheme not in {"https", "http"}
            or parsed.netloc.lower() not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}):
        raise ValueError("Use an arXiv abstract, PDF, or HTML paper link.")
    match = re.fullmatch(r"/(?:abs|pdf|html)/((?:[0-9]{4}\.[0-9]{4,5}|[a-zA-Z][a-zA-Z.\-]*/[0-9]{7})(?:v[0-9]+)?)(?:\.pdf)?/?", parsed.path)
    if not match:
        raise ValueError("This is not a valid arXiv paper link.")
    return re.sub(r"v[0-9]+$", "", match.group(1))


def import_paper_link(url: str) -> Paper:
    paper_id = arxiv_id_from_link(url)
    endpoint = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": paper_id})
    try:
        papers = parse_arxiv_feed(fetch(endpoint, timeout=20, polite_arxiv=True))
    except (FetchError, ET.ParseError) as exc:
        raise ValueError("Could not retrieve this paper from arXiv. Please try again shortly.") from exc
    for paper in papers:
        if paper.paper_id == paper_id and paper.title and paper.abstract:
            return paper
    raise ValueError("arXiv did not return a complete record for this paper. Check the link.")
