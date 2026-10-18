"""Choose the rows for an experiment from what the papers actually say.

A public dataset is used only when its name appears in an approved abstract.
A synthetic table is used only when an abstract states the sample size, the
number of features, the class balance, and a normal or Gaussian draw.
Otherwise there is no dataset. Nothing here invents one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sources import CS_CATEGORIES, Paper, plan_query, SearchPlan

# Ambiguous words only count when the sentence is actually citing a table.
_NEEDS_CUE = {"sonar", "yeast", "hepatitis"}
_CUE = re.compile(r"dataset|data set|\buci\b|keel|openml|benchmark|repository", re.I)
_REJECT = {
    "sonar": re.compile(r"aperture|imagery|image|acoustic|\bSAS\b|convolutional", re.I),
    "yeast": re.compile(r"two-hybrid|protein|cell cycle|gene expression", re.I),
    "hepatitis": re.compile(r"virus|vaccine|clinical trial", re.I),
}
# Words that describe a method, not the subject. A follow-up dataset search
# must not treat them as the topic, or a breast-cancer paper can satisfy "predict".
_GENERIC = {
    "algorithm", "algorithms", "approach", "approaches", "based", "classification",
    "classifier", "classifiers", "compare", "feature", "features", "learning",
    "machine", "method", "methods", "model", "models", "network", "neural",
    "predict", "predicted", "prediction", "predicts", "synthetic", "synthesis",
    "train", "training", "using",
}
_PUBLIC = (
    (
        re.compile(
            r"wisconsin(?: diagnostic)? breast[- ]cancer|breast[- ]cancer wisconsin|\bwdbc\b|\bwbcd\b",
            re.I,
        ),
        "breast_cancer",
        "Wisconsin breast cancer",
    ),
    (re.compile(r"\bionosphere\b", re.I), "ionosphere", "Ionosphere"),
    (re.compile(r"\bsonar\b", re.I), "sonar", "Sonar"),
    (re.compile(r"\bpima\b|pima indians", re.I), "diabetes", "Pima Indians diabetes"),
    (re.compile(r"german credit|credit-g", re.I), "credit-g", "German credit"),
    (re.compile(r"\bspambase\b", re.I), "spambase", "Spambase"),
    (re.compile(r"\bhaberman\b", re.I), "haberman", "Haberman"),
    (re.compile(r"\becoli\b|e\.?\s*coli", re.I), "ecoli", "E. coli"),
    (re.compile(r"\byeast\b", re.I), "yeast", "Yeast"),
    (re.compile(r"page[- ]blocks", re.I), "page-blocks", "Page blocks"),
    (re.compile(r"\bsatimage\b", re.I), "satimage", "Satimage"),
    (re.compile(r"\bhepatitis\b", re.I), "hepatitis", "Hepatitis"),
)


@dataclass
class Grounding:
    kind: str  # "public" or "synthesis"
    evidence: str
    paper_title: str
    paper_link: str
    dataset_name: str | None = None
    loader: str | None = None
    n_rows: int | None = None
    n_features: int | None = None
    positive_rate: float | None = None

    def as_public(self) -> dict:
        if self.kind == "public":
            description = (
                f"Rows are the {self.dataset_name} table, because {self.paper_title} names it. "
                f"The sentence is: {self.evidence}"
            )
        else:
            description = (
                f"Rows are synthesized from a procedure stated by {self.paper_title}: "
                f"{self.n_rows} rows, {self.n_features} features, "
                f"about {self.positive_rate:.0%} in the smaller class, normal draws. "
                f"The sentence is: {self.evidence} "
                "The paper does not state a difference between the classes, so none was added."
            )
        return {
            "seed": 11,
            "n_rows": self.n_rows,
            "positive_rate": None if self.positive_rate is None else round(self.positive_rate, 4),
            "n_fit": None,
            "n_val": None,
            "n_test": None,
            "n_pos_test": None,
            "n_features": self.n_features,
            "description": description,
        }


def _subject_terms(direction: str) -> list[str]:
    """Subject words only. Generic method words must not steer a dataset search."""
    plan = plan_query(direction)
    subject = [term for term in plan.terms if term not in _GENERIC]
    return (subject or plan.terms or ["classification"])[:4]


def dataset_plan(direction: str) -> SearchPlan:
    """A second search for papers that name a dataset on the same subject."""
    topic = _subject_terms(direction)
    topic_query = " AND ".join(f'all:"{term}"' if " " in term else f"all:{term}" for term in topic)
    return SearchPlan(
        query=(
            f"{CS_CATEGORIES} AND (all:dataset OR all:benchmark OR all:uci OR all:keel) "
            f"AND ({topic_query})"
        ),
        notes=[
            "The papers already on the desk do not name a table I can load. "
            "I searched again for a dataset on the same subject."
        ],
        raw_direction=direction,
        terms=topic,
        required=topic,
    )


def dataset_name_plan(direction: str) -> SearchPlan:
    """Search for a public table this desk can actually load."""
    topic = _subject_terms(direction)
    topic_query = " AND ".join(f'all:"{term}"' if " " in term else f"all:{term}" for term in topic)
    names = " OR ".join((
        "all:yeast",
        "all:pima",
        'all:"breast cancer"',
        "all:haberman",
        "all:spambase",
        "all:ionosphere",
        "all:ecoli",
        "all:satimage",
        'all:"page blocks"',
        'all:"sonar dataset"',
        "all:keel",
        'all:"german credit"',
        "all:hepatitis",
    ))
    return SearchPlan(
        query=f"{CS_CATEGORIES} AND ({topic_query}) AND ({names})",
        notes=[
            "The dataset search did not name a table I can load. "
            "I searched again for a named public dataset on the same subject."
        ],
        raw_direction=direction,
        terms=topic,
        required=topic,
    )


def public_groundings(papers: list[Paper]) -> list[Grounding]:
    """Every public table named by these papers, without a second copy of the same loader."""
    found: list[Grounding] = []
    seen: set[str] = set()
    for paper in papers:
        for sentence in _sentences(paper.abstract):
            grounding = _public(sentence, paper)
            if grounding is None or grounding.loader in seen:
                continue
            seen.add(grounding.loader or "")
            found.append(grounding)
    return found


def table_loads(grounding: Grounding) -> bool:
    """True when this machine can actually read the named table."""
    if grounding.kind != "public":
        return True
    import os

    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    try:
        if grounding.loader == "breast_cancer":
            from sklearn.datasets import load_breast_cancer

            bunch = load_breast_cancer()
        else:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml(
                name=grounding.loader,
                version=1,
                as_frame=False,
                parser="liac-arff",
            )
    except Exception:
        return False
    data = getattr(bunch, "data", None)
    return data is not None and len(data) > 0


def first_loadable(papers: list[Paper]) -> Grounding | None:
    """A named table this machine can read, otherwise a fully stated synthesis."""
    for grounding in public_groundings(papers):
        if table_loads(grounding):
            return grounding
    return _first_synthesis(papers)


def _first_synthesis(papers: list[Paper]) -> Grounding | None:
    for paper in papers:
        for sentence in _sentences(paper.abstract):
            grounding = _synthesis(sentence, paper)
            if grounding is not None:
                return grounding
    return None


def ground_papers(papers: list[Paper]) -> Grounding | None:
    """Prefer a named public dataset. Otherwise a fully stated normal synthesis."""
    found = public_groundings(papers)
    if found:
        return found[0]
    return _first_synthesis(papers)


def evidence_sentences(papers: list[Paper]) -> list[str]:
    found = []
    for paper in papers:
        for sentence in _sentences(paper.abstract):
            if _looks_like_data(sentence):
                found.append(sentence)
    return found


def _public(sentence: str, paper: Paper) -> Grounding | None:
    for pattern, loader, name in _PUBLIC:
        if not pattern.search(sentence):
            continue
        if loader in _REJECT and _REJECT[loader].search(sentence):
            continue
        if loader in _NEEDS_CUE and not _CUE.search(sentence):
            continue
        return Grounding(
            kind="public",
            evidence=sentence,
            paper_title=paper.title,
            paper_link=paper.link,
            dataset_name=name,
            loader=loader,
        )
    return None


def _synthesis(sentence: str, paper: Paper) -> Grounding | None:
    lowered = sentence.lower()
    if not re.search(r"synthetic|synthesi[sz]ed|simulated|artificially generated|we generate", lowered):
        return None
    if not re.search(r"normal|gaussian", lowered):
        return None
    rows = _first_int(sentence, r"(\d+)\s*(?:samples|instances|examples|rows)")
    features = _first_int(sentence, r"(\d+)\s*(?:features|attributes|dimensions)")
    rate = _positive_rate(sentence)
    if rows is None or features is None or rate is None:
        return None
    if rows < 20 or rows > 200_000 or features < 1 or features > 500:
        return None
    return Grounding(
        kind="synthesis",
        evidence=sentence,
        paper_title=paper.title,
        paper_link=paper.link,
        n_rows=rows,
        n_features=features,
        positive_rate=rate,
    )


def _positive_rate(sentence: str) -> float | None:
    percent = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of\s+)?(?:the\s+)?(?:rows\s+)?(?:are\s+)?(?:the\s+)?(?:minority|positive|rare)",
        sentence,
        flags=re.I,
    )
    if percent:
        value = float(percent.group(1)) / 100
        if 0 < value < 0.5:
            return value
    ratio = re.search(r"(\d+)\s*:\s*(\d+)", sentence)
    if ratio:
        left, right = int(ratio.group(1)), int(ratio.group(2))
        total = left + right
        if total > 0:
            value = min(left, right) / total
            if 0 < value < 0.5:
                return value
    counts = re.search(
        r"(\d+)\s*(?:minority|positive)\s*(?:samples|instances|examples|rows).{0,80}?(\d+)\s*(?:majority|negative)",
        sentence,
        flags=re.I,
    )
    if counts:
        pos, neg = int(counts.group(1)), int(counts.group(2))
        total = pos + neg
        if total > 0 and pos < neg:
            return pos / total
    return None


def _first_int(sentence: str, pattern: str) -> int | None:
    match = re.search(pattern, sentence, flags=re.I)
    if not match:
        return None
    return int(match.group(1))


def _looks_like_data(sentence: str) -> bool:
    return bool(
        re.search(
            r"dataset|data set|synthetic|synthesi|simulated|uci|keel|openml|benchmark",
            sentence,
            flags=re.I,
        )
    )


def _sentences(text: str) -> list[str]:
    clean = " ".join(text.split())
    parts = re.split(r"(?<=[.])\s+", clean)
    return [part.strip() for part in parts if len(part.strip()) >= 40]
