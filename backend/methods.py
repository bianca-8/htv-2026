"""Methods the bench can run, and the rule for when it is allowed to run them.

A method is eligible only when a kept paper's own words match it. A combination
is eligible only when every component has such a citation, except the bench
readout noted in the path itself: preprocessing papers often do not name a
classifier, and the desk says so instead of pretending they did.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sources import Paper, Quote, abstract_quotes, split_sentences


@dataclass(frozen=True)
class Method:
    method_id: str
    name: str
    family: str  # model, resample, weight, threshold, ensemble
    pattern: re.Pattern
    assumption: str
    # If set, scrub these phrases before matching so a proper name is not
    # mistaken for the generic technique ("synthetic minority oversampling").
    scrub: tuple[str, ...] = ()

    def matches(self, text: str) -> bool:
        hay = text.lower()
        for phrase in self.scrub:
            hay = hay.replace(phrase, " ")
        return self.pattern.search(hay) is not None


METHODS: list[Method] = [
    Method(
        "logreg",
        "Logistic regression",
        "model",
        re.compile(r"logistic regression"),
        "A linear classifier with a default cutoff at 0.5 unless a threshold step is added.",
    ),
    Method(
        "random_forest",
        "Random forest",
        "model",
        re.compile(r"random forests?"),
        "An ensemble of decision trees. It can split the two minority clusters the linear model averages over.",
    ),
    Method(
        "gbdt",
        "Gradient boosting",
        "model",
        re.compile(r"xgboost|gradient boosting|gbdt|lightgbm|catboost|gradient-boosted"),
        "Histogram gradient boosting on this bench. If the paper named XGBoost, LightGBM, or CatBoost, this is the same family, not their code.",
    ),
    Method(
        "decision_tree",
        "Decision tree",
        "model",
        re.compile(r"decision trees?"),
        "A single shallow tree. Papers that only say “random forest” do not match this.",
    ),
    Method(
        "smote",
        "SMOTE",
        "resample",
        re.compile(r"\bsmote\b|synthetic minority"),
        "New minority rows are interpolated between a real minority row and one of its neighbors. Those rows were never observed.",
    ),
    Method(
        "random_oversample",
        "Random oversampling",
        "resample",
        re.compile(r"random over-?sampling"),
        "Minority rows are copied until the class is less rare. Duplicates, not new coordinates.",
        scrub=(
            "synthetic minority over-sampling technique",
            "synthetic minority oversampling technique",
            "minority over-sampling technique",
            "minority oversampling technique",
        ),
    ),
    Method(
        "random_undersample",
        "Random undersampling",
        "resample",
        re.compile(r"under-?sampling"),
        "Majority rows are thrown away at random until the classes are closer in size. Information goes with them.",
    ),
    Method(
        "class_weight",
        "Class weighting",
        "weight",
        re.compile(r"class[- ]weights?|cost-sensitive|cost sensitive|misclassification costs?"),
        "Every real row is kept. Minority mistakes weigh more in the loss, by inverse class frequency. Not a custom cost matrix.",
    ),
    Method(
        "threshold",
        "Threshold selection",
        "threshold",
        re.compile(
            r"decision thresholds?|classification thresholds?|threshold (?:moving|tuning|selection|adjustment)|operating points?"
        ),
        "The model is unchanged. A cutoff is chosen on a validation slice of the training data, never on the test rows.",
    ),
    Method(
        "easy_ensemble",
        "EasyEnsemble",
        "ensemble",
        re.compile(r"easyensemble|easy ensemble"),
        "Several logistic regressions, each trained on all minority rows and a different random majority subset. Their probabilities are averaged.",
    ),
]

BY_ID = {method.method_id: method for method in METHODS}
MODEL_ORDER = ["logreg", "random_forest", "gbdt", "decision_tree"]
RESAMPLE_ORDER = ["smote", "random_oversample", "random_undersample"]


@dataclass
class Citation:
    paper_id: str
    paper_title: str
    link: str
    quote: str
    quote_source: str
    section: str | None
    method_id: str

    def as_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "link": self.link,
            "quote": self.quote,
            "quote_source": self.quote_source,
            "section": self.section,
            "method_id": self.method_id,
        }


@dataclass
class Path:
    path_id: str
    title: str
    component_ids: list[str]
    citations: list[Citation]
    assumptions: list[str]
    bench_note: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.path_id,
            "title": self.title,
            "components": [
                {"id": method_id, "name": BY_ID[method_id].name, "family": BY_ID[method_id].family}
                for method_id in self.component_ids
            ],
            "citations": [c.as_dict() for c in self.citations],
            "assumptions": self.assumptions,
            "bench_note": self.bench_note,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


def attach_quotes(papers: list[Paper], focus: list[str] | None = None) -> None:
    """Fill each paper with verbatim sentences that mention the search or a bench method."""
    needles = list(focus or []) + [
        "imbalanced",
        "imbalance",
        "minority",
        "smote",
        "oversampling",
        "over-sampling",
        "undersampling",
        "under-sampling",
        "cost-sensitive",
        "class weight",
        "logistic regression",
        "random forest",
        "gradient boosting",
        "threshold",
        "resampling",
        "classification",
    ]
    for paper in papers:
        quotes = abstract_quotes(paper, needles, limit=3)
        kept: list[Quote] = []
        seen = {_norm(quote.text) for quote in quotes}
        for quote in paper.quotes:
            key = _norm(quote.text)
            if not key or key in seen:
                continue
            if quote.source == "papiers" and key in _norm(paper.abstract):
                kept.append(quote)
                seen.add(key)
            elif quote.source == "paper-html" and (
                any(method.matches(quote.text) for method in METHODS)
                or any(needle in quote.text.lower() for needle in ("imbalance", "minority", "resample"))
            ):
                kept.append(quote)
                seen.add(key)
        paper.quotes = quotes + kept[:3]


def plan_paths(
    papers: list[Paper],
    *,
    dropped: set[str] | None = None,
    blocked_methods: set[str] | None = None,
) -> list[Path]:
    dropped = dropped or set()
    blocked_methods = blocked_methods or set()
    kept = [paper for paper in papers if paper.paper_id not in dropped]
    hits: dict[str, list[Citation]] = {method.method_id: [] for method in METHODS}
    for paper in kept:
        for method in METHODS:
            citation = _best_citation(paper, method)
            if citation:
                hits[method.method_id].append(citation)

    available = {method_id for method_id, citations in hits.items() if citations}
    models = [method_id for method_id in MODEL_ORDER if method_id in available][:3]
    resamples = [method_id for method_id in RESAMPLE_ORDER if method_id in available]
    bench_note = None
    if not models and (resamples or "class_weight" in available or "easy_ensemble" in available):
        models = ["logreg"]
        bench_note = (
            "These papers describe a preprocessing or reweighting step and do not name logistic regression. "
            "Logistic regression is the bench's fixed readout so those steps can be compared. "
            "It is not claimed as their method."
        )

    paths: list[Path] = []
    if bench_note and not hits["logreg"]:
        paths.append(
            Path(
                path_id="logreg",
                title="Logistic regression",
                component_ids=["logreg"],
                citations=[],
                assumptions=[BY_ID["logreg"].assumption],
                bench_note=(
                    "No kept paper names logistic regression. This is the unweighted readout, "
                    "a control for the methods the papers do name. It is not one of their contributions."
                ),
            )
        )

    def add(component_ids: list[str], *, note: str | None = None) -> None:
        if len(paths) >= 8:
            return
        if any(component in blocked_methods for component in component_ids):
            return
        path_id = "__".join(component_ids)
        if any(path.path_id == path_id for path in paths):
            return
        citations: list[Citation] = []
        assumptions: list[str] = []
        for component in component_ids:
            assumptions.append(BY_ID[component].assumption)
            # The bench readout has no paper citation on purpose.
            if note and component == "logreg" and not hits["logreg"]:
                continue
            if hits[component]:
                citations.append(hits[component][0])
        if not citations:
            return
        paths.append(
            Path(
                path_id=path_id,
                title=_title(component_ids),
                component_ids=component_ids,
                citations=citations,
                assumptions=_unique(assumptions),
                bench_note=note if note and "logreg" in component_ids and not hits["logreg"] else None,
            )
        )

    # Families first, then combinations. A full cartesian product crowds out
    # class weighting and threshold selection, which are often the paths a
    # person wants to compare against resampling.
    primary = models[0] if models else None
    secondary = models[1] if len(models) > 1 else None
    for model in models[:2]:
        add([model], note=bench_note)
    if "class_weight" in available and primary:
        add(["class_weight", primary], note=bench_note)
    for resample in resamples:
        if primary:
            add([resample, primary], note=bench_note)
    if secondary and resamples:
        add([resamples[0], secondary], note=bench_note)
    if "threshold" in available and primary:
        add(["threshold", primary], note=bench_note)
        if "class_weight" in available:
            add(["class_weight", "threshold", primary], note=bench_note)
    if "easy_ensemble" in available:
        add(["easy_ensemble"])
    if secondary and "class_weight" in available:
        add(["class_weight", secondary], note=bench_note)

    return paths


def forbidden_from_rules(rules: set[str]) -> set[str]:
    """Map a human's constraints onto method ids that must not be run."""
    blocked: set[str] = set()
    if "no_synthetic" in rules:
        blocked.update({"smote", "random_oversample"})
    if "no_undersample" in rules:
        blocked.add("random_undersample")
    if "no_weight" in rules:
        blocked.add("class_weight")
    if "no_threshold" in rules:
        blocked.add("threshold")
    return blocked


def _best_citation(paper: Paper, method: Method) -> Citation | None:
    candidates: list[tuple[str, str, str | None]] = []
    if method.matches(paper.title):
        candidates.append((paper.title, "title", None))
    for sentence in split_sentences(paper.abstract):
        if method.matches(sentence):
            candidates.append((sentence, "abstract", None))
    for quote in paper.quotes:
        if quote.source == "paper-html" and method.matches(quote.text):
            candidates.append((quote.text, "paper-html", quote.section))
    if not candidates:
        return None
    # Prefer a full sentence from the abstract or HTML over a bare title.
    candidates.sort(key=lambda item: {"paper-html": 0, "abstract": 1, "title": 2}[item[1]])
    text, source, section = candidates[0]
    return Citation(
        paper_id=paper.paper_id,
        paper_title=paper.title,
        link=paper.link,
        quote=text,
        quote_source=source,
        section=section,
        method_id=method.method_id,
    )


def _title(component_ids: list[str]) -> str:
    names = [BY_ID[method_id].name for method_id in component_ids]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]}, then {names[1]}"
    return f"{names[0]} and {names[1]}, then {names[2]}"


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def method_needles() -> list[str]:
    return [method.name for method in METHODS]
