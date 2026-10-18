"""One investigation: search, quote, run, and stop when a person writes in the margin."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path as FilePath

from .publication import PublicationWorkflow
from .workflow import ResearchWorkflow
from . import storage
from .bench import Dataset, StoredRun, score_run
from .http_cache import FetchError
from .margin import interpret
from .methods import BY_ID, Path, attach_quotes, forbidden_from_rules, plan_paths
from .narrative import conclude, reading_for
from .sources import Paper, compose_direction, enrich_with_html, plan_query, search_papers, import_paper_link

ALLOWED_METRICS = {"average_precision", "f1", "recall", "precision", "accuracy", "balanced_accuracy"}


class Restart(Exception):
    def __init__(self, direction: str, query: str | None = None):
        self.direction = direction
        self.query = query


@dataclass
class Investigation(ResearchWorkflow, PublicationWorkflow):
    direction: str
    query_override: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    phase: str = "idle"
    paused: bool = False
    rules: set[str] = field(default_factory=set)
    dropped: set[str] = field(default_factory=set)
    metric: str = "average_precision"
    precision_floor: float | None = None
    papers: list[Paper] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    runs: dict[str, StoredRun] = field(default_factory=dict)
    scripts: dict[str, str] = field(default_factory=dict)
    dataset: Dataset | None = None
    grounding: object | None = None
    events: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    seq: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    redirect: tuple[str, str | None] | None = None
    task: asyncio.Task | None = None
    approved: bool = False
    papers_ready: bool = False
    paper_imports: int = 0

    research: dict = field(default_factory=lambda: {"branches": [], "runs": [], "history": [], "revision": 0})
    cancel_requested: bool = False
    persistent: bool = False
    publication_task: asyncio.Task | None = None

    def emit(self, event: dict) -> None:
        self.seq += 1
        event = {**event, "ts": time.time(), "n": self.seq}
        self.events.append(event)
        if self.persistent:
            storage.save(self, event)
        for queue in list(self.subscribers):
            queue.put_nowait(event)

    async def checkpoint(self) -> None:
        while True:
            async with self.lock:
                if self.redirect:
                    direction, query = self.redirect
                    self.redirect = None
                    raise Restart(direction, query)
                paused = self.paused
            if not paused:
                return
            self.phase = "paused"
            await asyncio.sleep(0.2)

    def start(self) -> None:
        self.task = asyncio.create_task(self._task())

    async def _task(self) -> None:
        direction = self.direction
        query = self.query_override
        while True:
            try:
                await self._run_once(direction, query)
                return
            except asyncio.CancelledError:
                raise
            except Restart as restart:
                direction = restart.direction
                query = restart.query
                previous_direction = self.direction
                self.direction = direction
                self.query_override = query
                self.research.setdefault("history", []).append({"type": "iteration", "direction": previous_direction, "branches": self.research.get("branches", []), "time": time.time()})
                self.research["branches"] = []
                self.research["data"] = None
                self.research["literature_review"] = None
                self.research["approval"] = None
                self.research["revision"] = self.research.get("revision", 0) + 1
                self.grounding = None
                self.dataset = None
                self.papers = []
                self.paths = []
                self.runs = {}
                self.scripts = {}
                self.dropped = set()
                self.approved = False
                self.papers_ready = False
                self.emit({"type": "reset", "detail": "Starting again with the new direction. Your metric and constraints stay."})

            except Exception as exc:
                self.phase = "interrupted"
                self.emit({"type": "error", "message": str(exc)})
                self.emit({"type": "status", "phase": "interrupted", "detail": "Work stopped; saved evidence and results remain available."})
                return

    async def _run_once(self, direction: str, query: str | None) -> None:
        self.phase = "searching"
        self.emit({"type": "status", "phase": "searching", "detail": "Searching the literature."})
        self.emit({"type": "direction", "direction": direction})
        try:
            found_query, notes, papers, origin = await asyncio.to_thread(
                search_papers, direction, limit=8, query_override=query
            )
        except FetchError as exc:
            found_query, notes, papers, origin = query or direction, [str(exc)], [], "arxiv"

        # An import can be in flight while the automated search finishes.
        while self.paper_imports:
            await self.checkpoint()
            await asyncio.sleep(0.1)

        self.emit({"type": "query", "query": found_query, "notes": notes, "origin": origin})
        for note in notes:
            self.emit({"type": "agent", "text": note})
        if origin == "openalex":
            self.emit({
                "type": "agent",
                "text": (
                    "arXiv rate-limited this machine, so I asked OpenAlex for published works with abstracts. "
                    "Each card links to arXiv when an arXiv id was in the record, and to the DOI otherwise. "
                    "I did not fill in missing papers."
                ),
            })

        papers = list({p.paper_id: p for p in [*papers, *self.papers]}.values())
        if not papers:
            self.phase = "done"
            self.emit({
                "type": "agent",
                "text": "Nothing matched after I read the titles and abstracts. I am not going to invent papers. The note above is the search I actually sent.",
            })
            self.emit({"type": "status", "phase": "done", "detail": "No papers."})
            return

        self.papers = papers
        self.phase = "reading"
        focus = plan_query(direction).terms
        html_budget = 8
        for index, paper in enumerate(papers, start=1):
            await self.checkpoint()
            if paper.paper_id in self.dropped:
                continue
            attach_quotes([paper], focus)
            self.emit({
                "type": "status",
                "phase": "reading",
                "detail": f"Reading paper {index} of {len(papers)}.",
            })
            self.emit({"type": "paper", "paper": {**paper.as_dict(), "kept": paper.paper_id not in self.dropped}})
            if html_budget > 0 and paper.origin == "arxiv":
                html_budget -= 1
                before = len(paper.quotes)
                try:
                    await asyncio.to_thread(enrich_with_html, paper)
                except FetchError:
                    html_budget = 0
                if len(paper.quotes) > before:
                    attach_quotes([paper], focus)
                    self.emit({"type": "paper", "paper": {**paper.as_dict(), "kept": True}, "update": True})

        await self.checkpoint()
        await self._wait_for_papers()
        await self._research(direction)

    def _table_path(self) -> str:
        folder = FilePath(__file__).resolve().parents[1] / "experiments" / "tables"
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / f"{self.id}.csv")

    async def _wait_for_papers(self) -> None:
        self.phase = "papers"
        self.papers_ready = False
        kept = [paper for paper in self.papers if paper.paper_id not in self.dropped]
        self.emit({
            "type": "status",
            "phase": "papers",
            "detail": "Approve the papers, or strike the ones that do not belong.",
        })
        self.emit({
            "type": "agent",
            "text": (
                f"{len(kept)} papers are on the desk. "
                "Strike any that do not belong. I will not look for branches until you approve the rest."
            ),
        })
        while True:
            await self.checkpoint()
            if self.papers_ready:
                self.papers_ready = False
                return
            await asyncio.sleep(0.2)

    async def _wait_to_run(self) -> None:
        self.phase = "review"
        self.approved = False
        self.emit({"type": "status", "phase": "review", "detail": "Press Begin the investigation."})
        while True:
            await self.checkpoint()
            if self.approved:
                self.approved = False
                return
            await asyncio.sleep(0.2)

    def _path_allowed(self, path: Path) -> bool:
        blocked = forbidden_from_rules(self.rules)
        if any(component in blocked for component in path.component_ids):
            return False
        kept = [paper for paper in self.papers if paper.paper_id not in self.dropped]
        for component in path.component_ids:
            method = BY_ID[component]
            if any(method.matches(paper.blob()) for paper in kept):
                continue
            # Logistic regression may be the bench's readout, not a paper's method.
            if component == "logreg" and path.bench_note:
                continue
            return False
        return True

    def _public_result(self, path: Path, scored, reference: str | None) -> dict:
        baseline_scored = None
        if reference and reference in self.runs and reference != path.path_id:
            reference_path = next(item for item in self.paths if item.path_id == reference)
            baseline_scored = score_run(
                self.runs[reference],
                components=reference_path.component_ids,
                precision_floor=self.precision_floor,
            )
        elif reference == path.path_id:
            baseline_scored = scored
        reading = reading_for(scored, None if reference != path.path_id else scored, self.metric)
        if reference and reference != path.path_id and baseline_scored is not None:
            reading = reading_for(scored, baseline_scored, self.metric)
        return {
            **scored.as_dict(),
            "title": path.title,
            "reading": reading,
            "reference": reference == path.path_id,
            "code": self.scripts.get(path.path_id, ""),
        }

    def _emit_conclusion(self, because: str) -> None:
        self.paths = plan_paths(
            self.papers,
            dropped=self.dropped,
            blocked_methods=forbidden_from_rules(self.rules),
        )
        self.emit({"type": "paths", "paths": [path.as_dict() for path in self.paths]})
        allowed = [path for path in self.paths if path.path_id in self.runs]
        reference = _reference_id(allowed) or _reference_id(self.paths)
        titles = {path.path_id: path.title for path in self.paths}
        scored = [
            score_run(self.runs[path.path_id], components=path.component_ids, precision_floor=self.precision_floor)
            for path in allowed
        ]
        summary = conclude(
            titles,
            scored,
            self.metric,
            precision_floor=self.precision_floor,
            reference_id=reference,
        )
        # Refresh result cards so a metric change rewrites "worked / did not".
        self.emit({"type": "results_reset"})
        for path, item in zip(allowed, scored):
            self.emit({"type": "result", "result": self._public_result(path, item, reference)})
        self.emit({"type": "conclusion", "conclusion": summary, "because": because})
        self.emit({"type": "agent", "text": " ".join(summary["paragraphs"])})
        if summary.get("question"):
            self.emit({"type": "agent", "text": summary["question"]})

    async def add_paper_link(self, url: str) -> None:
        if self.phase not in {"searching", "reading", "papers"} or self.papers_ready:
            raise ValueError("Add paper links during literature search or paper review.")
        revision = self.research.get("revision", 0)
        self.paper_imports += 1
        try:
            paper = await asyncio.to_thread(import_paper_link, url)
            if (self.phase not in {"searching", "reading", "papers"} or self.papers_ready
                    or self.redirect or self.research.get("revision", 0) != revision):
                raise ValueError("The literature search changed. Add the link to the current search again.")
            existing = next((p for p in self.papers if p.paper_id == paper.paper_id), None)
            if existing:
                paper = existing
            else:
                attach_quotes([paper], plan_query(self.direction).terms)
                self.papers.append(paper)
            self.dropped.discard(paper.paper_id)
            self.emit({"type": "paper", "paper": {**paper.as_dict(), "kept": True}})
            self.emit({"type": "human", "text": f"Added paper: {paper.link}"})
        finally:
            self.paper_imports -= 1

    async def handle(self, text: str | None = None, action: dict | None = None) -> None:
        actions: list[dict] = []
        if action:
            actions.append(action)
        if text:
            self.emit({"type": "human", "text": text.strip()})
            parsed = interpret(text)
            if parsed:
                actions.extend(parsed)
            elif not action:
                self.research.setdefault("instructions", []).append(text.strip())
                self.invalidate_approval()
                self.research_event()
                self.emit({"type": "agent", "text": "Saved your research guidance. It will accompany the next approved plan; review the plan before running."})
                return

        changed = False
        for item in actions:
            kind = item.get("type")
            if kind == "add_paper_link":
                await self.add_paper_link(item.get("url", ""))
                continue
            if await self.handle_publication_action(item):
                continue
            if await self.handle_research_action(item):
                continue
            if kind == "pause":
                self.paused = True
                self.emit({"type": "paused", "paused": True})
                self.emit({"type": "agent", "text": "Paused. I will stop between steps. Write in the margin, or tell me to resume."})
            elif kind == "resume":
                self.paused = False
                self.phase = "running" if self.runs and self.phase == "paused" else self.phase
                if self.phase == "paused":
                    self.phase = "review" if self.research.get("branches") and not any(r["status"] == "running" for r in self.research.get("runs", [])) else "running"
                self.emit({"type": "status", "phase": self.phase, "detail": "Resuming saved work."})
                self.emit({"type": "paused", "paused": False})
                self.emit({"type": "agent", "text": "Resuming."})
            elif kind == "drop_paper":
                paper_id = str(item.get("paper_id") or "")
                if paper_id:
                    self.dropped.add(paper_id)
                    title = next((paper.title for paper in self.papers if paper.paper_id == paper_id), paper_id)
                    self.emit({"type": "paper_flag", "id": paper_id, "kept": False})
                    self.emit({
                        "type": "agent",
                        "text": f"Struck “{title}”. A path stays only if some paper you kept still names each of its methods.",
                    })
                    changed = True
            elif kind == "remove_rule":
                rule = str(item.get("rule") or "")
                self.rules.discard(rule)
                changed = True
                self.emit({"type": "agent", "text": "Constraint removed. Review plans before running again."})
            elif kind == "rule":
                rule = str(item.get("rule") or "")
                if rule:
                    self.rules.add(rule)
                    self.emit({"type": "agent", "text": _rule_text(rule)})
                    changed = True
            elif kind == "metric":
                metric = str(item.get("metric") or "")
                if metric in ALLOWED_METRICS:
                    self.metric = metric
                    self.rules.add("human_metric")
                    self.precision_floor = None
                    self.emit({"type": "agent", "text": f"Ranking by {metric.replace('_', ' ')}."})
                    changed = True
            elif kind == "precision_floor":
                value = float(item.get("value"))
                if not 0 < value < 1:
                    raise ValueError("Precision floor must be between 0 and 1.")
                self.precision_floor = value
                self.metric = "recall"
                self.rules.add("human_metric")
                self.emit({
                    "type": "agent",
                    "text": (
                        f"I will retune cutoffs on the validation slice to maximize recall at precision ≥ {value:.2f}. "
                        "The test rows still do not choose the cutoff. Paths that cannot hold the floor on validation are not winners."
                    ),
                })
                changed = True
            elif kind == "approve_papers":
                if self.phase == "papers":
                    if self.paper_imports:
                        raise ValueError("Wait for the paper import to finish before generating ideas.")
                    kept = [paper for paper in self.papers if paper.paper_id not in self.dropped]
                    if not kept:
                        raise ValueError("Keep at least one paper before generating research ideas.")
                    self.papers_ready = True
                    kept = [paper for paper in self.papers if paper.paper_id not in self.dropped]
                    self.emit({
                        "type": "agent",
                        "text": f"Using the {len(kept)} papers you kept. Looking for branches from those only.",
                    })
                else:
                    self.emit({"type": "agent", "text": "Approve waits until the papers are on the desk."})
            elif kind == "proceed":
                if self.phase == "review":
                    self.approved = True
                    self.emit({"type": "agent", "text": "Running the branches still on the desk."})
                else:
                    self.emit({"type": "agent", "text": "Next waits until the branch list is on the desk."})
            elif kind == "redirect":
                direction = str(item.get("direction") or "").strip()
                query = item.get("query")
                if not direction and not query:
                    continue
                combined = compose_direction(self.direction, direction) if direction else self.direction
                if direction and combined != direction:
                    self.emit({
                        "type": "agent",
                        "text": (
                            "Searching again with both. The direction you started from stays, "
                            f"and I am adding this focus: {direction}"
                        ),
                    })
                else:
                    self.emit({"type": "agent", "text": "Setting this pass down and searching again. Your score and constraints stay."})
                self.cancel_requested = True
                self.approved = False
                self.research["approval"] = None
                if self.task and not self.task.done():
                    async with self.lock:
                        self.redirect = (combined, query)
                        self.paused = False
                else:
                    previous_direction = self.direction
                    self.direction = combined
                    self.query_override = query
                    self.research.setdefault("history", []).append({"type": "iteration", "direction": previous_direction, "branches": self.research.get("branches", []), "time": time.time()})
                    self.research["branches"] = []
                    self.research["data"] = None
                    self.research["literature_review"] = None
                    self.research["revision"] = self.research.get("revision", 0) + 1
                    self.grounding = None
                    self.dataset = None
                    self.papers = []
                    self.paths = []
                    self.runs = {}
                    self.scripts = {}
                    self.dropped = set()
                    self.emit({"type": "reset", "detail": "Starting again with the new direction."})
                    self.start()
                return

        if changed and (self.research.get("branches") or self.phase == "branching"):
            self.invalidate_approval()
            kept_ids = {p.paper_id for p in self.papers if p.paper_id not in self.dropped}
            kept_links = {p.link for p in self.papers if p.paper_id in kept_ids}
            data = self.research.get("data")
            if data and data.get("source") not in kept_links:
                self.research["data"] = None
                self.grounding = None
            for branch in self.research["branches"]:
                branch["version"] += 1
                if any(c["paper_id"] not in kept_ids for c in branch["citations"]):
                    branch.update(status="blocked", selected=False, decision="deferred", block_reason="Supporting paper removed; update evidence before approval.")
                if branch.get("data") is not None:
                    branch["data"] = self.research.get("data")
                if branch["executor"] == "bench":
                    branch["evaluation"] = f"{self.metric}; seed 11; identical fit/validation/test splits; validation-only cutoffs; precision floor {self.precision_floor}."
            self.research_event()
            self.emit({"type": "agent", "text": "Your changes invalidate prior approval. Review the updated plans before running again; previous run settings remain recorded."})
        elif changed and self.phase == "review":
            self.paths = plan_paths(
                self.papers,
                dropped=self.dropped,
                blocked_methods=forbidden_from_rules(self.rules),
            )
            self.emit({"type": "paths", "paths": [path.as_dict() for path in self.paths]})
            if self.paths:
                self.emit({
                    "type": "agent",
                    "text": "The branch list now follows the papers you kept and the constraints in the margin. Press Next when this is the set to run.",
                })
            else:
                self.emit({
                    "type": "agent",
                    "text": "Nothing left is named by a paper you kept. Press Next and I will stop without inventing a method.",
                })
        elif changed and self.phase == "done" and self.runs:
            self._emit_conclusion("Your note changed the ranking. I did not refit. I rescored the probabilities I already had.")
        elif changed and self.phase == "done" and not self.runs and self.papers:
            # Literature finished with nothing run, and a constraint was lifted
            # or the bench was previously skipped. Do not auto-rerun a skipped topic.
            self.emit({"type": "agent", "text": "Nothing has been fit yet, so the note is saved for the next pass."})


def _reference_id(paths: list[Path]) -> str | None:
    for path in paths:
        if path.component_ids in (["logreg"], ["random_forest"], ["gbdt"], ["decision_tree"]):
            return path.path_id
    return paths[0].path_id if paths else None


def _rule_text(rule: str) -> str:
    return {
        "no_synthetic": "No synthetic rows. I will drop SMOTE and random oversampling. Real copies and real reweighting can stay.",
        "no_undersample": "No undersampling. I will not throw majority rows away.",
        "no_weight": "No class weighting.",
        "no_threshold": "No threshold sweeps beyond the default cutoff.",
    }.get(rule, "Constraint saved.")


def _standing(rules: set[str], precision_floor: float | None) -> str:
    bits = []
    if "no_synthetic" in rules:
        bits.append("no synthetic samples")
    if "no_undersample" in rules:
        bits.append("no undersampling")
    if "no_weight" in rules:
        bits.append("no class weighting")
    if precision_floor is not None:
        bits.append(f"precision floor {precision_floor:.2f}")
    if not bits:
        return ""
    return "Still in force from the margin: " + ", ".join(bits) + "."


INVESTIGATIONS: dict[str, Investigation] = {}


def create(direction: str, query: str | None = None) -> Investigation:
    investigation = Investigation(direction=direction.strip(), query_override=(query or None), persistent=True)
    INVESTIGATIONS[investigation.id] = investigation
    investigation.emit({"type": "human", "text": investigation.direction})
    investigation.start()
    return investigation


def restore_saved() -> None:
    from .sources import Quote
    from .data_ground import Grounding
    for saved in storage.snapshots():
        if saved["id"] in INVESTIGATIONS:
            continue
        investigation = Investigation(
            direction=saved["direction"], query_override=saved.get("query_override"), id=saved["id"],
            phase=saved["phase"], rules=set(saved.get("rules", [])), dropped=set(saved.get("dropped", [])),
            metric=saved.get("metric", "average_precision"), precision_floor=saved.get("precision_floor"),
            research=saved.get("research", {"branches": [], "runs": [], "history": [], "revision": 0}),
            seq=saved.get("seq", 0), persistent=True,
        )
        for p in saved.get("papers", []):
            investigation.papers.append(Paper(
                paper_id=p["id"], title=p["title"], authors=p["authors"], published=p["published"],
                abstract=p["abstract"], link=p["link"], pdf_url=p.get("pdf_url"), categories=p["categories"],
                venue=p.get("venue"), origin=p["origin"], quotes=[Quote(**q) for q in p.get("quotes", [])],
            ))
        investigation.events = storage.events(investigation.id)
        ground = investigation.research.get("grounding")
        investigation.grounding = Grounding(**ground) if ground else None
        for run in investigation.research.get("runs", []):
            if run["status"] == "running":
                run.update(status="interrupted", error="Server stopped during this run. Fresh approval is required.")
                branch = next((b for b in investigation.research["branches"] if b["id"] == run["branch_id"]), None)
                if branch:
                    branch["status"] = "interrupted"
        if investigation.phase not in {"done", "idle"}:
            investigation.phase = "interrupted"
        investigation.research["approval"] = None
        publication = investigation.research.get("publication", {})
        if publication.get("search_status") == "searching":
            publication["search_status"] = "interrupted"
        INVESTIGATIONS[investigation.id] = investigation
        investigation.emit({"type": "status", "phase": investigation.phase, "detail": "Investigation restored. Resume and review before running."})
        investigation.research_event()
