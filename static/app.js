const state = {
  id: null,
  phase: "idle",
  detail: "Waiting for a direction.",
  paused: false,
  direction: "",
  query: "",
  origin: "",
  papers: [],
  paths: [],
  results: [],
  conclusion: null,
  question: null,
  draft: null,
  because: "",
  metric: "average_precision",
  dataset: null,
  datasetSearch: null,
  branches: [],
  research: { branches: [], runs: [], history: [] },
  trail: [],
  error: null,
  started: {},
  skipped: {},
  lastN: 0,
};

const METRIC_LABEL = {
  average_precision: "Average precision",
  f1: "F1",
  recall: "Recall",
  precision: "Precision",
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  roc_auc: "ROC AUC",
};

const pill = document.getElementById("pill");
const detail = document.getElementById("detail");
const pauseBtn = document.getElementById("pause");
const desk = document.getElementById("desk");
const trail = document.getElementById("trail");
let source = null;
let scheduled = false;
let importingPaper = false;

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function apply(event) {
  if (event.n && event.n <= state.lastN) return;
  if (event.n) state.lastN = event.n;
  switch (event.type) {
    case "research":
      state.research = event.research;
      break;
    case "status":
      state.phase = event.phase;
      state.detail = event.detail;
      if (event.phase === "done") refreshInvestigations();
      break;
    case "direction":
      state.direction = event.direction;
      break;
    case "query":
      state.query = event.query;
      state.origin = event.origin;
      break;
    case "paper": {
      const index = state.papers.findIndex((paper) => paper.id === event.paper.id);
      if (index >= 0) state.papers[index] = event.paper;
      else state.papers.push(event.paper);
      break;
    }
    case "paper_flag": {
      const paper = state.papers.find((item) => item.id === event.id);
      if (paper) paper.kept = event.kept;
      break;
    }
    case "paths":
      state.paths = event.paths;
      break;
    case "path_started":
      state.started[event.id] = true;
      break;
    case "path_skipped":
      state.skipped[event.id] = event.reason;
      break;
    case "results_reset":
      state.results = [];
      break;
    case "result": {
      const index = state.results.findIndex((item) => item.path_id === event.result.path_id);
      if (index >= 0) state.results[index] = event.result;
      else state.results.push(event.result);
      break;
    }
    case "metric":
      state.metric = event.metric;
      break;
    case "conclusion":
      state.conclusion = event.conclusion;
      state.because = event.because || "";
      break;
    case "code_draft":
      state.draft = event.code || "";
      break;
    case "question_result":
      state.question = event.result;
      state.draft = null;
      break;
    case "dataset":
      state.dataset = event.dataset;
      break;
    case "dataset_search": {
      const prior = state.datasetSearch || { papers: [], searches: [] };
      const seen = new Set(prior.papers.map((paper) => paper.id || paper.link));
      const added = (event.papers || []).filter((paper) => !seen.has(paper.id || paper.link));
      state.datasetSearch = {
        papers: prior.papers.concat(added),
        searches: prior.searches.concat([{ query: event.query || "", origin: event.origin || "" }]),
      };
      break;
    }
    case "branches":
      state.branches = event.branches || [];
      break;
    case "branch_result": {
      const index = state.branches.findIndex((item) => item.id === event.branch.id);
      if (index >= 0) state.branches[index] = event.branch;
      else state.branches.push(event.branch);
      break;
    }
    case "human":
      state.trail.push({ role: "human", text: event.text });
      break;
    case "agent":
      state.trail.push({ role: "agent", text: event.text });
      break;
    case "paused":
      state.paused = event.paused;
      if (event.paused) state.phase = "paused";
      break;
    case "error":
      state.error = event.message;
      state.phase = "error";
      break;
    case "reset":
      state.papers = [];
      state.paths = [];
      state.results = [];
      state.conclusion = null;
      state.question = null;
      state.draft = null;
      state.dataset = null;
      state.datasetSearch = null;
      state.branches = [];
      state.research.branches = [];
      state.research.literature_review = null;
      state.research.data = null;
      state.metric = "average_precision";
      state.query = "";
      state.error = null;
      state.started = {};
      state.skipped = {};
      state.detail = event.detail;
      break;
    default:
      break;
  }
  schedule();
}

function schedule() {
  if (scheduled) return;
  scheduled = true;
  setTimeout(() => {
    scheduled = false;
    render();
  }, 0);
}

function render() {
  const open = new Set(
    [...desk.querySelectorAll("details[open]")].map((node) => node.dataset.keep)
  );
  pill.textContent = labelForPhase();
  pill.className = "pill " + (state.phase === "paused" || state.paused ? "paused" : state.phase === "done" ? "done" : state.phase === "error" ? "error" : "");
  detail.textContent = state.detail;
  pauseBtn.disabled = !state.id;
  const go = document.getElementById("go");
  if (go) go.disabled = ["searching", "reading", "papers", "branching", "review", "running"].includes(state.phase);
  pauseBtn.textContent = state.paused ? "Resume" : "Pause";
  const drafts = [...desk.querySelectorAll('form[data-idea-form][data-dirty="true"], form[data-publication-form][data-dirty="true"]')].map(form => ({
    id: form.getAttribute("data-idea-form") || form.getAttribute("data-publication-form"), values: [...form.elements].filter(el => el.name).map(el => [el.name, el.value, el.type === "checkbox" ? el.checked : null]),
    open: form.closest("details")?.open, dirty: form.getAttribute("data-dirty"),
  }));
  const reviewNotes = [...desk.querySelectorAll("[data-review-note]")].map(el => [el.dataset.reviewNote, el.value]);
  const focused = document.activeElement;
  const activeName = focused?.name;
  const activeFormElement = focused?.closest("form[data-idea-form], form[data-publication-form]");
  const activeForm = activeFormElement?.getAttribute("data-idea-form") || activeFormElement?.getAttribute("data-publication-form");
  const selection = typeof focused?.selectionStart === "number" ? [focused.selectionStart, focused.selectionEnd] : null;
  desk.innerHTML = deskHtml();
  for (const draft of drafts) {
    const form = [...desk.querySelectorAll("form[data-idea-form], form[data-publication-form]")].find(f => (f.getAttribute("data-idea-form") || f.getAttribute("data-publication-form")) === draft.id);
    if (!form) continue;
    for (const [name, value, checked] of draft.values) {
      const el = form.elements.namedItem(name);
      if (el) { if (checked !== null) el.checked = checked; else el.value = value; }
    }
    form.setAttribute("data-dirty", draft.dirty || "false");
    if (draft.open) form.closest("details").open = true;
    if (draft.id === activeForm && activeName) {
      const el = form.elements.namedItem(activeName);
      el?.focus({preventScroll: true});
      if (selection && el?.setSelectionRange) el.setSelectionRange(...selection);
    }
  }
  for (const [id, value] of reviewNotes) {
    const el = [...desk.querySelectorAll("[data-review-note]")].find(el => el.dataset.reviewNote === id);
    if (el) el.value = value;
  }
  const live = document.getElementById("live-code");
  if (live) {
    live.scrollTop = live.scrollHeight;
    live.scrollIntoView({ block: "nearest" });
  }
  for (const node of desk.querySelectorAll("details")) {
    if (open.has(node.dataset.keep)) node.open = true;
  }
  trail.innerHTML = trailHtml();
}

function labelForPhase() {
  if (state.paused || state.phase === "paused") return "Paused";
  return {
    idle: "Idle",
    searching: "Searching",
    reading: "Reading",
    papers: "Papers",
    branching: "Branching",
    review: "Review",
    running: "Running",
    done: "Ready for review",
    interrupted: "Saved / interrupted",
    error: "Stopped",
  }[state.phase] || state.phase;
}

function deskHtml() {
  if (!state.id) return "";
  const parts = [];
  if (state.error) parts.push(`<div class="error">${esc(state.error)}</div>`);
  if (state.query) {
    const origin = state.origin === "openalex"
      ? "Fallback index: OpenAlex, after arXiv rate-limited the search."
      : state.origin === "papiers"
        ? "Source: your Papiers library, plus the public index for papers that were not already saved."
        : "Source: the arXiv API. Titles, authors, and abstracts are the feed’s fields.";
    parts.push(`
      <section class="section">
        <p class="num">01 — Search</p>
        <h2>What was sent</h2>
        <div class="query">${esc(state.query)}</div>
        ${state.direction ? `<p class="origin">Direction: ${esc(state.direction)}</p>` : ""}
        <p class="origin">${esc(origin)}</p>
      </section>`);
  }
  if (state.papers.length || ["searching", "reading", "papers"].includes(state.phase)) {
    parts.push(`
      <section class="section">
        <p class="num">02 — Literature</p>
        <h2>${state.papers.length} papers, in their own words</h2>
        ${["searching", "reading", "papers"].includes(state.phase) ? `<form data-publication-form="import-paper" data-paper-import><label>Add a paper by arXiv link<input type="url" name="url" placeholder="https://arxiv.org/abs/1706.03762" required></label><p class="origin">Abstract, PDF, and HTML links supported. Imported papers join your literature review before ideas are generated.</p><button type="submit" ${importingPaper ? "disabled" : ""}>${importingPaper ? "Adding paper…" : "Add paper"}</button></form>` : ""}
        ${state.papers.map((paper, index) => paperHtml(paper, index)).join("")}
        ${state.phase === "papers" ? `<button type="button" id="approve-papers" data-approve="1">Find branches from these papers</button>` : ""}
      </section>`);
  }
  if (state.research.literature_review) {
    const review = state.research.literature_review;
    parts.push(`<section class="section"><h2>Literature synthesis for your review</h2><p>${esc(review.summary)}</p>
      ${(review.findings || []).map(f => `<p>${esc(f.claim)}</p><blockquote>${esc(f.citation.quote)}<br><a href="${esc(f.citation.link)}" target="_blank" rel="noopener">${esc(f.citation.paper_title)}</a></blockquote>`).join("")}
      <p class="origin">${esc(review.limitations)}</p></section>`);
  }
  if (state.research.branches.length || state.research.runs.length || (state.phase === "review" && state.papers.length)) parts.push(researchHtml());
  if (state.phase === "interrupted") parts.push(`<button type="button" data-resume-saved>Resume saved investigation</button>`);
  const branches = state.branches.filter((branch) => branch.status !== "published" && branch.status !== "joined");
  if (branches.length) {
    parts.push(`
      <section class="section">
        <p class="num">03 — Branches</p>
        <h2>Branches from the papers you kept</h2>
        <p class="origin">Each branch is two approaches named in those papers. Begin the investigation to try them.</p>
        ${branches.map((branch) => branchHtml(branch)).join("")}
        ${state.phase === "review" ? `<button type="button" id="next-branches" data-proceed="1">Begin the investigation</button>` : ""}
      </section>`);
  }
  if (state.phase === "review" && !branches.length && !state.paths.length && !state.research.branches.length && !state.research.literature_review) {
    parts.push(`<p class="origin">No branch pair was named. You can still begin from the methods in the papers above.</p><button type="button" id="next-branches" data-proceed="1">Begin the investigation</button>`);
  }
  if (state.paths.length || state.dataset) {
    parts.push(`
      <section class="section">
        <p class="num">03 — Branches</p>
        <h2>${state.phase === "review" ? "Methods waiting for Next" : "Methods the papers named"}</h2>
        ${state.dataset ? `<div class="dataset">${esc(state.dataset.description)}${state.dataset.n_fit != null ? ` Fit ${state.dataset.n_fit}, validation ${state.dataset.n_val}, test ${state.dataset.n_test} (${state.dataset.n_pos_test} rare).` : ""} Seed ${state.dataset.seed}.${state.dataset.download ? ` <a href="${esc(state.dataset.download)}">Download the table (CSV)</a>` : ""}</div>` : ""}
        ${state.paths.map((path) => pathHtml(path)).join("")}
        ${state.phase === "review" ? `<button type="button" id="next-branches" data-proceed="1">Begin the investigation</button>` : ""}
      </section>`);
  }
  if (state.draft !== null) {
    parts.push(`
      <section class="section">
        <p class="num">04 — Writing</p>
        <h2>${esc(state.detail || "Writing the script")}</h2>
        <pre class="code live" id="live-code">${state.draft ? esc(state.draft) : "Writing the script. It will appear here when it is finished."}</pre>
      </section>`);
  }
  if (state.question) {
    const rows = (state.question.rows || []).filter((row) => row.all);
    parts.push(`
      <section class="section">
        <p class="num">04 — Result</p>
        <h2>Experiment Gemini wrote</h2>
        <p class="origin">The script was written from your direction and the approved papers, then run. The text is the script's own output.</p>
        <p class="reading">${esc(state.question.reading || "")}</p>
        ${rows.map((row) => `<div class="branch-paper"><a href="${esc(row.link)}" target="_blank" rel="noopener">${esc(row.title)}</a></div>`).join("")}
        ${state.question.code ? `<details><summary>Code the agent wrote</summary><pre class="code">${esc(state.question.code)}</pre></details>` : ""}
        ${state.question.download ? `<p><a href="${esc(state.question.download)}">Download the table (CSV)</a></p>` : ""}
      </section>`);
  }
  if (state.results.length || state.conclusion) {
    const metric = state.conclusion?.metric || state.metric || "average_precision";
    parts.push(`
      <section class="section">
        <p class="num">04 — Bench</p>
        <h2>${esc(METRIC_LABEL[metric] || metric)}</h2>
        ${state.because ? `<p class="origin">${esc(state.because)}</p>` : ""}
        ${chartHtml(metric)}
        ${state.results.map((result) => resultHtml(result)).join("")}
        ${conclusionHtml()}
      </section>`);
  }
  if (typeof publicationHtml === "function" && (state.research.branches.length || state.research.publication)) parts.push(publicationHtml());
  return parts.join("");
}

function paperHtml(paper, index) {
  const dropped = paper.kept === false ? " dropped" : "";
  const year = paper.year || paper.published || "";
  const venueName = (paper.venue || "").split("/")[0].trim();
  const venue = venueName ? ` · ${esc(venueName.slice(0, 72))}` : "";
  const cats = (paper.categories || []).slice(0, 3).join(", ");
  const linkLabel = paper.origin === "papiers"
    ? "Open the saved source"
    : String(paper.link || "").includes("arxiv.org")
      ? "Open on arXiv"
      : "Open the publisher page";
  const quotes = (paper.quotes || []).map((quote) => {
    const where = quote.source === "paper-html"
      ? `Paper HTML${quote.section ? ", " + quote.section : ""}, verbatim`
      : quote.source === "papiers"
        ? "Your Papiers library, verbatim"
        : quote.source === "title"
          ? "Title, verbatim"
          : "Abstract, verbatim";
    return `<blockquote><span class="badge">${esc(where)}</span><br />${esc(quote.text)}</blockquote>`;
  }).join("");
  return `
    <article class="paper${dropped}">
      <p class="meta">${String(index + 1).padStart(2, "0")} · ${esc(cats)} · ${esc(year)}${venue}</p>
      <h3>${esc(paper.title)}</h3>
      <p class="meta">${esc(paper.author_line)}</p>
      <div class="links">
        ${paper.link ? `<a href="${esc(paper.link)}" target="_blank" rel="noopener">${linkLabel}</a>` : ""}
        ${paper.pdf_url ? `<a href="${esc(paper.pdf_url)}" target="_blank" rel="noopener">PDF</a>` : ""}
      </div>
      ${quotes}
      <details data-keep="abs-${esc(paper.id)}">
        <summary>Full abstract</summary>
        <p class="abstract">${esc(paper.abstract)}</p>
      </details>
      ${paper.kept === false ? `<p class="skip">Struck from this investigation.</p>` : `<button type="button" class="text-button" data-drop="${esc(paper.id)}">Remove this paper</button>`}
    </article>`;
}

function branchHtml(branch) {
  const tag = {
    gap: ["gap", "Gap"],
    published: ["published", "Already published"],
    proposed: ["waiting", "Proposed"],
    joined: ["joined", "Already joined"],
    skipped: ["skipped", "Skipped"],
    queued: state.phase === "review" ? ["waiting", "Proposed"] : ["waiting", "Queued"],
  }[branch.status] || ["waiting", branch.status || "Queued"];
  const cites = (branch.citations || []).map((cite) => `
    <blockquote>
      <span class="badge">${esc(cite.source === "title" ? "Title" : "Abstract")}, verbatim · ${esc(cite.approach_name)}</span><br />
      ${esc(cite.quote)}
      <br /><a href="${esc(cite.link)}" target="_blank" rel="noopener">${esc(cite.paper_title)}</a>
    </blockquote>`).join("");
  const found = (branch.papers || []).map((paper) => {
    const quote = (paper.quotes || [])[0];
    return `
      <div class="branch-paper">
        <a href="${esc(paper.link)}" target="_blank" rel="noopener">${esc(paper.title)}</a>
        ${quote ? `<blockquote><span class="badge">Abstract, verbatim</span><br />${esc(quote.text)}</blockquote>` : ""}
      </div>`;
  }).join("");
  return `
    <article class="path">
      <span class="tag ${tag[0]}">${tag[1]}</span>
      <h3>${esc(branch.title)}</h3>
      <p class="reading">${esc(branch.reading || "")}</p>
      ${branch.query && branch.kind !== "joined" ? `<div class="query">${esc(branch.query)}</div>` : ""}
      ${cites}
      ${found}
    </article>`;
}

function uniqueCitations(path) {
  const seen = new Set();
  const out = [];
  for (const cite of path.citations || []) {
    const key = cite.link || cite.paper_title;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(cite);
  }
  return out;
}

function whyLine(path) {
  const named = new Map();
  for (const cite of path.citations || []) {
    const component = (path.components || []).find((item) => item.id === cite.method_id);
    const method = component ? component.name : cite.method_id;
    if (!named.has(method)) named.set(method, new Set());
    if (cite.paper_title) named.get(method).add(cite.paper_title);
  }
  const parts = [...named.entries()].map(([method, titles]) => {
    const list = [...titles];
    const source = list.length === 1 ? list[0] : list.join(" and ");
    return `${method}, named by ${source}`;
  });
  if (parts.length) return `Trying ${parts.join("; ")}.`;
  if (path.bench_note) return "Trying logistic regression as the bench readout, because no kept paper named a classifier.";
  return "";
}

function pathHtml(path) {
  const skipped = state.skipped[path.id];
  const running = state.started[path.id] && !state.results.some((item) => item.path_id === path.id) && !skipped;
  const done = state.results.some((item) => item.path_id === path.id);
  const proposed = state.phase === "review" && !skipped && !running && !done;
  const tag = skipped ? "skipped" : running ? "running" : done ? "reference" : "waiting";
  const tagLabel = skipped ? "Dropped" : running ? "Running" : done ? "Scored" : proposed ? "Proposed" : "Not run";
  const cites = (path.citations || []).map((cite) => `
    <blockquote>
      <span class="badge">${esc(cite.quote_source)}${cite.section ? ", " + cite.section : ""}, verbatim · ${esc(cite.method_id)}</span><br />
      ${esc(cite.quote)}
      <br /><a href="${esc(cite.link)}" target="_blank" rel="noopener">${esc(cite.paper_title)}</a>
    </blockquote>`).join("");
  const links = uniqueCitations(path).map((cite) => `
    <div class="links"><a href="${esc(cite.link)}" target="_blank" rel="noopener">${esc(cite.paper_title)}</a></div>`).join("");
  return `
    <article class="path">
      <span class="tag ${tag}">${tagLabel}</span>
      <h3>${esc(path.title)}</h3>
      <p class="assumption">${esc(whyLine(path))}</p>
      ${path.bench_note ? `<p class="bench-note">${esc(path.bench_note)}</p>` : ""}
      ${links}
      ${cites}
      ${skipped ? `<p class="skip">${esc(skipped)}</p>` : ""}
    </article>`;
}

function chartHtml(metric) {
  if (!state.results.length) return "";
  const rows = state.results.map((result) => {
    const value = Number(result.metrics[metric] ?? 0);
    const ci = result.ci?.[metric];
    const leader = state.conclusion?.leader_id === result.path_id;
    const left = ci ? Math.max(0, ci[0]) * 100 : 0;
    const right = ci ? Math.min(1, ci[1]) * 100 : 0;
    return `
      <div class="bar-row">
        <div>
          <div class="bar-name">${esc(result.title)}</div>
          <div class="track${leader ? " is-leader" : ""}">
            <div class="bar" style="width:${Math.max(0, Math.min(1, value)) * 100}%"></div>
            ${ci ? `<div class="whisker" style="left:${left}%; width:${Math.max(0, right - left)}%"></div>` : ""}
          </div>
        </div>
        <div class="bar-value">${value.toFixed(2)}</div>
      </div>`;
  }).join("");
  return rows;
}

function resultHtml(result) {
  const headline = result.reading?.headline || "";
  const tag = headline === "Worked" ? "worked" : headline === "Did not work" ? "failed" : headline === "Reference path" ? "reference" : "flat";
  const leader = state.conclusion?.leader_id === result.path_id ? " leader" : "";
  const metrics = ["average_precision", "recall", "precision", "f1", "accuracy"]
    .map((key) => `<span>${esc(METRIC_LABEL[key])} ${Number(result.metrics[key]).toFixed(2)}</span>`)
    .join("");
  return `
    <article class="result${leader}">
      <span class="tag ${tag}">${esc(headline)}</span>
      <h3>${esc(result.title)}</h3>
      <div class="metrics">${metrics}</div>
      <p class="fit">${esc(result.threshold_rule)} ${esc(result.fit_note)}</p>
      <p class="reading">${esc(result.reading?.body || "")}</p>
      ${result.code ? `<details><summary>Code that ran</summary><pre class="code">${esc(result.code)}</pre></details>` : ""}
    </article>`;
}

function conclusionHtml() {
  if (!state.conclusion) return "";
  const paragraphs = (state.conclusion.paragraphs || []).map((p) => `<p>${esc(p)}</p>`).join("");
  const question = state.conclusion.question ? `<p class="question">${esc(state.conclusion.question)}</p>` : "";
  return `<div class="conclusion">${paragraphs}${question}<p class="origin">Bands are 95% bootstrap intervals over the test rows. Models are not refit inside the band, so training noise is not in it.</p></div>`;
}

function trailHtml() {
  if (!state.trail.length) {
    return `<p class="empty-margin">Nothing in the margin yet. A note here can remove a kind of method, change the score, or hold a precision floor. Your changes are recorded with the next approved plan. Earlier results keep their original settings.</p>`;
  }
  return state.trail.map((note) => `
    <div class="note ${note.role}">
      <p class="who">${note.role === "human" ? "You" : "Agent"}</p>
      <p>${esc(note.text)}</p>
    </div>`).join("");
}

async function begin(direction, query) {
  state.id = null;
  state.lastN = 0;
  state.papers = [];
  state.paths = [];
  state.results = [];
  state.conclusion = null;
  state.question = null;
  state.draft = null;
  state.branches = [];
  state.research = { branches: [], runs: [], history: [] };
  state.trail = [];
  state.error = null;
  state.started = {};
  state.skipped = {};
  state.paused = false; state.dataset = null; state.datasetSearch = null; state.query = "";
  state.phase = "searching";
  state.detail = "Opening the investigation.";
  if (source) source.close();
  const response = await fetch("/api/investigations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ direction, query: query || null }),
  });
  if (!response.ok) {
    state.error = "Could not start.";
    state.phase = "error";
    schedule();
    return;
  }
  const payload = await response.json();
  state.id = payload.id;
  refreshInvestigations();
  source = new EventSource(`/api/investigations/${payload.id}/events`);
  source.onmessage = (message) => {
    apply(JSON.parse(message.data));
  };
  source.onerror = () => {
    if (state.phase !== "done") {
      state.detail = "The live connection dropped. Refresh if the desk stops moving.";
      schedule();
    }
  };
  schedule();
}

async function sendMargin(body) {
  if (!state.id) {
    state.trail.push({ role: "agent", text: "Begin an investigation first. The margin writes into a live one." });
    schedule();
    return;
  }
  const response = await fetch(`/api/investigations/${state.id}/margin`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = typeof error.detail === "string" ? error.detail : "That action was not accepted.";
    state.error = detail;
    state.trail.push({ role: "agent", text: detail });
    schedule();
    return false;
  }
  state.error = null;
  return true;
}

document.getElementById("start").addEventListener("submit", (event) => {
  event.preventDefault();
  const direction = document.getElementById("direction").value.trim();
  const query = document.getElementById("query").value.trim();
  if (direction) begin(direction, query);
});

document.getElementById("direction-chips").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-direction]");
  if (!button) return;
  document.getElementById("direction").value = button.dataset.direction;
});

document.getElementById("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const note = document.getElementById("note");
  const text = note.value.trim();
  if (!text) return;
  note.value = "";
  sendMargin({ text });
});

document.getElementById("composer").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-note]");
  if (!button) return;
  sendMargin({ text: button.dataset.note });
});

document.getElementById("redirect-go").addEventListener("click", () => {
  const input = document.getElementById("redirect");
  const direction = input.value.trim();
  if (!direction) return;
  input.value = "";
  sendMargin({ text: direction, action: { type: "redirect", direction } });
});

pauseBtn.addEventListener("click", () => {
  sendMargin({ action: { type: state.paused ? "resume" : "pause" } });
});

desk.addEventListener("click", async (event) => {
  const removeRule = event.target.closest("[data-remove-rule]");
  if (removeRule) { await sendMargin({action: {type: "remove_rule", rule: removeRule.dataset.removeRule}}); return; }
  const decision = event.target.closest("[data-idea-decision]");
  if (decision) {
    await sendMargin({action: {type: "edit_branch", branch_id: decision.dataset.ideaId, patch: {decision: decision.dataset.ideaDecision}}});
    return;
  }
  const runButton = event.target.closest("[data-run-review]");
  if (runButton) {
    const card = runButton.closest("[data-run-card]");
    await sendMargin({action: {type: "review_run", run_id: card.dataset.runCard, decision: runButton.dataset.runReview,
      note: card.querySelector("textarea")?.value || ""}});
    return;
  }
  const follow = event.target.closest("[data-follow-run]");
  if (follow) {
    const card = follow.closest("[data-run-card]");
    await sendMargin({action: {type: "follow_up", run_id: follow.dataset.followRun, note: card.querySelector("textarea")?.value || ""}});
    return;
  }
  const draftDemo = event.target.closest("[data-draft-demo]");
  if (draftDemo) { await sendMargin({action: {type: "draft_demonstration", branch_id: draftDemo.dataset.draftDemo}}); return; }
  const rerun = event.target.closest("[data-rerun-idea]");
  if (rerun) { await sendMargin({action: {type: "rerun", branch_id: rerun.dataset.rerunIdea}}); return; }
  if (event.target.closest("[data-cancel-work]")) { await sendMargin({action: {type: "cancel"}}); return; }
  if (event.target.closest("[data-resume-saved]")) { await sendMargin({action: {type: "resume_saved"}}); return; }
  if (event.target.closest("[data-approve-plans]")) {
    if (desk.querySelector("form[data-dirty='true']")) {
      state.error = "Save your plan edits before approving execution.";
      state.trail.push({role: "agent", text: state.error});
      schedule();
      return;
    }
    await sendMargin({action: {type: "proceed"}}); return;
  }
  if (event.target.closest("[data-approve]")) {
    sendMargin({ action: { type: "approve_papers" } });
    return;
  }
  if (event.target.closest("[data-proceed]")) {
    sendMargin({ action: { type: "proceed" } });
    return;
  }
  const button = event.target.closest("button[data-drop]");
  if (!button) return;
  sendMargin({ action: { type: "drop_paper", paper_id: button.dataset.drop } });
});

render();


function researchNumber(value) {
  return typeof value === "number" ? Number(value.toPrecision(6)).toString() : String(value ?? "");
}

const PLAN_LABELS = {
  title: "Idea title", hypothesis: "Testable hypothesis (agent proposal)", method: "Method and experimental procedure",
  dataset: "Dataset or explicit demonstration inputs", baseline: "Baseline comparison", evaluation: "Evaluation and success criteria",
  assumptions: "Assumptions and missing details", limitations: "Limitations", evidence_level: "Evidence level",
};

function ideaForm(branch = null) {
  const fields = Object.entries(PLAN_LABELS).map(([key, label]) => `<label>${esc(label)}${key === "evidence_level"
    ? `<select name="${key}">${["exploratory", "demonstration", "reproduction"].map(level => `<option ${branch?.[key] === level ? "selected" : ""}>${level}</option>`).join("")}</select>`
    : `<textarea name="${key}" rows="${key === "title" ? 1 : 3}" maxlength="6000" required>${esc(branch?.[key] || "")}</textarea>`}</label>`).join("");
  const cited = new Set((branch?.citations || []).map(c => c.paper_id));
  return `<form data-idea-form="${esc(branch?.id || "new")}">${fields}
    <label>Supporting paper IDs (comma separated)<input name="paper_ids" value="${esc([...cited].join(", "))}" required></label>
    <p class="origin">Saving changed plan fields uses the model executor so your edits are honored. Demonstration results use explicitly specified small inputs and do not establish benchmark performance.</p>
    <button type="submit">${branch ? "Save plan edits" : "Add my idea"}</button></form>`;
}

function researchHtml() {
  const research = state.research;
  const busy = ["running", "branching", "reading", "searching"].includes(state.phase);
  return `<section class="section"><p class="num">03 — Research ideas</p><h2>You choose the work</h2>
    <p class="origin">These hypotheses are proposals. Source passages support their motivation; they do not prove the hypotheses or novelty.
    Plans include researcher-selected baselines, seeds, and implementation choices. Every execution requires your approval.</p>
    <p class="origin">Current objective: ${esc(research.settings?.metric || state.metric)}${research.settings?.precision_floor ? ` · precision floor ${research.settings.precision_floor}` : ""}.
      These settings and your guidance are recorded with each approval.</p>
    ${(research.settings?.guidance || []).map(note => `<p class="origin">Your guidance: ${esc(note)}</p>`).join("")}
    <div class="chips">${(research.settings?.rules || []).filter(rule => rule !== "human_metric").map(rule => `<button type="button" data-remove-rule="${esc(rule)}" ${busy ? "disabled" : ""}>Remove constraint: ${esc(rule)}</button>`).join("")}</div>
    ${research.data ? `<p class="dataset">Prepared input: ${esc(research.data.name)} · ${research.data.n_rows} rows · ${research.data.n_features} features · seed 11.
      <a href="${esc(research.data.source)}" target="_blank" rel="noopener">Data evidence</a><br>${esc(research.data.label_rule)}<br>SHA-256: <code>${esc(research.data.sha256)}</code></p>` : `<p class="origin">${esc(research.data_error || "No prepared dataset. Complete a demonstration plan with explicit inputs, or revise the literature search.")}</p>`}
    ${research.branches.map(branch => `<article class="research-card">
      <p class="kicker">${esc(branch.decision)} · ${esc(branch.status)} · plan v${branch.version} · ${esc(branch.evidence_level)}</p>
      <h3>${esc(branch.title)}</h3><p>${esc(branch.hypothesis)}</p>
      ${branch.parent_id ? `<p class="origin">Follow-up to run ${esc(branch.parent_id)}</p>` : ""}
      ${branch.block_reason ? `<p class="origin">${esc(branch.block_reason)}</p>` : ""}
      <dl>${["method", "dataset", "baseline", "evaluation", "assumptions", "limitations"].map(key => `<dt>${esc(PLAN_LABELS[key])}</dt><dd>${esc(branch[key])}</dd>`).join("")}</dl>
      <p class="origin">${esc(branch.estimated_effort || "Local CPU; maximum 180 seconds per experiment process, plus model writing.")}</p>
      ${(branch.citations || []).map(c => `<blockquote>${esc(c.quote)}<br><a href="${esc(c.link)}" target="_blank" rel="noopener">${esc(c.paper_title)}</a> <small>(${esc(c.paper_id)})</small></blockquote>`).join("")}
      <div class="chips">${["selected", "deferred", "rejected"].map(decision => `<button type="button" data-idea-id="${esc(branch.id)}" data-idea-decision="${decision}" ${busy || branch.decision === decision ? "disabled" : ""}>${{selected: "Select", deferred: "Defer", rejected: "Reject"}[decision]}</button>`).join("")}
      <button type="button" data-rerun-idea="${esc(branch.id)}" ${busy ? "disabled" : ""}>Prepare rerun</button>${branch.status === "blocked" || branch.requires_edit ? `<button type="button" data-draft-demo="${esc(branch.id)}" ${busy ? "disabled" : ""}>Draft a demonstration plan</button>` : ""}</div>
      <details data-keep="idea-${esc(branch.id)}"><summary>Edit this plan</summary>${ideaForm(branch)}</details>
    </article>`).join("")}
    <details data-keep="new-idea"><summary>Add a human research idea</summary>${ideaForm()}</details>
    ${["review", "done", "interrupted"].includes(state.phase) ? `<button type="button" data-approve-plans ${state.paused ? "disabled" : ""}>Approve and run selected plans</button>` : ""}
    ${busy || state.phase === "review" ? `<button type="button" data-cancel-work>Cancel active work</button>` : ""}
    <p><a href="/api/investigations/${state.id}/export" target="_blank">Export investigation and approval history (JSON)</a></p>
  </section>
  ${research.runs.length ? `<section class="section"><p class="num">04 — Results for your review</p><h2>Evidence from each run</h2>
    ${research.runs.map(run => researchResult(run)).join("")}</section>` : ""}`;
}

function artifactUrl(run, filename) {
  return `/api/investigations/${state.id}/runs/${run.id}/artifacts/${filename.split("/").map(encodeURIComponent).join("/")}`;
}

function researchResult(run) {
  const result = run.result;
  return `<article class="research-card" data-run-card="${esc(run.id)}">
    <p class="kicker">${esc(run.status)} · ${esc(run.evidence_level)} · ${esc(run.review.replaceAll("_", " "))}</p>
    <h3>${esc(run.plan.title)}</h3><p class="origin">Run ${esc(run.id)} · plan v${run.plan.version} · ${esc(run.writer)} · seed ${run.settings.seed}</p>
    ${run.error ? `<p class="error">${esc(run.error)}</p>` : ""}
    ${result ? `<p>${esc(result.summary)}</p><p>${esc(run.interpretation)}</p>
      <table><thead><tr><th>Measurement</th><th>This idea</th><th>Baseline</th><th>Difference</th></tr></thead><tbody>
      ${Object.entries(result.measurements).map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(researchNumber(value))}</td><td>${esc(researchNumber(result.baseline[key]))}</td><td>${esc(researchNumber(run.comparison[key]))}</td></tr>`).join("")}</tbody></table>
      <p class="origin">Data: ${esc(result.data)}</p><p>Limitations: ${esc(result.limitations)}</p>
      ${result.details ? `<p class="origin">${esc(result.details.threshold_rule)}</p><details data-keep="scores-${run.id}"><summary>Scores, feasibility and bootstrap intervals</summary><pre class="code">${esc(JSON.stringify({idea: result.details, baseline: result.baseline_details}, null, 2))}</pre></details>` : ""}
      <label>Your interpretation or follow-up<textarea rows="3" data-review-note="${esc(run.id)}" maxlength="6000"></textarea></label>
      <div class="chips"><button data-run-review="accepted">Accept result</button><button data-run-review="needs_revision">Needs revision</button><button data-run-review="rejected">Reject result</button>
      <button data-follow-run="${esc(run.id)}" ${["running", "branching", "review"].includes(state.phase) ? "disabled" : ""}>Propose follow-up ideas</button></div>
      ${(run.review_history || []).map(review => `<p class="origin">Human review: ${esc(review.decision)} — ${esc(review.note)}</p>`).join("")}` : ""}
    <details data-keep="plan-run-${run.id}"><summary>Exact approved plan and settings</summary><pre class="code">${esc(JSON.stringify({plan: run.plan, settings: run.settings}, null, 2))}</pre></details>
    <p>${run.artifacts.map(filename => `<a href="${artifactUrl(run, filename)}">${esc(filename)}</a>`).join(" · ")}</p>
  </article>`;
}

desk.addEventListener("input", event => {
  const form = event.target.closest("form[data-idea-form]");
  if (form) form.setAttribute("data-dirty", "true");
});

desk.addEventListener("submit", async event => {
  const form = event.target.closest("form[data-idea-form]");
  if (!form) return;
  event.preventDefault();
  const plan = Object.fromEntries(new FormData(form));
  const paper_ids = plan.paper_ids.split(",").map(s => s.trim()).filter(Boolean);
  delete plan.paper_ids;
  const action = form.getAttribute("data-idea-form") === "new" ? {type: "add_branch", plan, paper_ids}
    : {type: "edit_branch", branch_id: form.getAttribute("data-idea-form"), patch: {...plan, paper_ids}};
  if (await sendMargin({action})) {
    const current = [...desk.querySelectorAll("form[data-idea-form]")].find(f => f.getAttribute("data-idea-form") === form.getAttribute("data-idea-form"));
    if (current) { current.setAttribute("data-dirty", "false"); current.closest("details").open = false; }
    form.setAttribute("data-dirty", "false");
    form.closest("details").open = false;
    schedule();
    if (form.getAttribute("data-idea-form") === "new") form.reset();
  }
});

async function refreshInvestigations() {
  const response = await fetch("/api/investigations");
  if (!response.ok) return;
  const investigations = await response.json();
  const selectedId = state.id || "";
  document.getElementById("saved-investigations").innerHTML = `<option value="">Open a saved investigation…</option>` +
    investigations.map(i => `<option value="${esc(i.id)}" ${i.id === selectedId ? "selected" : ""}>${esc(i.direction.slice(0, 90))} (${esc(i.phase)}, ${i.runs} runs)</option>`).join("");
}

function openSaved(id) {
  if (source) source.close();
  state.id = id;
  state.lastN = 0;
  state.papers = []; state.trail = []; state.results = []; state.paths = []; state.branches = [];
  state.research = {branches: [], runs: [], history: []}; state.error = null; state.draft = null;
  state.question = null; state.dataset = null; state.conclusion = null; state.query = ""; state.paused = false;
  source = new EventSource(`/api/investigations/${id}/events`);
  source.onmessage = message => apply(JSON.parse(message.data));
  schedule();
}

document.getElementById("saved-investigations").addEventListener("change", event => {
  if (event.target.value) openSaved(event.target.value);
});
refreshInvestigations();

desk.addEventListener("submit", async event => {
  const form = event.target.closest("form[data-paper-import]");
  if (!form) return;
  event.preventDefault();
  if (importingPaper) return;
  importingPaper = true;
  const url = new FormData(form).get("url");
  form.querySelector("button").disabled = true;
  form.querySelector("button").textContent = "Adding paper…";
  try {
    if (await sendMargin({action: {type: "add_paper_link", url}})) {
      const current = desk.querySelector("form[data-paper-import]");
      if (current) { current.reset(); current.removeAttribute("data-dirty"); }
    }
  } finally { importingPaper = false; schedule(); }
});
