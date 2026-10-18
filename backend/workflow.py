"""Human-reviewed branch lifecycle shared by every research topic."""
import asyncio
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np
from . import research, solana_payments
from .bench import StoredRun, grounded_source, _ground_rows, score_run
from .data_ground import ground_papers
from .gemini_agent import extract_script, writer_name
from .methods import Path as MethodPath, forbidden_from_rules, plan_paths
from .runner import execute, RunCancelled

ROOT = Path(__file__).resolve().parents[1]


def prepare_data(grounding, folder):
    """Only trusted loaders fetch data; experiments receive local immutable input copies."""
    folder.mkdir(parents=True, exist_ok=True)
    if grounding.kind == 'public':
        from sklearn.datasets import load_breast_cancer, fetch_openml
        bunch = load_breast_cancer() if grounding.loader == 'breast_cancer' else fetch_openml(
            name=grounding.loader, version=1, as_frame=False, parser='liac-arff')
        x = np.asarray(bunch.data, dtype=float)
        labels = np.asarray(bunch.target).astype(str)
        values, counts = np.unique(labels, return_counts=True)
        y = (labels == values[np.argmin(counts)]).astype(int)
    else:
        if not (10 <= grounding.n_rows <= 100000 and 1 <= grounding.n_features <= 1000):
            raise ValueError('Stated synthesis exceeds the local execution budget or is too small.')
        x = np.random.default_rng(11).normal(size=(grounding.n_rows, grounding.n_features))
        count = max(1, round(grounding.n_rows * grounding.positive_rate))
        y = np.r_[np.zeros(grounding.n_rows-count, dtype=int), np.ones(count, dtype=int)]
    if x.ndim != 2 or not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError('The table needs finite numeric features and at least two classes. No silent imputation was applied.')
    if x.nbytes + y.nbytes > 24 * 1024 * 1024:
        raise ValueError('Input exceeds the 24 MB local experiment budget; choose a smaller supported dataset.')
    path = folder / 'inputs.npz'
    np.savez_compressed(path, x=x, y=y)
    return {**grounding.as_public(), 'kind': grounding.kind, 'name': grounding.dataset_name or 'paper-stated synthesis',
            'source': grounding.paper_link, 'evidence': grounding.evidence,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n_rows': len(y), 'n_features': x.shape[1],
            'label_rule': 'Smallest class is positive; multiclass tables become one-versus-rest.', 'seed': 11}


def bench_script(path, grounding):
    source = grounded_source(path, grounding, 'dataset.csv')
    original = '\n'.join(_ground_rows(grounding))
    rows = '''data = np.load('inputs.npz', allow_pickle=False)
x = data['x']
y = data['y']
seed = 11
rng = np.random.default_rng(seed)
notes = ['Prepared local dataset; seed and splits are bench choices, not paper claims.']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed, stratify=y_train)
x, y = x_fit.copy(), y_fit.copy()
n_pos_before = int(y.sum())'''
    return source.replace(original, rows)


def scored_payload(payload, path, floor):
    arrays = {k: np.asarray(payload[k], dtype=float) for k in ('y_val', 'p_val', 'y_test', 'p_test')}
    for split in ('val', 'test'):
        y, p = arrays['y_'+split], arrays['p_'+split]
        if y.ndim != 1 or p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not set(np.unique(y)) <= {0,1} or np.any((p<0)|(p>1)):
            raise ValueError('Invalid labels or probabilities in experiment output.')
    stored = StoredRun(path.path_id, arrays['y_val'], arrays['p_val'], arrays['y_test'], arrays['p_test'],
                       int(payload['n_pos_before']), int(payload['n_pos_after']), int(payload['n_fit_rows']), str(payload['fit_note']))
    return score_run(stored, components=path.component_ids, precision_floor=floor).as_dict()


def validate_result(payload):
    for key in ('summary', 'data', 'limitations'):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f'Missing result field: {key}.')
    for key in ('measurements', 'baseline'):
        numbers = payload.get(key)
        if not isinstance(numbers, dict) or not numbers:
            raise ValueError(f'{key} must contain named numeric measurements.')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not np.isfinite(v) for v in numbers.values()):
            raise ValueError(f'{key} contains invalid measurements.')
    if set(payload['measurements']) != set(payload['baseline']):
        raise ValueError('Branch and baseline must report the same measurement names.')
    return payload


_UNFINISHED = re.compile(r'\bMissing:|Specify a testable|Draft plan', re.I)


def _finish_draft(branch, direction):
    """A placeholder plan can run as a labeled demonstration. A finished plan is left as written."""
    if branch.get('executor') == 'bench':
        return False
    if not any(_UNFINISHED.search(str(branch.get(key) or '')) for key in research.FIELDS[:-1]):
        return False
    question = ' '.join(str(direction or '').split())[:300]
    branch['hypothesis'] = f'A small demonstration of the cited method changes the outcome the direction asks about. Direction: {question}'
    branch['method'] = 'Implement the procedure named in the cited passage on a small explicit input, and label that input a demonstration.'
    branch['dataset'] = 'No public table was prepared. The script uses a small explicit input and says so.'
    branch['baseline'] = 'The same procedure on the simpler input named in the direction, with seed 11.'
    branch['evaluation'] = 'Report the measurements the direction asks for. This run is a demonstration, not a benchmark.'
    branch['assumptions'] = 'Inputs the abstract does not state are demonstration choices and are labeled as such.'
    branch['limitations'] = 'A small demonstration does not reproduce the paper’s full experiment.'
    branch['evidence_level'] = 'demonstration'
    branch['status'] = 'proposed'
    branch['data'] = None
    branch.pop('block_reason', None)
    return True


def write_script(branch, papers, data, settings):
    brief = []
    for paper in papers[:2]:
        excerpt = paper.quotes[0].text if paper.quotes else ' '.join(paper.abstract.split())[:400]
        brief.append({'id': paper.paper_id, 'title': paper.title, 'excerpt': excerpt})
    plan = {key: branch.get(key) for key in ('title', 'hypothesis', 'method', 'dataset', 'baseline', 'evaluation', 'evidence_level')}
    prompt = ('Return {"code":"<python>"} only. No comments. A short demonstration script. '
              'numpy and the standard library only. No network, subprocess, or file reads except inputs.npz when data is provided. '
              'Seed 11. Last line prints one JSON object with keys summary, measurements, baseline, data, limitations. '
              'measurements and baseline are numeric dicts with the same keys. '
              'For a graph demonstration, compare a random graph with a scale-free graph and report peak and time. '
              'Label the input a demonstration.\n' + json.dumps({'plan': plan, 'papers': brief, 'has_table': bool(data)}))
    payload = research.model_json(prompt, timeout=600, num_ctx=2048, num_predict=700)
    script = extract_script(payload.get('code', 'INSUFFICIENT'))
    forbidden = forbidden_from_rules(set(settings['rules']))
    tokens = {'smote': r'\bSMOTE\b', 'random_oversample': r'RandomOverSampler|random_oversample',
              'random_undersample': r'RandomUnderSampler|random_undersample', 'class_weight': r'class_weight\s*=\s*(?!None)',
              'threshold': r'threshold_sweep|threshold_grid'}
    if any(re.search(tokens[name], script, re.I) for name in forbidden if name in tokens):
        raise ValueError('Generated code contradicts a human constraint and was not executed.')
    return script


class ResearchWorkflow:
    def research_event(self):
        if "publication" in self.research:
            self.research["publication"]["network"] = solana_payments.network()
        self.research["settings"] = {"rules": sorted(self.rules), "metric": self.metric, "precision_floor": self.precision_floor, "guidance": self.research.get("instructions", [])}
        self.emit({'type': 'research', 'research': copy.deepcopy(self.research)})

    def set_phase(self, phase, detail):
        self.phase = phase
        self.emit({'type': 'status', 'phase': phase, 'detail': detail})

    def invalidate_approval(self):
        self.approved = False
        self.research['approval'] = None
        self.research['revision'] = self.research.get('revision', 0) + 1
        self.cancel_requested = True

    async def _research(self, direction):
        self.research.setdefault('branches', [])
        self.research.setdefault('runs', [])
        self.research.setdefault('history', [])
        await self._prepare_ideas(direction)
        await self._research_loop()

    async def _prepare_ideas(self, direction, parent=None):
        self.set_phase('branching', 'Reading evidence and proposing editable research plans.')
        version = self.research.get('revision', 0)
        kept = [p for p in self.papers if p.paper_id not in self.dropped]
        grounding = ground_papers(kept)
        self.grounding = grounding
        self.research['grounding'] = asdict(grounding) if grounding else None
        self.research['data'] = None
        self.research['data_error'] = None
        if grounding:
            try:
                self.research['data'] = await asyncio.to_thread(prepare_data, grounding, ROOT / 'experiments' / self.id / 'inputs')
            except Exception as exc:
                self.research['data_error'] = str(exc)
        branches, notes, review = await asyncio.to_thread(research.propose, direction, kept,
                                                  forbidden_from_rules(self.rules), self.metric, parent=parent, guidance=self.research.get("instructions", []))
        await self.checkpoint()
        if version != self.research.get('revision', 0):
            # A human changed source evidence during synthesis. Do not publish stale plans.
            return await self._prepare_ideas(self.direction, parent)
        self.research['literature_review'] = review
        for branch in branches:
            branch['data'] = copy.deepcopy(self.research['data'])
            branch['estimated_effort'] = 'Local CPU; at most 180 seconds per process, plus model writing. Bench includes a baseline run.'
            if branch['data']:
                branch['dataset'] = branch['data']['description']
            else:
                if branch.get('evidence_level') == 'demonstration':
                    branch['status'] = 'proposed'
                else:
                    branch['status'] = 'blocked'
                    branch['block_reason'] = self.research['data_error'] or 'No runnable data. Complete the plan and explicitly choose demonstration, or supply literature naming a supported dataset.'
        self.research['branches'].extend(branches)
        self.research['approval'] = None
        for note in notes:
            self.emit({'type': 'agent', 'text': note})
        self.set_phase('review', 'Edit and select ideas, then approve the exact plans to run.')
        self.research_event()
        self.start_conference_search()

    async def _research_loop(self):
        while True:
            await self.checkpoint()
            if self.approved and self.research.get('approval'):
                approval = copy.deepcopy(self.research['approval'])
                self.approved = False
                self.cancel_requested = False
                await self._run_approved(approval)
                await self.checkpoint()
                self.research['approval'] = None
                self.set_phase('done', 'Results are ready for your review. Revise, rerun, or branch from a result.')
                self.research_event()
                return
            await asyncio.sleep(0.2)

    async def _run_approved(self, approval):
        for plan in approval['plans']:
            if self.cancel_requested or self.redirect is not None or approval['revision'] != self.research.get('revision', 0):
                break
            await self.checkpoint()
            branch = next(b for b in self.research['branches'] if b['id'] == plan['id'])
            run_id = uuid.uuid4().hex[:12]
            run = {'id': run_id, 'branch_id': plan['id'], 'plan': plan, 'settings': approval['settings'],
                   'approval_id': approval['id'], 'status': 'running', 'started': time.time(), 'review': 'unreviewed',
                   'review_history': [], 'artifacts': [], 'writer': writer_name(), 'python': sys.version,
                   'evidence_level': plan['evidence_level']}
            self.research['runs'].append(run)
            branch['runs'].append(run_id)
            branch['status'] = 'running'
            self.set_phase('running', f'Running: {plan["title"]}')
            self.research_event()
            folder = ROOT / 'experiments' / self.id / run_id
            folder.mkdir(parents=True, exist_ok=True)
            import importlib.metadata
            run['dependencies'] = {name: importlib.metadata.version(name) for name in ('numpy', 'scikit-learn', 'scipy')}
            (folder / 'environment.json').write_text(json.dumps({'python': sys.version, 'dependencies': run['dependencies'], 'seed': 11}, indent=2))
            (folder / 'plan.json').write_text(json.dumps({'plan': plan, 'settings': approval['settings']}, indent=2))
            def cancelled():
                return self.cancel_requested or self.redirect is not None or approval['revision'] != self.research.get('revision', 0)
            try:
                if plan.get('data'):
                    input_path = ROOT / 'experiments' / self.id / 'inputs' / 'inputs.npz'
                    raw = input_path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != plan['data']['sha256']:
                        raise ValueError('Prepared data changed since approval. Approve a fresh plan.')
                    (folder / 'inputs.npz').write_bytes(raw)
                current_papers = [p for p in self.papers if p.paper_id not in self.dropped]
                cited_ids = {c['paper_id'] for c in plan['citations']}
                cited_papers = [p for p in current_papers if p.paper_id in cited_ids]
                if plan['executor'] == 'bench':
                    paths = plan_paths(current_papers, blocked_methods=forbidden_from_rules(self.rules))
                    path = next((p for p in paths if p.path_id == plan['path_id']), None)
                    if path is None or self.grounding is None:
                        raise ValueError('This bench method or dataset is no longer supported by kept papers.')
                    script = bench_script(path, self.grounding)
                    result = await execute(script, folder, cancelled)
                    scored = await asyncio.to_thread(scored_payload, result, path, approval['settings']['precision_floor'])
                    baseline_path = MethodPath('baseline', 'Baseline logistic regression', ['logreg'], [], [])
                    base_folder = folder / 'baseline'
                    base_folder.mkdir()
                    (base_folder / 'inputs.npz').write_bytes((folder / 'inputs.npz').read_bytes())
                    base = await execute(bench_script(baseline_path, self.grounding), base_folder, cancelled)
                    baseline = await asyncio.to_thread(scored_payload, base, baseline_path, approval['settings']['precision_floor'])
                    payload = {'summary': 'Exploratory comparison against an unmodified logistic regression baseline.',
                               'measurements': scored['metrics'], 'baseline': baseline['metrics'], 'details': scored,
                               'baseline_details': baseline, 'data': plan['data']['description'], 'limitations': plan['limitations']}
                else:
                    script = await asyncio.to_thread(write_script, plan, cited_papers, plan.get('data'), approval['settings'])
                    if cancelled():
                        raise RunCancelled('Plan changed or run cancelled while writing; script was not executed.')
                    payload = validate_result(await execute(script, folder, cancelled))
                if cancelled():
                    raise RunCancelled('Run invalidated by a human edit; no result was accepted.')
                run['result'] = payload
                run['comparison'] = {k: payload['measurements'][k] - payload['baseline'][k] for k in payload['measurements']}
                run['interpretation'] = 'Computed observations only. Hypothesis support and scientific validity await human review.'
                run['status'] = 'completed'
                (folder / 'result.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
            except asyncio.CancelledError:
                run.update(status='interrupted', error='Work stopped; fresh approval is required.')
                raise
            except RunCancelled as exc:
                run.update(status='cancelled', error=str(exc))
            except Exception as exc:
                run.update(status='failed', error=str(exc))
            finally:
                run['finished'] = time.time()
                run['artifacts'] = [str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and not p.is_symlink()]
                branch['status'] = run['status']
                self.research_event()

    async def handle_research_action(self, action):
        kind = action.get('type')
        if kind not in {'edit_branch', 'add_branch', 'review_run', 'follow_up', 'rerun', 'cancel', 'resume_saved', 'proceed', 'draft_demonstration'}:
            return False
        branches = self.research.get('branches', [])
        branch = next((b for b in branches if b['id'] == action.get('branch_id')), None)
        if kind == 'proceed':
            if not branches:
                return False  # Compatibility for the original bench API.
            if self.phase not in {'review', 'done', 'interrupted'} or self.paused:
                raise ValueError('Plans can be approved only while reviewing, with the investigation resumed.')
            if 'no_threshold' in self.rules and self.precision_floor is not None:
                raise ValueError('A precision floor requires threshold tuning. Remove the no threshold constraint or choose a metric to clear the floor.')
            selected = [b for b in branches if b['selected'] and b['decision'] == 'selected']
            if not selected:
                raise ValueError('Select at least one idea.')
            kept_ids = {p.paper_id for p in self.papers if p.paper_id not in self.dropped}
            for b in selected:
                if b.get('requires_edit'):
                    raise ValueError(f'{b["title"]}: specify concrete inputs, method and evaluation in the editor before approval.')
                if not b['citations'] or any(c['paper_id'] not in kept_ids for c in b['citations']):
                    raise ValueError(f'{b["title"]}: a supporting paper was removed. Edit its evidence first.')
                if any(not b.get(k) or re.search(r'\bMissing:|Specify a testable|Draft plan', b[k], re.I) for k in research.FIELDS[:-1]):
                    raise ValueError(f'{b["title"]}: complete every plan field before approval.')
                if not b.get('data') and b['evidence_level'] != 'demonstration':
                    raise ValueError(f'{b["title"]}: missing data. Only an explicitly approved demonstration may run without a prepared table.')
                if b['executor'] == 'bench' and b['path_id'] not in {p.path_id for p in plan_paths([p for p in self.papers if p.paper_id in kept_ids], blocked_methods=forbidden_from_rules(self.rules))}:
                    raise ValueError('This bench method is excluded by current papers or constraints.')
                if b['executor'] == 'bench' and (not b.get('data') or b['evidence_level'] != 'exploratory'):
                    raise ValueError('Bench approximations require a prepared table and exploratory evidence level.')
                if 'no_synthetic' in self.rules and (b['evidence_level'] == 'demonstration' or (b.get('data') or {}).get('kind') == 'synthesis'):
                    raise ValueError('The no synthetic samples constraint excludes this input. Revise the plan or constraint.')
            approval = {'id': uuid.uuid4().hex, 'time': time.time(), 'revision': self.research.get('revision', 0),
                        'plans': copy.deepcopy(selected), 'settings': {'rules': sorted(self.rules), 'metric': self.metric,
                                                                     'precision_floor': self.precision_floor, 'seed': 11, 'guidance': copy.deepcopy(self.research.get('instructions', []))}}
            self.research['approval'] = approval
            self.research['history'].append({'type': 'approval', **copy.deepcopy(approval)})
            self.approved = True
            self.cancel_requested = False
            self.research_event()
            if not self.task or self.task.done():
                self.task = asyncio.create_task(self._resume_task())
        elif kind == 'draft_demonstration':
            if branch is None or self.phase not in {'review', 'done', 'interrupted'}:
                raise ValueError('Choose an idea while reviewing plans.')
            self.invalidate_approval()
            self.research['history'].append({'type': 'draft_demonstration', 'before': copy.deepcopy(branch), 'time': time.time()})
            if not _finish_draft(branch, self.direction):
                branch['evidence_level'] = 'demonstration'
                branch['executor'] = 'model'
                branch['data'] = None
            branch['requires_edit'] = True
            branch['version'] += 1
            branch['block_reason'] = 'Demonstration draft prepared. Specify the actual small inputs and method before approval.'
            self.set_phase('review', 'Review and complete the demonstration plan; nothing has been approved or executed.')
        elif kind == 'edit_branch':
            if branch is None:
                raise ValueError('Unknown idea.')
            patch = action.get('patch', {})
            if not isinstance(patch, dict) or set(patch) - set(research.FIELDS) - {'decision', 'paper_ids'}:
                raise ValueError('Unknown plan edit field.')
            normalized = {}
            for key in research.FIELDS:
                if key in patch:
                    value = str(patch[key]).strip()
                    if not value or len(value) > 6000:
                        raise ValueError('Plan fields must contain 1–6000 characters.')
                    if key == 'evidence_level' and value not in research.LEVELS:
                        raise ValueError('Unknown evidence level.')
                    normalized[key] = value
            decision = patch.get('decision', branch['decision'])
            if decision not in {'selected', 'rejected', 'deferred'}:
                raise ValueError('Unknown idea decision.')
            cites = branch['citations']
            if 'paper_ids' in patch:
                if not isinstance(patch['paper_ids'], list) or not patch['paper_ids']:
                    raise ValueError('Choose supporting paper IDs.')
                kept = [p for p in self.papers if p.paper_id not in self.dropped and p.paper_id in patch['paper_ids']]
                if {p.paper_id for p in kept} != set(patch['paper_ids']):
                    raise ValueError('A supporting paper is unknown or removed.')
                cites = [research.citation(p) for p in kept]
            before = copy.deepcopy(branch)
            self.invalidate_approval()
            self.research['history'].append({'type': 'edit', 'before': before, 'time': time.time()})
            branch.update(normalized)
            branch.update(citations=cites, decision=decision, selected=decision == 'selected', version=branch['version']+1, status='proposed')
            branch.pop('block_reason', None)
            if any(value != before.get(key) for key, value in normalized.items()):
                branch['executor'] = 'model'
                if all(key in normalized for key in ('method', 'dataset', 'evaluation')):
                    branch['requires_edit'] = False
            if 'dataset' in normalized and normalized['dataset'] != before['dataset']:
                # A human-proposed input differs from the prepared table. Never silently reuse that table.
                branch['data'] = None
            self.set_phase('review' if self.phase != 'running' else self.phase, 'Plan edited; approval must be renewed.')
        elif kind == 'add_branch':
            if len(branches) >= 20:
                raise ValueError('Limit this investigation to 20 active ideas; redirect for another iteration.')
            patch = action.get('plan', {})
            if not isinstance(patch, dict):
                raise ValueError('A plan must be an object.')
            ids = action.get('paper_ids', [])
            kept = [p for p in self.papers if p.paper_id in ids and p.paper_id not in self.dropped]
            if not kept or {p.paper_id for p in kept} != set(ids):
                raise ValueError('Choose at least one valid kept paper for the idea.')
            self.invalidate_approval()
            added = research.idea(patch, [research.citation(p) for p in kept], parent_id=action.get('parent_id'))
            added['data'] = copy.deepcopy(self.research.get('data'))
            added['estimated_effort'] = 'Local CPU, 180-second process limit plus model writing.'
            branches.append(added)
            self.research['branches'] = branches
            self.set_phase('review', 'Your idea is ready to edit and approve.')
        elif kind == 'review_run':
            run = next((r for r in self.research.get('runs', []) if r['id'] == action.get('run_id')), None)
            decision = action.get('decision')
            if run is None or run['status'] != 'completed' or decision not in {'accepted', 'needs_revision', 'rejected'}:
                raise ValueError('Choose a completed run and a valid review decision.')
            entry = {'decision': decision, 'note': str(action.get('note', ''))[:6000], 'time': time.time()}
            run['review'] = decision
            run['review_history'].append(entry)
        elif kind == 'follow_up':
            if self.task and not self.task.done():
                raise ValueError('Finish or cancel the active work before proposing follow-ups.')
            run = next((r for r in self.research.get('runs', []) if r['id'] == action.get('run_id')), None)
            if run is None or run['status'] != 'completed':
                raise ValueError('Choose a completed result for follow-up ideas.')
            for b in branches:
                b['selected'] = False
                b['decision'] = 'deferred'
            self.invalidate_approval()
            self.cancel_requested = False
            self.task = asyncio.create_task(self._resume_task(mode='follow_up', run=run, note=str(action.get('note', ''))[:6000]))
        elif kind == 'rerun':
            if branch is None or (self.task and not self.task.done() and self.phase != 'review'):
                raise ValueError('Choose an idea after active work finishes.')
            self.invalidate_approval()
            for b in branches:
                b['selected'] = b['id'] == branch['id']
                b['decision'] = 'selected' if b['selected'] else 'deferred'
            self.set_phase('review', 'Review the plan, then approve a new run. Previous results remain saved.')
        elif kind == 'cancel':
            if self.publication_task and not self.publication_task.done():
                self.publication_task.cancel()
            self.cancel_requested = True
            self.approved = False
            if self.phase != 'running' and self.task and not self.task.done():
                self.task.cancel()
                self.set_phase('interrupted', 'Work cancelled. Saved plans and results remain available.')
            self.emit({'type': 'agent', 'text': 'Cancellation requested. Active experiment processes stop and partial logs are retained.'})
        elif kind == 'resume_saved':
            self.paused = False
            if self.task and not self.task.done():
                raise ValueError('This investigation already has active work.')
            if branches:
                self.set_phase('review', 'Restored plans need fresh approval before execution.')
                self.task = asyncio.create_task(self._resume_task())
            elif self.papers:
                self.task = asyncio.create_task(self._resume_task(mode='papers'))
            else:
                self.start()
        self.research_event()
        return True

    async def _resume_task(self, mode='run', run=None, note=''):
        from .session import Restart
        try:
            if mode == 'papers':
                await self._wait_for_papers()
                await self._prepare_ideas(self.direction)
            elif mode == 'follow_up':
                await self._prepare_ideas(self.direction + '\nHuman follow-up: ' + note, parent=run)
            await self._research_loop()
        except Restart as restart:
            self.research['history'].append({'type': 'iteration', 'direction': self.direction,
                                             'branches': self.research['branches'], 'time': time.time()})
            self.direction, self.query_override = restart.direction, restart.query
            self.research['branches'] = []
            self.research['approval'] = None
            self.research['data'] = None
            self.research['literature_review'] = None
            self.research['revision'] = self.research.get('revision', 0) + 1
            self.approved = False
            self.papers = []
            self.dropped = set()
            self.grounding = None
            self.emit({'type': 'reset', 'detail': 'Starting a new literature iteration; previous results remain saved.'})
            await self._task()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.set_phase('interrupted', str(exc))

