"""Conference matching, human-reviewed manuscripts, and wallet-approved registration fees."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import time
import uuid

from . import conferences, solana_payments as solana

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def run_digest(run):
    return digest({'plan': run['plan'], 'result': run['result']})


def manuscript_text(title, authors, runs, papers):
    lines = [f'# {title}', '', authors, '', '## Abstract', '',
             'This research draft reports the reviewed computational experiments below. '
             'The authors must write and verify the contribution, context, and interpretation before submission.', '',
             '## Introduction and related work', '',
             'Author revision required: explain the question, motivation, contribution and relationship to the cited literature.', '']
    for run in runs:
        plan, result = run['plan'], run['result']
        lines += [f'## Experiment: {plan["title"]}', '', f'**Hypothesis:** {plan["hypothesis"]}', '',
                  '### Method', '', plan['method'], '', f'**Baseline:** {plan["baseline"]}', '',
                  f'**Evaluation:** {plan["evaluation"]}', '', f'**Data:** {result["data"]}', '',
                  f'**Evidence level:** {run["evidence_level"]}', '', '### Results', '',
                  '| Measurement | Experiment | Baseline |', '|---|---:|---:|']
        for metric, value in result['measurements'].items():
            lines.append(f'| {metric} | {value} | {result["baseline"].get(metric, "not reported")} |')
        lines += ['', '### Limitations', '', result['limitations'], '', plan['limitations'], '',
                  f'**Reproducibility:** run `{run["id"]}`, approved plan version {plan["version"]}, seed {run["settings"]["seed"]}.', '']
    lines += ['## Discussion and conclusion', '', 'Author revision required: interpret the observations without overstating their scope.', '', '## References', '']
    cited = {c['paper_id'] for r in runs for c in r['plan']['citations']}
    lines += [f'- {p.title}. {p.link}' for p in papers if p.paper_id in cited]
    return '\n'.join(lines) + '\n'


class PublicationWorkflow:
    def publication_state(self):
        return self.research.setdefault('publication', {'conferences': [], 'search_status': 'idle', 'searches': [],
                                                       'manuscripts': [], 'payments': [], 'selected_conference_id': None})

    def publication_event(self):
        self.publication_state()['network'] = solana.network()
        self.research_event()

    def selected_conference(self):
        state = self.publication_state()
        return next((c for c in state['conferences'] if c['id'] == state['selected_conference_id']), None)

    def active_manuscript(self):
        state = self.publication_state()
        return next((m for m in state['manuscripts'] if m['id'] == state.get('active_manuscript_id')), None)

    def ready_manuscript(self):
        manuscript = self.active_manuscript()
        if not manuscript or manuscript['status'] != 'reviewed' or manuscript['sha256'] != hashlib.sha256(manuscript['content'].encode()).hexdigest():
            raise ValueError('Create and review the current manuscript before preparing a registration payment.')
        runs = {r['id']: r for r in self.research.get('runs', [])}
        for rid, expected in manuscript['run_hashes'].items():
            run = runs.get(rid)
            if not run or run.get('review') != 'accepted' or run_digest(run) != expected:
                raise ValueError('A manuscript result changed or is no longer accepted. Review a new draft.')
        return manuscript

    def start_conference_search(self, source_url=None):
        if self.publication_task and not self.publication_task.done():
            return
        self.publication_task = asyncio.create_task(self._conference_search(source_url))

    async def _conference_search(self, source_url=None):
        state = self.publication_state()
        branches = copy.deepcopy(self.research.get('branches', []))
        if not branches:
            return
        fingerprint = [(b['id'], b['version']) for b in branches]
        state['search_status'] = 'searching'
        self.publication_event()
        try:
            found = await asyncio.to_thread(conferences.discover, self.direction, branches, source_url)
            current = [(b['id'], b['version']) for b in self.research.get('branches', [])]
            if current != fingerprint:
                state['search_status'] = 'outdated'
                return
            state['searches'].append(found)
            prior = {c['id']: c for c in state['conferences']}
            for candidate in found['candidates']:
                if candidate['id'] in prior:
                    candidate['organizer_confirmed'] = prior[candidate['id']].get('organizer_confirmed', False)
                    candidate['acceptance'] = prior[candidate['id']].get('acceptance')
                prior[candidate['id']] = candidate
            state['conferences'] = list(prior.values())
            state['search_status'] = 'done'
        except asyncio.CancelledError:
            state['search_status'] = 'interrupted'
            raise
        except Exception as exc:
            state['search_status'] = 'failed'
            state['search_error'] = f'{type(exc).__name__}: {exc}'
        finally:
            self.publication_event()

    def usable_intent(self, payment_id, *, require_ready=True):
        intent = next((p for p in self.publication_state()['payments'] if p['id'] == payment_id), None)
        if intent is None:
            raise ValueError('Unknown payment request.')
        if intent['network'] != solana.network():
            raise ValueError('Payment network changed; do not sign this request.')
        if require_ready:
            manuscript = self.ready_manuscript()
            conference = self.selected_conference()
            if not conference or conference['id'] != intent['conference_id'] or manuscript['sha256'] != intent['manuscript_hash']:
                raise ValueError('Conference or manuscript changed. Do not sign the old payment request.')
            acceptance = conference.get('acceptance') or {}
            if acceptance.get('manuscript_hash') != manuscript['sha256']:
                raise ValueError('Record acceptance for the current manuscript before payment.')
            if intent['expires_at'] < time.time():
                raise ValueError('Payment request expired; verify any submitted transaction before preparing another request.')
        return intent

    async def handle_publication_action(self, action):
        kind = action.get('type')
        if kind not in {'find_conferences', 'select_conference', 'create_manuscript', 'save_manuscript', 'review_manuscript',
                        'record_acceptance', 'prepare_registration', 'cancel_payment', 'verify_payment'}:
            return False
        state = self.publication_state()
        if kind == 'find_conferences':
            if not self.research.get('branches'):
                raise ValueError('Generate research ideas before matching conferences.')
            source_url = str(action.get('source_url') or '').strip() or None
            self.start_conference_search(source_url)
        elif kind == 'select_conference':
            candidate = next((c for c in state['conferences'] if c['id'] == action.get('conference_id')), None)
            if not candidate or action.get('organizer_confirmed') is not True:
                raise ValueError('Confirm that you reviewed the official organizer and current conference edition.')
            candidate['organizer_confirmed'] = True
            state['selected_conference_id'] = candidate['id']
        elif kind == 'create_manuscript':
            runs = [r for r in self.research.get('runs', []) if r.get('status') == 'completed' and r.get('review') == 'accepted']
            if not runs:
                raise ValueError('Accept at least one completed experimental result before drafting a manuscript.')
            title = str(action.get('title', '')).strip()[:300]
            authors = str(action.get('authors', '')).strip()[:1000]
            if not title or not authors:
                raise ValueError('Enter a manuscript title and author line.')
            content = manuscript_text(title, authors, runs, self.papers)
            manuscript = {'id': uuid.uuid4().hex[:12], 'title': title, 'authors': authors, 'content': content,
                          'sha256': hashlib.sha256(content.encode()).hexdigest(), 'status': 'draft', 'version': 1,
                          'run_hashes': {r['id']: run_digest(r) for r in runs}, 'history': [], 'created_at': time.time()}
            state['manuscripts'].append(manuscript)
            state['active_manuscript_id'] = manuscript['id']
        elif kind in {'save_manuscript', 'review_manuscript'}:
            manuscript = self.active_manuscript()
            if not manuscript or action.get('manuscript_id') != manuscript['id']:
                raise ValueError('Choose the current manuscript.')
            if action.get('version') != manuscript['version']:
                raise ValueError('This manuscript changed. Reopen the current version before saving or reviewing.')
            if kind == 'save_manuscript':
                content = str(action.get('content', '')).strip()
                if not 100 <= len(content) <= 100000:
                    raise ValueError('Manuscript must contain 100–100000 characters.')
                manuscript['history'].append({'version': manuscript['version'], 'content': manuscript['content'], 'sha256': manuscript['sha256']})
                manuscript.update(content=content, sha256=hashlib.sha256(content.encode()).hexdigest(), status='draft', version=manuscript['version'] + 1)
            else:
                if action.get('confirmed') is not True or 'Author revision required:' in manuscript['content']:
                    raise ValueError('Complete the draft sections and confirm author review before marking it ready.')
                prior_status = manuscript['status']
                manuscript['status'] = 'reviewed'
                try:
                    self.ready_manuscript()
                except ValueError:
                    manuscript['status'] = prior_status
                    raise
                manuscript['reviewed_at'] = time.time()
        elif kind == 'record_acceptance':
            manuscript = self.ready_manuscript()
            conference = self.selected_conference()
            if not conference or action.get('confirmed') is not True:
                raise ValueError('Select a conference and confirm acceptance before recording it.')
            reference = str(action.get('reference', '')).strip()[:1000]
            if not reference:
                raise ValueError('Enter the organizer’s acceptance/submission reference. Do not paste confidential correspondence.')
            conference['acceptance'] = {'manuscript_hash': manuscript['sha256'], 'reference': reference,
                                        'recorded_at': time.time(), 'verified_by': 'human'}
        elif kind == 'prepare_registration':
            manuscript = self.ready_manuscript()
            conference = self.selected_conference()
            if not conference or (conference.get('acceptance') or {}).get('manuscript_hash') != manuscript['sha256']:
                raise ValueError('Record acceptance of this manuscript by the selected conference first.')
            expected_hash = manuscript['sha256']
            expected_conference_id = conference['id']
            invoice_url = str(action.get('invoice_url', '')).strip()
            mode = solana.network()
            if mode == 'devnet':
                if action.get('demo_confirmed') is not True:
                    raise ValueError('Confirm this is a devnet test; it does not purchase conference registration.')
                invoice = solana.parse_invoice(invoice_url)
                evidence = {'url': None, 'checked_at': time.time(), 'type': 'human-specified devnet demo'}
            else:
                if action.get('organizer_payment_confirmed') is not True:
                    raise ValueError('Confirm the official organizer issued this Solana payment invoice.')
                try:
                    invoice, evidence = await asyncio.to_thread(solana.invoice_evidence, invoice_url, str(action.get('source_url', '')))
                except ValueError:
                    raise
                except Exception as exc:
                    raise ValueError('The organizer payment page could not be checked; no request was prepared.') from exc
            # Recheck after network I/O. A concurrent edit cannot authorize a stale payment.
            if self.ready_manuscript()['sha256'] != expected_hash or self.selected_conference()['id'] != expected_conference_id:
                raise ValueError('Publication changed while checking the invoice. Review it again.')
            previous = [p for p in state['payments'] if p['conference_id'] == conference['id'] and p['manuscript_id'] == manuscript['id'] and p['status'] != 'cancelled']
            if previous:
                raise ValueError('A registration request already exists. Verify it or explicitly cancel an unsigned request before creating another.')
            intent = solana.create_intent(invoice, evidence, conference, manuscript, 'registration')
            state['payments'].append(intent)
        elif kind == 'cancel_payment':
            intent = self.usable_intent(action.get('payment_id'), require_ready=False)
            if intent['status'] != 'awaiting_wallet':
                raise ValueError('Only an unsigned request can be cancelled. Check the receipt for wallet-requested or submitted payments.')
            intent['status'] = 'cancelled'
        elif kind == 'verify_payment':
            intent = self.usable_intent(action.get('payment_id'), require_ready=False)
            if intent['status'] == 'cancelled':
                raise ValueError('This request was cancelled.')
            signature = str(action.get('signature') or intent.get('signature') or '')
            try:
                receipt = await asyncio.to_thread(solana.verify_receipt, copy.deepcopy(intent), signature)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError('The receipt could not be checked. Do not pay again; retry verification when RPC is available.') from exc
            if receipt:
                if any(p['id'] != intent['id'] and p.get('signature') == signature and p['status'] == 'confirmed' for p in state['payments']):
                    raise ValueError('This transaction has already been used for another invoice.')
                intent.update(status='confirmed', signature=signature, receipt=receipt)
            else:
                intent['verification_notice'] = 'Transaction is not finalized yet. No payment has been confirmed; check again without paying twice.'
        self.publication_event()
        return True
