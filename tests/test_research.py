"""Integration coverage for human control, branch provenance and durable recovery."""
import asyncio
import copy
import json
from pathlib import Path

import pytest

from backend import research, storage, workflow
from backend.session import Investigation, INVESTIGATIONS, restore_saved
from backend.sources import Paper


def paper(pid='p'):
    return Paper(pid, 'A general calibration study', ['Author'], '2025-01-01',
                 'We study calibration with temperature scaling. We evaluate reliability on held-out predictions.',
                 'https://example.org/' + pid, None, ['cs.LG'], None, 'arxiv')


def plan(p=None, **patch):
    p = p or paper()
    values = dict(title='Compare calibration', hypothesis='Temperature scaling reduces calibration error.',
                  method='Fit a temperature on validation predictions, then compare on held-out predictions.',
                  dataset='Demonstration: probabilities [0.1, 0.8, 0.7, 0.2] and labels [0, 1, 0, 0].',
                  baseline='Uncalibrated probabilities on the same test inputs.', evaluation='Compare Brier score.',
                  assumptions='Small inputs are researcher-chosen; no benchmark claim.', limitations='Toy inputs cannot establish external validity.',
                  evidence_level='demonstration')
    values.update(patch)
    branch = research.idea(values, [research.citation(p)])
    branch['data'] = None
    return branch


def desk():
    i = Investigation('Compare calibration methods')
    i.papers = [paper()]
    i.phase = 'review'
    i.research['branches'] = [plan()]
    return i


def result():
    return dict(summary='The toy procedure changed Brier score.', measurements={'brier': .2}, baseline={'brier': .3},
                data='Explicit researcher-chosen demonstration inputs.', limitations='Toy example only.')


def test_planner_uses_complete_sources_and_verifies_every_citation(monkeypatch):
    p = paper()
    received = []
    def model(prompt):
        received.append(prompt)
        valid = {k: v for k, v in plan(p).items() if k in research.FIELDS}
        valid['citations'] = [{'paper_id': p.paper_id, 'quote': 'We study calibration with temperature scaling.'}]
        fake = copy.deepcopy(valid)
        fake['citations'][0]['quote'] = 'This invented finding is not in the source.'
        return {'ideas': [valid, fake]}
    monkeypatch.setattr(research, 'model_json', model)
    ideas, notes, review = research.propose('calibration', [p], set(), 'accuracy')
    assert len(ideas) == 1
    assert p.abstract in received[0]
    assert ideas[0]['citations'][0]['paper_id'] == p.paper_id
    assert any('Discarded' in n for n in notes)


def test_planner_failure_preserves_editable_evidence_without_inventing_results(monkeypatch):
    monkeypatch.setattr(research, 'model_json', lambda _: (_ for _ in ()).throw(RuntimeError('offline')))
    ideas, notes, review = research.propose('calibration', [paper()], set(), 'accuracy')
    assert ideas and notes
    assert ideas[0]['requires_edit'] is True
    assert ideas[0]['evidence_level'] == 'demonstration'
    assert not ideas[0]['runs']


def test_selected_branch_is_the_exact_plan_executed_and_results_are_linked(monkeypatch, tmp_path):
    async def scenario():
        i = desk()
        rejected = plan(title='Rejected idea')
        rejected.update(decision='rejected', selected=False)
        i.research['branches'].append(rejected)
        captured = []
        def writer(branch, papers, data, settings):
            captured.append(copy.deepcopy(branch))
            assert settings['metric'] == 'average_precision'
            return 'print("result")'
        async def execute(script, folder, cancelled):
            assert not cancelled()
            return result()
        monkeypatch.setattr(workflow, 'write_script', writer)
        monkeypatch.setattr(workflow, 'execute', execute)
        monkeypatch.setattr(workflow, 'ROOT', tmp_path)
        await i.handle(action={'type': 'proceed'})
        await asyncio.wait_for(i.task, 3)
        assert len(captured) == 1
        assert captured[0]['id'] == i.research['branches'][0]['id']
        run = i.research['runs'][0]
        assert run['plan'] == captured[0]
        assert run['branch_id'] == captured[0]['id']
        assert run['comparison']['brier'] == pytest.approx(-.1)
        assert run['review'] == 'unreviewed'
        assert run['status'] == 'completed'
        assert 'environment.json' in run['artifacts']
        await i.handle(action={'type': 'review_run', 'run_id': run['id'], 'decision': 'needs_revision', 'note': 'Use real data next.'})
        assert run['review_history'][0]['note'] == 'Use real data next.'
        await i.handle(action={'type': 'rerun', 'branch_id': captured[0]['id']})
        assert i.phase == 'review' and not i.approved
        assert len(i.research['runs']) == 1
    asyncio.run(scenario())


def test_edit_during_writing_invalidates_approval_and_never_executes(monkeypatch, tmp_path):
    async def scenario():
        import threading
        started, finish = threading.Event(), threading.Event()
        i = desk()
        branch = i.research['branches'][0]
        def writer(*args):
            started.set()
            finish.wait(3)
            return 'print("stale")'
        async def execute(*args):
            raise AssertionError('Stale code executed')
        monkeypatch.setattr(workflow, 'write_script', writer)
        monkeypatch.setattr(workflow, 'execute', execute)
        monkeypatch.setattr(workflow, 'ROOT', tmp_path)
        await i.handle(action={'type': 'proceed'})
        while not started.is_set():
            await asyncio.sleep(.01)
        approved = copy.deepcopy(i.research['approval'])
        await i.handle(action={'type': 'edit_branch', 'branch_id': branch['id'], 'patch': {'hypothesis': 'A revised human hypothesis.'}})
        finish.set()
        await asyncio.wait_for(i.task, 3)
        assert i.research['runs'][0]['status'] == 'cancelled'
        assert approved['plans'][0]['hypothesis'] != branch['hypothesis']
        assert i.research['approval'] is None
    asyncio.run(scenario())


def test_dropped_evidence_cannot_be_approved():
    async def scenario():
        i = desk()
        await i.handle(action={'type': 'drop_paper', 'paper_id': 'p'})
        assert not i.research['branches'][0]['selected']
        with pytest.raises(ValueError, match='Select'):
            await i.handle(action={'type': 'proceed'})
    asyncio.run(scenario())


def test_constraints_block_synthetic_inputs_and_invalid_floor():
    async def scenario():
        i = desk()
        await i.handle(action={'type': 'rule', 'rule': 'no_synthetic'})
        with pytest.raises(ValueError, match='synthetic'):
            await i.handle(action={'type': 'proceed'})
        with pytest.raises(ValueError, match='between'):
            await i.handle(action={'type': 'precision_floor', 'value': float('nan')})
        assert i.precision_floor is None
    asyncio.run(scenario())


def test_bad_edit_is_atomic_and_dataset_change_does_not_silently_reuse_input():
    async def scenario():
        i = desk()
        branch = i.research['branches'][0]
        before = copy.deepcopy(branch)
        with pytest.raises(ValueError):
            await i.handle(action={'type': 'edit_branch', 'branch_id': branch['id'], 'patch': {'title': 'would change', 'method': ''}})
        assert branch == before
        branch['data'] = {'sha256': 'old'}
        await i.handle(action={'type': 'edit_branch', 'branch_id': branch['id'], 'patch': {'dataset': 'A different explicit demonstration input.'}})
        assert branch['data'] is None
    asyncio.run(scenario())


def test_restart_restores_sources_results_and_needs_fresh_approval(monkeypatch, tmp_path):
    monkeypatch.setenv('MARGINALIA_DB', str(tmp_path / 'research.db'))
    i = desk()
    i.persistent = True
    i.phase = 'running'
    i.research['approval'] = {'id': 'old-approval'}
    branch = i.research['branches'][0]
    branch['status'] = 'running'
    i.research['runs'].append({'id': 'r', 'branch_id': branch['id'], 'status': 'running'})
    i.emit({'type': 'human', 'text': 'Keep the baseline unchanged.'})
    monkeypatch.setitem(INVESTIGATIONS, i.id, i)
    del INVESTIGATIONS[i.id]
    restore_saved()
    restored = INVESTIGATIONS[i.id]
    try:
        assert restored.papers[0].abstract == i.papers[0].abstract
        assert restored.phase == 'interrupted'
        assert restored.research['approval'] is None
        assert restored.research['runs'][0]['status'] == 'interrupted'
        assert restored.research['branches'][0]['status'] == 'interrupted'
        assert restored.task is None
        assert any(e.get('text') == 'Keep the baseline unchanged.' for e in restored.events)
    finally:
        del INVESTIGATIONS[i.id]


def test_result_schema_rejects_nonfinite_missing_or_unmatched_measurements():
    assert workflow.validate_result(result())
    for patch in ({'measurements': {'brier': float('nan')}}, {'baseline': {}}, {'baseline': {'accuracy': .5}}, {'data': ''}):
        with pytest.raises(ValueError):
            workflow.validate_result({**result(), **patch})


def test_no_automatic_run_after_adding_a_human_idea():
    async def scenario():
        i = desk()
        await i.handle(action={'type': 'add_branch', 'paper_ids': ['p'], 'plan': {k: v for k,v in plan().items() if k in research.FIELDS}})
        assert len(i.research['branches']) == 2
        assert i.phase == 'review' and not i.approved and i.task is None
    asyncio.run(scenario())


def test_unsupported_platform_fails_closed(monkeypatch, tmp_path):
    from backend import runner
    monkeypatch.setattr(runner.platform, 'system', lambda: 'unsupported')
    with pytest.raises(RuntimeError, match='No supported OS sandbox'):
        runner.sandbox_command(tmp_path)


def test_artifacts_cannot_escape_run_directory(monkeypatch, tmp_path):
    from backend import main
    from fastapi import HTTPException
    i = desk()
    i.research['runs'] = [{'id': 'r', 'artifacts': ['../secret', 'result.json']}]
    monkeypatch.setitem(INVESTIGATIONS, i.id, i)
    monkeypatch.setattr(main, 'ROOT', tmp_path)
    with pytest.raises(HTTPException):
        main.run_artifact(i.id, 'r', '../secret')
    with pytest.raises(HTTPException):
        main.run_artifact(i.id, 'unknown', 'result.json')


def test_general_human_guidance_is_saved_and_invalidates_previous_approval():
    async def scenario():
        i = desk()
        i.research['approval'] = {'id': 'prior'}
        await i.handle(text='Compare calibration quality across three random seeds.')
        assert i.research['instructions'] == ['Compare calibration quality across three random seeds.']
        assert i.research['approval'] is None
        assert i.research['settings']['guidance'] == i.research['instructions']
    asyncio.run(scenario())


def test_review_synthesis_drops_unsupported_passages(monkeypatch):
    p = paper()
    monkeypatch.setattr(research, 'model_json', lambda _: {'ideas': [], 'literature_review': {
        'summary': 'Calibration is studied.', 'limitations': 'Abstract evidence only.', 'findings': [
            {'claim': 'The paper studies calibration.', 'paper_id': p.paper_id, 'quote': 'We study calibration with temperature scaling.'},
            {'claim': 'An invented result.', 'paper_id': p.paper_id, 'quote': 'It improves everything by 40 percent.'}]}})
    _, _, review = research.propose('calibration', [p], set(), 'accuracy')
    assert len(review['findings']) == 1
    assert review['findings'][0]['citation']['paper_id'] == p.paper_id


def test_redirect_keeps_prior_iteration_and_run_history():
    async def scenario():
        i = desk()
        original_direction = i.direction
        old_branch = copy.deepcopy(i.research['branches'][0])
        i.research['runs'] = [{'id': 'old', 'status': 'completed'}]
        # Suppress actual search while inspecting the redirect transition.
        i.start = lambda: None
        await i.handle(action={'type': 'redirect', 'direction': 'focus on temperature scaling'})
        assert i.research['history'][-1]['direction'] == original_direction
        assert i.research['history'][-1]['branches'][0]['id'] == old_branch['id']
        assert i.research['branches'] == []
        assert i.research['runs'][0]['id'] == 'old'
    asyncio.run(scenario())


def test_incomplete_plan_is_not_rewritten_or_executed_at_approval():
    async def scenario():
        i = desk()
        branch = i.research['branches'][0]
        branch['requires_edit'] = True
        before = copy.deepcopy(branch)
        with pytest.raises(ValueError, match='specify concrete inputs'):
            await i.handle(action={'type': 'proceed'})
        assert branch == before and i.task is None and not i.approved
        await i.handle(action={'type': 'draft_demonstration', 'branch_id': branch['id']})
        assert branch['requires_edit'] is True
        assert i.phase == 'review' and not i.approved and i.task is None
    asyncio.run(scenario())


def test_saved_work_redirect_enters_a_new_search_and_preserves_prior_ideas():
    async def scenario():
        i = desk()
        old_id = i.research['branches'][0]['id']
        i.redirect = ('A new calibration focus', None)
        searched = []
        async def task():
            searched.append(i.direction)
        i._task = task
        await i._resume_task()
        assert searched == ['A new calibration focus']
        assert i.research['branches'] == []
        assert i.research['history'][-1]['branches'][0]['id'] == old_id
    asyncio.run(scenario())


def test_complete_demo_question_fits_start_request():
    from backend.main import StartBody
    question = 'Temperature scaling demonstration. ' * 40
    assert len(question) > 500
    assert StartBody(direction=question).direction == question
    with pytest.raises(ValueError):
        StartBody(direction='x' * 6001)
