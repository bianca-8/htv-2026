import asyncio
import copy
import hashlib
import json
import socket
import time

import pytest

from backend import conferences, publication, solana_payments as solana
from backend.session import Investigation
from backend.main import WalletBody, SignatureBody, wallet_request, payment_submitted, INVESTIGATIONS

RECIPIENT = solana.b58encode(bytes(range(1,33)))
PAYER = solana.b58encode(bytes(range(2,34)))
SIGNATURE = solana.b58encode(bytes(range(64)))


def investigation():
    i = Investigation('Calibrated classifiers')
    run = {'id': 'run1', 'status': 'completed', 'review': 'accepted', 'evidence_level': 'exploratory',
           'plan': {'title': 'Calibration test', 'hypothesis': 'Calibration reduces error.', 'method': 'Fit on training only.',
                    'baseline': 'Uncalibrated predictions.', 'evaluation': 'Brier score.', 'limitations': 'One split.', 'version': 1, 'citations': []},
           'result': {'data': 'Named table.', 'measurements': {'brier': .1}, 'baseline': {'brier': .2}, 'limitations': 'One seed.'},
           'settings': {'seed': 11}}
    i.research['runs'] = [run]
    state = i.publication_state()
    state['conferences'] = [{'id': 'conf', 'name': 'Test Conference', 'url': 'https://conference.example.org', 'organizer_confirmed': False}]
    return i


async def ready(i):
    await i.handle(action={'type': 'select_conference', 'conference_id': 'conf', 'organizer_confirmed': True})
    await i.handle(action={'type': 'create_manuscript', 'title': 'A calibration study', 'authors': 'Researcher'})
    m = i.active_manuscript()
    await i.handle(action={'type': 'save_manuscript', 'manuscript_id': m['id'], 'version': m['version'],
                           'content': m['content'].replace('Author revision required:', 'Author interpretation:')})
    await i.handle(action={'type': 'review_manuscript', 'manuscript_id': m['id'], 'version': m['version'], 'confirmed': True})
    await i.handle(action={'type': 'record_acceptance', 'confirmed': True, 'reference': 'DEMO-ACCEPTANCE-1'})
    return m


async def intent(i):
    await ready(i)
    await i.handle(action={'type': 'prepare_registration', 'demo_confirmed': True, 'invoice_url': f'solana:{RECIPIENT}?amount=0.01'})
    return i.publication_state()['payments'][0]


def transaction(p):
    return {'slot': 123, 'blockTime': int(time.time()), 'meta': {'err': None}, 'transaction': {'message': {
        'accountKeys': [{'pubkey': p['reference'], 'signer': False}, {'pubkey': PAYER, 'signer': True}, {'pubkey': RECIPIENT, 'signer': False}],
        'instructions': [{'programId': solana.SYSTEM_PROGRAM, 'parsed': {'type': 'transfer', 'info': {'source': PAYER, 'destination': RECIPIENT, 'lamports': p['lamports']}}}]}}}


def rpc_mock(monkeypatch, tx):
    monkeypatch.setattr(solana, 'rpc', lambda method, params: solana.GENESIS['devnet'] if method == 'getGenesisHash' else copy.deepcopy(tx))


def test_invoice_uses_exact_decimal_lamports_and_rejects_bad_input():
    assert solana.parse_invoice(f'solana:{RECIPIENT}?amount=0.000000001')['lamports'] == 1
    for amount in ('0', '-1', 'NaN', 'Infinity', '0.0000000001', '999999999999999'):
        with pytest.raises(ValueError):
            solana.parse_invoice(f'solana:{RECIPIENT}?amount={amount}')
    for url in (f'solana:{RECIPIENT}?amount=1&amount=2', f'solana:{RECIPIENT}?amount=1&spl-token=anything', 'solana:https://evil.example.org'):
        with pytest.raises(ValueError):
            solana.parse_invoice(url)


def test_manuscript_only_contains_human_accepted_results_and_needs_review():
    async def scenario():
        i = investigation()
        i.research['runs'][0]['review'] = 'unreviewed'
        with pytest.raises(ValueError, match='Accept'):
            await i.handle(action={'type': 'create_manuscript', 'title': 'Study', 'authors': 'Author'})
        i.research['runs'][0]['review'] = 'accepted'
        await i.handle(action={'type': 'create_manuscript', 'title': 'Study', 'authors': 'Author'})
        m = i.active_manuscript()
        assert '| brier | 0.1 | 0.2 |' in m['content']
        with pytest.raises(ValueError, match='Complete'):
            await i.handle(action={'type': 'review_manuscript', 'manuscript_id': m['id'], 'version': m['version'], 'confirmed': True})
        assert m['status'] == 'draft'
    asyncio.run(scenario())


def test_registration_requires_manuscript_review_and_acceptance_and_prevents_duplicates(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    async def scenario():
        i = investigation()
        action = {'type': 'prepare_registration', 'demo_confirmed': True, 'invoice_url': f'solana:{RECIPIENT}?amount=0.01'}
        with pytest.raises(ValueError, match='manuscript'):
            await i.handle(action=action)
        await ready(i)
        await i.handle(action=action)
        p = i.publication_state()['payments'][0]
        assert p['network'] == 'devnet' and p['status'] == 'awaiting_wallet'
        assert p['signature'] is None and p['fee_kind'] == 'registration'
        assert p['manuscript_hash'] == i.active_manuscript()['sha256']
        with pytest.raises(ValueError, match='already exists'):
            await i.handle(action=action)
        await i.handle(action={'type': 'cancel_payment', 'payment_id': p['id']})
        await i.handle(action=action)
        assert len(i.publication_state()['payments']) == 2
    asyncio.run(scenario())


def test_editing_manuscript_invalidates_wallet_request_and_acceptance(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    async def scenario():
        i = investigation()
        p = await intent(i)
        m = i.active_manuscript()
        await i.handle(action={'type': 'save_manuscript', 'manuscript_id': m['id'], 'version': m['version'], 'content': m['content'] + '\nA revision.'})
        with pytest.raises(ValueError, match='review'):
            i.usable_intent(p['id'])
        assert m['status'] == 'draft'
    asyncio.run(scenario())


def test_rejected_result_invalidates_manuscript_readiness():
    async def scenario():
        i = investigation()
        await ready(i)
        i.research['runs'][0]['review'] = 'rejected'
        with pytest.raises(ValueError, match='no longer accepted'):
            i.ready_manuscript()
    asyncio.run(scenario())


def test_mainnet_requires_matching_organizer_invoice(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'mainnet-beta')
    async def scenario():
        i = investigation()
        await ready(i)
        with pytest.raises(ValueError, match='official organizer'):
            await i.handle(action={'type': 'prepare_registration', 'invoice_url': f'solana:{RECIPIENT}?amount=1', 'demo_confirmed': True})
    asyncio.run(scenario())
    monkeypatch.setattr(solana, 'page', lambda _: {'url': 'https://organizer.example.org/pay', 'fetched_at': 'now', 'payment_links': [f'solana:{RECIPIENT}?amount=0.5']})
    with pytest.raises(ValueError, match='exact SOL invoice'):
        solana.invoice_evidence(f'solana:{RECIPIENT}?amount=1', 'https://organizer.example.org/pay')


def test_finalized_receipt_checks_reference_recipient_amount_payer_and_failure(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    async def scenario():
        i = investigation()
        p = await intent(i)
        p['payer'] = PAYER
        tx = transaction(p)
        rpc_mock(monkeypatch, tx)
        assert solana.verify_receipt(p, SIGNATURE)['slot'] == 123
        for mutate in ('recipient', 'amount', 'reference', 'failed'):
            bad = copy.deepcopy(tx)
            if mutate == 'recipient': bad['transaction']['message']['instructions'][0]['parsed']['info']['destination'] = PAYER
            if mutate == 'amount': bad['transaction']['message']['instructions'][0]['parsed']['info']['lamports'] += 1
            if mutate == 'reference': bad['transaction']['message']['accountKeys'][0]['pubkey'] = RECIPIENT
            if mutate == 'failed': bad['meta']['err'] = 'failed'
            rpc_mock(monkeypatch, bad)
            with pytest.raises(ValueError): solana.verify_receipt(p, SIGNATURE)
    asyncio.run(scenario())


def test_rpc_network_mismatch_is_blocked(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    monkeypatch.setattr(solana, 'rpc', lambda *args: solana.GENESIS['mainnet-beta'])
    with pytest.raises(ValueError, match='does not match'):
        solana.latest_blockhash()


def test_wallet_claim_is_single_use_and_signature_is_not_payment_confirmation(monkeypatch):
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    monkeypatch.setattr(solana, 'latest_blockhash', lambda: {'value': {'blockhash': RECIPIENT, 'lastValidBlockHeight': 100}})
    async def scenario():
        from fastapi import HTTPException
        i = investigation()
        p = await intent(i)
        monkeypatch.setitem(INVESTIGATIONS, i.id, i)
        request = await wallet_request(i.id, p['id'], WalletBody(payer=PAYER))
        assert request['intent']['status'] == 'wallet_requested'
        with pytest.raises(HTTPException): await wallet_request(i.id, p['id'], WalletBody(payer=PAYER))
        await payment_submitted(i.id, p['id'], SignatureBody(signature=SIGNATURE))
        assert p['status'] == 'submitted' and 'receipt' not in p
        rpc_mock(monkeypatch, None)
        await i.handle(action={'type': 'verify_payment', 'payment_id': p['id'], 'signature': SIGNATURE})
        assert p['status'] == 'submitted'
        rpc_mock(monkeypatch, transaction(p))
        await i.handle(action={'type': 'verify_payment', 'payment_id': p['id'], 'signature': SIGNATURE})
        assert p['status'] == 'confirmed'
    asyncio.run(scenario())


def test_conference_claims_require_exact_source_passages(monkeypatch):
    source = {'url': 'https://conf.example.org', 'title': 'Test Conference', 'text': 'Submission deadline is November 1, 2026. Conference scope includes calibration.', 'payment_links': [], 'fetched_at': 'now'}
    monkeypatch.setattr(conferences, 'model_json', lambda *args, **kwargs: {'is_conference': True, 'name': 'Test Conference', 'fit': 'Calibration topics overlap.', 'branch_ids': ['b'], 'details': [
        {'kind': 'deadline', 'value': '2027-01-01', 'quote': 'Submission deadline is November 1, 2026.'},
        {'kind': 'fees', 'value': 'Free', 'quote': 'There are no registration fees.'}]})
    candidate = conferences.extract(source, [{'id': 'b', 'title': 'Calibration', 'hypothesis': 'Test'}])
    assert len(candidate['details']) == 1
    assert candidate['details'][0]['quote'] == 'Submission deadline is November 1, 2026.'
    assert candidate['organizer_confirmed'] is False
    assert candidate['payment_links'] == []


def test_conference_urls_and_dns_cannot_target_private_networks(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))])
    for url in ('http://example.org', 'https://localhost', 'https://user:pass@example.org', 'https://example.org:8080'):
        with pytest.raises(ValueError): conferences.public_url(url)
    with pytest.raises(ValueError): conferences.safe_socket(('example.org', 443))


def test_closed_deadlines_are_flagged_and_today_requires_timezone_review():
    from datetime import date
    details = lambda quote: [{'kind': 'deadline', 'quote': quote}]
    assert conferences.deadline_status(details('Paper deadline Sep 25, 2026 AOE'), date(2026, 10, 18)) == 'past_date_quoted'
    assert conferences.deadline_status(details('Deadline 2027-01-10'), date(2026, 10, 18)) == 'future_date_quoted'
    assert conferences.deadline_status(details('Deadline October 18, 2026'), date(2026, 10, 18)) == 'check_today_timezone'
    assert conferences.deadline_status(details('Deadline January 10'), date(2026, 10, 18)) == 'unknown'


def test_manuscript_and_payment_state_survive_restart_without_automatic_payment(monkeypatch, tmp_path):
    from backend.session import restore_saved
    monkeypatch.setenv('MARGINALIA_DB', str(tmp_path / 'publication.sqlite3'))
    monkeypatch.setenv('SOLANA_NETWORK', 'devnet')
    async def scenario():
        i = investigation()
        p = await intent(i)
        i.persistent = True
        i.research_event()
        restore_saved()
        restored = INVESTIGATIONS[i.id]
        try:
            assert restored.active_manuscript()['sha256'] == i.active_manuscript()['sha256']
            assert restored.publication_state()['payments'][0]['id'] == p['id']
            assert restored.publication_state()['payments'][0]['status'] == 'awaiting_wallet'
            assert restored.task is None and restored.publication_task is None
        finally:
            del INVESTIGATIONS[i.id]
    asyncio.run(scenario())


def test_search_drops_unrelated_results_instead_of_recommending_them():
    leads = conferences.conference_leads([{'title': 'Machine definition', 'url': 'https://dictionary.example.org/machine'},
                                          {'title': 'Conference call for papers', 'url': 'https://organizer.example.org/cfp'}])
    assert len(leads) == 1 and leads[0]['url'].endswith('/cfp')


def test_discovery_falls_back_to_live_official_source_leads_without_invented_opportunities(monkeypatch):
    monkeypatch.setattr(conferences, 'search', lambda _: [])
    visited = []
    def read(url):
        visited.append(url)
        return {'url': url, 'title': 'Official conference source', 'text': 'Call for papers: calibration topics. Deadline November 1, 2027.',
                'payment_links': [], 'fetched_at': 'now'}
    monkeypatch.setattr(conferences, 'page', read)
    monkeypatch.setattr(conferences, 'model_json', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('offline')))
    found = conferences.discover('Calibration', [{'id': 'idea', 'title': 'Calibration', 'hypothesis': 'Test calibration.'}])
    assert found['source_mode'] == 'official_source_sweep'
    assert visited == ['https://iclr.cc/', 'https://icml.cc/', 'https://neurips.cc/']
    assert len(found['candidates']) == 3
    assert all(not c['organizer_confirmed'] and not c['payment_links'] for c in found['candidates'])


def test_submission_deadline_ignores_later_review_dates():
    from datetime import date
    details = [{'kind': 'deadline', 'quote': 'Abstract deadline Sep 18, 2026 AOE Paper deadline Sep 25, 2026 AOE Reviews released Nov 05, 2026 Final decisions Dec 16, 2026'}]
    assert conferences.deadline_status(details, date(2026, 10, 18)) == 'past_date_quoted'
    mixed = [{'kind': 'deadline', 'quote': 'Important dates Sep 25, 2026 and Dec 16, 2026'}]
    assert conferences.deadline_status(mixed, date(2026, 10, 18)) == 'unknown'
