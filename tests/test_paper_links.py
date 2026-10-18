import asyncio
import pytest
from backend import sources, session, solana_payments
from backend.session import Investigation
from tests.test_research import paper


@pytest.mark.parametrize('link,expected', [
    ('https://arxiv.org/abs/1706.03762v7', '1706.03762'),
    ('https://arxiv.org/pdf/1706.03762.pdf', '1706.03762'),
    ('https://arxiv.org/html/2401.12345v2', '2401.12345'),
    ('https://arxiv.org/abs/hep-th/9901001v3', 'hep-th/9901001'),
])
def test_link_normalization(link, expected):
    assert sources.arxiv_id_from_link(link) == expected


@pytest.mark.parametrize('link', ['https://arxiv.org.evil.com/abs/1706.03762',
    'https://evil.com/abs/1706.03762', 'http://localhost/abs/1706.03762',
    'https://user@arxiv.org/abs/1706.03762', 'https://arxiv.org/search/anything'])
def test_reject_other_destinations(link):
    with pytest.raises(ValueError):
        sources.arxiv_id_from_link(link)


def test_fetch_exact_record(monkeypatch):
    urls = []
    monkeypatch.setattr(sources, 'fetch', lambda url, **kwargs: urls.append(url) or b'feed')
    monkeypatch.setattr(sources, 'parse_arxiv_feed', lambda _: [paper('1706.03762')])
    assert sources.import_paper_link('https://arxiv.org/pdf/1706.03762v2.pdf').paper_id == '1706.03762'
    assert urls == ['https://export.arxiv.org/api/query?id_list=1706.03762']
    monkeypatch.setattr(sources, 'parse_arxiv_feed', lambda _: [paper('different')])
    with pytest.raises(ValueError, match='complete record'):
        sources.import_paper_link('https://arxiv.org/abs/1706.03762')


def test_import_deduplicates_and_restores_kept(monkeypatch):
    monkeypatch.setattr(session, 'import_paper_link', lambda _: paper('p'))
    async def run():
        inv = Investigation('calibration', phase='papers')
        await inv.handle(action={'type': 'add_paper_link', 'url': 'url'})
        inv.dropped.add('p')
        await inv.handle(action={'type': 'add_paper_link', 'url': 'url'})
        assert len(inv.papers) == 1 and not inv.dropped
        assert inv.paper_imports == 0 and not inv.papers_ready
        inv.paper_imports = 1
        with pytest.raises(ValueError, match='import'):
            await inv.handle(action={'type': 'approve_papers'})
        inv.phase = 'branching'
        with pytest.raises(ValueError, match='literature'):
            await inv.handle(action={'type': 'add_paper_link', 'url': 'url'})
    asyncio.run(run())


def test_search_retains_manually_added_paper(monkeypatch):
    monkeypatch.setattr(session, 'search_papers', lambda *a, **k: ('query', [], [paper('found')], 'arxiv'))
    monkeypatch.setattr(session, 'enrich_with_html', lambda *a: None)
    class StopReview(Exception): pass
    async def review(self):
        assert {p.paper_id for p in self.papers} == {'found', 'manual'}
        raise StopReview()
    monkeypatch.setattr(Investigation, '_wait_for_papers', review)
    async def run():
        inv = Investigation('calibration', papers=[paper('manual')])
        with pytest.raises(StopReview):
            await inv._run_once(inv.direction, None)
    asyncio.run(run())


def test_mainnet_is_default(monkeypatch):
    monkeypatch.delenv('SOLANA_NETWORK', raising=False)
    assert solana_payments.network() == 'mainnet-beta'
