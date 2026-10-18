"""Deterministic local UI fixture, separate from real investigations and model providers.

Run: .venv/bin/python tests/ui_smoke_server.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['SOLANA_NETWORK'] = 'devnet'
os.environ.pop('DATABASE_URL', None)
os.environ.pop('TIGER_DATABASE_URL', None)
os.environ['MARGINALIA_DB'] = str(Path(tempfile.mkdtemp(prefix='marginalia-ui-')) / 'research.db')

from backend import research, workflow
from backend.main import app
from backend.session import Investigation, INVESTIGATIONS
from backend.sources import Paper

p = Paper('fixture-paper', 'Calibration methods (UI fixture)', ['Fixture author'], '2025-01-01',
          'We study calibration using temperature scaling on held-out predictions.',
          'https://example.org/calibration', None, ['cs.LG'], None, 'arxiv')
values = dict(title='Compare calibration on explicit toy inputs', hypothesis='Calibration changes Brier score on these inputs.',
              method='Compare the Brier score of probabilities [0.1, 0.8, 0.7, 0.2] with [0.1, 0.7, 0.5, 0.2].',
              dataset='Demonstration inputs: four probabilities and labels [0, 1, 0, 0].',
              baseline='Probabilities [0.1, 0.8, 0.7, 0.2] on the same labels.', evaluation='Compare Brier score.',
              assumptions='Human-specified toy inputs.', limitations='This does not establish benchmark performance.',
              evidence_level='demonstration')
branch = research.idea(values, [research.citation(p)])
branch['data'] = None
branch['estimated_effort'] = 'Four toy inputs; local CPU.'
i = Investigation('UI verification: calibration research', id='ui-fixture', persistent=True)
i.papers = [p]
i.phase = 'review'
i.research['branches'] = [branch]
i.publication_state()['conferences'] = [{'id': 'fixture-conference', 'name': 'Research Conference — UI test fixture', 'url': 'https://example.org/conference', 'fit': 'Synthetic conference data for testing the publication interface, not an actual submission opportunity.', 'details': [{'kind': 'scope', 'quote': 'UI fixture: research on calibration and evaluation.'}], 'checked_at': 'Test fixture', 'organizer_confirmed': False, 'deadline_status': 'unknown', 'payment_links': []}]
INVESTIGATIONS[i.id] = i
i.emit({'type': 'direction', 'direction': i.direction})
i.emit({'type': 'paper', 'paper': p.as_dict()})
i.set_phase('review', 'Fixture plans ready for approval.')
i.research_event()

SCRIPT = '''import json
from sklearn.metrics import brier_score_loss
y = [0, 1, 0, 0]
baseline = brier_score_loss(y, [0.1, 0.8, 0.7, 0.2])
score = brier_score_loss(y, [0.1, 0.7, 0.5, 0.2])
print(json.dumps({'summary': 'Computed Brier scores on the approved toy inputs.', 'measurements': {'brier': float(score)}, 'baseline': {'brier': float(baseline)}, 'data': 'Four human-specified demonstration inputs.', 'limitations': 'Toy example only.'}))
'''
workflow.write_script = lambda *args: SCRIPT
research.model_json = lambda prompt: {'ideas': [{**values, 'title': 'Follow-up calibration comparison',
                                                'citations': [{'paper_id': p.paper_id, 'quote': p.abstract}]}]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8766)
