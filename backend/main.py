"""Marginalia: a desk for reading papers with an agent and writing in the margin."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from urllib.parse import urlparse

from .session import INVESTIGATIONS, create, restore_saved
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app):
    restore_saved()
    yield
    tasks = [task for i in INVESTIGATIONS.values() for task in (i.task, i.publication_task) if task and not task.done()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"

app = FastAPI(title="Marginalia", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def same_origin_mutations(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/"):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != request.headers.get("host"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Cross-origin changes are not allowed."}, status_code=403)
    return await call_next(request)


class StartBody(BaseModel):
    direction: str = Field(min_length=3, max_length=6000)
    query: str | None = Field(default=None, max_length=500)


class MarginBody(BaseModel):
    text: str | None = Field(default=None, max_length=1000)
    action: dict | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/api/investigations")
async def start(body: StartBody) -> dict:
    query = (body.query or "").strip() or None
    investigation = create(body.direction.strip(), query)
    return {"id": investigation.id}


@app.get("/api/investigations/{investigation_id}/events")
async def events(investigation_id: str, request: Request) -> StreamingResponse:
    investigation = INVESTIGATIONS.get(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="No investigation with that id.")

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()
        # Subscribe before copying so a note that lands during the replay
        # is queued rather than lost. Both lines are synchronous.
        investigation.subscribers.append(queue)
        try:
            last = int(request.headers.get("last-event-id", "0"))
        except ValueError:
            last = 0
        snapshot = [e for e in investigation.events if e["n"] > last]
        try:
            for event in snapshot:
                yield _encode(event)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield _encode(event)
        finally:
            if queue in investigation.subscribers:
                investigation.subscribers.remove(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/investigations/{investigation_id}/dataset.csv")
def dataset_csv(investigation_id: str) -> FileResponse:
    investigation = INVESTIGATIONS.get(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="No investigation with that id.")
    path = Path(investigation._table_path())
    if not path.is_file():
        raise HTTPException(status_code=404, detail="This investigation has not written a table.")
    return FileResponse(path, media_type="text/csv", filename=f"{investigation_id}-dataset.csv")


@app.post("/api/investigations/{investigation_id}/margin")
async def margin(investigation_id: str, body: MarginBody) -> dict:
    investigation = INVESTIGATIONS.get(investigation_id)
    if investigation is None:
        raise HTTPException(status_code=404, detail="No investigation with that id.")
    if not (body.text or "").strip() and not body.action:
        raise HTTPException(status_code=400, detail="Write a note or send an action.")
    try:
        await investigation.handle(text=(body.text or "").strip() or None, action=body.action)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True}


def _encode(event: dict) -> str:
    return f"id: {event['n']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/investigations")
def list_investigations() -> list[dict]:
    return [{"id": i.id, "direction": i.direction, "phase": i.phase,
             "runs": len(i.research.get("runs", []))} for i in reversed(list(INVESTIGATIONS.values()))]


@app.get("/api/investigations/{investigation_id}/export")
def export_investigation(investigation_id: str) -> dict:
    i = INVESTIGATIONS.get(investigation_id)
    if i is None:
        raise HTTPException(404, "Unknown investigation.")
    return {"id": i.id, "direction": i.direction, "phase": i.phase, "papers": [p.as_dict() for p in i.papers],
            "rules": sorted(i.rules), "metric": i.metric, "research": i.research, "events": i.events}


@app.get("/api/investigations/{investigation_id}/runs/{run_id}/artifacts/{filename:path}")
def run_artifact(investigation_id: str, run_id: str, filename: str):
    i = INVESTIGATIONS.get(investigation_id)
    run = next((r for r in i.research.get("runs", []) if r["id"] == run_id), None) if i else None
    if run is None or filename not in run.get("artifacts", []):
        raise HTTPException(404, "Unknown artifact.")
    folder = ROOT / "experiments" / investigation_id / run_id
    path = folder / filename
    if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink() or not path.is_file():
        raise HTTPException(404, "Unknown artifact.")
    return FileResponse(path, filename=path.name)


class WalletBody(BaseModel):
    payer: str = Field(min_length=32, max_length=44)


class SignatureBody(BaseModel):
    signature: str = Field(min_length=64, max_length=88)


def payment_investigation(investigation_id):
    i = INVESTIGATIONS.get(investigation_id)
    if i is None:
        raise HTTPException(404, 'Unknown investigation.')
    return i


@app.post('/api/investigations/{investigation_id}/payments/{payment_id}/wallet')
async def wallet_request(investigation_id: str, payment_id: str, body: WalletBody):
    from . import solana_payments as solana
    i = payment_investigation(investigation_id)
    try:
        payer = solana.address(body.payer)
        intent = i.usable_intent(payment_id)
        if intent['status'] != 'awaiting_wallet':
            raise ValueError('This request is already open in a wallet or submitted. Verify the receipt instead of paying again.')
        latest = await asyncio.to_thread(solana.latest_blockhash)
        intent = i.usable_intent(payment_id)
        if intent['status'] != 'awaiting_wallet':
            raise ValueError('Another wallet request already claimed this invoice.')
        intent.update(status='wallet_requested', payer=payer, wallet_requested_at=__import__('time').time())
        i.publication_event()
        return {'intent': intent, 'blockhash': latest['value']['blockhash'], 'last_valid_block_height': latest['value']['lastValidBlockHeight']}
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, 'Solana RPC is unavailable; no wallet request was opened.') from exc


@app.post('/api/investigations/{investigation_id}/payments/{payment_id}/submitted')
async def payment_submitted(investigation_id: str, payment_id: str, body: SignatureBody):
    from . import solana_payments as solana
    i = payment_investigation(investigation_id)
    try:
        intent = i.usable_intent(payment_id, require_ready=False)
        if len(solana.b58decode(body.signature)) != 64 or intent['status'] not in {'wallet_requested', 'submitted'}:
            raise ValueError('Invalid payment submission state or signature.')
        if intent.get('signature') and intent['signature'] != body.signature:
            raise ValueError('A different transaction is already recorded for this request.')
        intent.update(status='submitted', signature=body.signature)
        i.publication_event()
        return {'ok': True}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class RejectionBody(BaseModel):
    code: Literal[4001]


@app.post('/api/investigations/{investigation_id}/payments/{payment_id}/rejected')
async def payment_rejected(investigation_id: str, payment_id: str, body: RejectionBody):
    i = payment_investigation(investigation_id)
    try:
        intent = i.usable_intent(payment_id, require_ready=False)
        if intent['status'] != 'wallet_requested' or intent.get('signature'):
            raise ValueError('This payment is already submitted or not awaiting wallet review.')
        intent.update(status='awaiting_wallet', payer=None)
        i.publication_event()
        return {'ok': True}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/investigations/{investigation_id}/manuscripts/{manuscript_id}.md')
def download_manuscript(investigation_id: str, manuscript_id: str):
    from fastapi.responses import Response
    i = payment_investigation(investigation_id)
    manuscript = next((m for m in i.publication_state()['manuscripts'] if m['id'] == manuscript_id), None)
    if not manuscript:
        raise HTTPException(404, 'Unknown manuscript.')
    return Response(manuscript['content'], media_type='text/markdown',
                    headers={'Content-Disposition': f'attachment; filename="manuscript-{manuscript_id}.md"'})
