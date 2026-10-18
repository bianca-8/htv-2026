"""Small HTTP getter with a disk cache and a certificate bundle.

The system Python on this machine does not trust public CAs, so every
outbound call uses certifi. arXiv asks clients to wait between requests;
the cache keeps a demo from repeating those calls.
"""

from __future__ import annotations

import hashlib
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".cache" / "http"
USER_AGENT = "Marginalia/0.1 (human-agent research prototype; mailto:research@example.com)"

_lock = threading.Lock()
_last_arxiv = 0.0
_MIN_ARXIV_GAP = 3.1


class FetchError(RuntimeError):
    def __init__(self, message: str, *, rate_limited: bool = False):
        super().__init__(message)
        self.rate_limited = rate_limited


def _context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


def _key(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return CACHE_DIR / digest


def fetch(url: str, *, timeout: float = 25, cache: bool = True, polite_arxiv: bool = False) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _key(url)
    if cache and path.exists():
        return path.read_bytes()

    global _last_arxiv
    if polite_arxiv:
        with _lock:
            wait = _MIN_ARXIV_GAP - (time.time() - _last_arxiv)
            if wait > 0:
                time.sleep(wait)
            _last_arxiv = time.time()

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_context()) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise FetchError("The source rate-limited this request.", rate_limited=True) from exc
        raise FetchError(f"HTTP {exc.code} from {url}") from exc
    except Exception as exc:
        raise FetchError(f"Could not reach the source: {exc}") from exc

    if body.strip()[:20].lower().startswith(b"rate exceeded"):
        raise FetchError("arXiv returned 'Rate exceeded'.", rate_limited=True)

    if cache and _cacheable(body):
        path.write_bytes(body)
    return body


def _cacheable(body: bytes) -> bool:
    head = body.lstrip()[:80].lower()
    if head.startswith(b"rate exceeded") or b"bad request" in head[:200]:
        return False
    return len(body) > 80 and (
        head.startswith(b"<?xml")
        or head.startswith(b"{")
        or head.startswith(b"<!")
        or head.startswith(b"<html")
        or head.startswith(b"<feed")
    )
