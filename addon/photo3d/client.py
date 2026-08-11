"""Transport to the solve daemon. stdlib urllib only.

Blender's bundled Python gets no packages installed into it — not requests, not
anything. urllib is enough: one POST, one JSON response, and every heavy payload
travels as a filesystem path rather than inline.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://127.0.0.1:8765"


class SolverUnreachable(RuntimeError):
    """The daemon is not running, or not where we are looking."""


class SolverRefused(RuntimeError):
    """The daemon answered, and the answer was no. Carries its explanation."""


def endpoint() -> str:
    return os.environ.get("PHOTO3D_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")


def _request(route: str, payload: dict | None, timeout: float):
    url = endpoint() + route
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        try:
            parsed = json.loads(detail).get("detail", detail)
            if isinstance(parsed, list):        # pydantic validation errors
                parsed = "; ".join(e.get("msg", str(e)) for e in parsed)
        except (ValueError, AttributeError):
            parsed = detail
        raise SolverRefused(f"{exc.code}: {parsed}") from None
    except urllib.error.URLError as exc:
        raise SolverUnreachable(
            f"no solver at {url} ({exc.reason}). Start it with:\n"
            f"    uvicorn server.solver_server:app --port 8765") from None
    except TimeoutError:
        raise SolverUnreachable(
            f"solver at {url} did not answer within {timeout:.0f}s. The first "
            "solve loads ~1.9 GB of weights; raise the timeout or wait.") from None


def health(timeout: float = 2.0) -> dict:
    return _request("/health", None, timeout)


def solve(payload: dict, timeout: float = 600.0) -> dict:
    """Long timeout on purpose: the first solve of a session pays the weight
    load, which is ~15 s, and a refine pass roughly triples the rest."""
    return _request("/solve", payload, timeout)


def shadow_mask(payload: dict, timeout: float = 300.0) -> dict:
    return _request("/shadow_mask", payload, timeout)
