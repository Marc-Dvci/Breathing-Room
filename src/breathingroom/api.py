"""HTTP API and steward console.

    uvicorn breathingroom.api:app --host 0.0.0.0 --port 8000

Environment:
    BR_DATA          folder with the Jülich data (default ../data next to the repository)
    BR_CAMERA_RUN    rated entrance run replayed by camera E1 (default 150_q_56_h0)
    BR_USE_MODEL     1 = agent on the Bedrock API endpoint, 0 = deterministic planner
    BR_TRACE         trace file (default ./var/trace.jsonl)
    BR_MODE          "production" (default) or "demo". In production every page and API except the
                     sign-in page needs a steward signed in, a decision needs the steward's PIN again,
                     the MCP endpoint needs BR_MCP_TOKEN, and the demonstration controls are off.
                     "demo" is the public demonstration: anyone may look, and decide while no steward
                     accounts exist, and the trace says so
    BR_STEWARDS_FILE steward accounts, JSON name -> PIN hash (python -m breathingroom.stewards)
    BR_STEWARD_PIN   a single account named "steward"
    BR_SESSION_SECRET key that signs sign-in cookies (default: random at start, so a restart signs everyone out)
    BR_SESSION_HOURS how long a sign-in lasts (default 12, one shift)
    BR_MCP_TOKEN     bearer token for /mcp in production; without it the endpoint is closed
    BR_TRACE_KEY     Ed25519 private key (PEM) that signs every trace entry
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import mcp, report
from .runtime import CATALOGUE, ROOT, CameraZone, Runtime
from .stewards import Stewards
from .trace import compare_cloudwatch, verify, verify_signatures

WEB = Path(__file__).parent / "web"
DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
RUN = os.environ.get("BR_CAMERA_RUN", "150_q_56_h0")
TRACE = Path(os.environ.get("BR_TRACE", ROOT / "var" / "trace.jsonl"))
MODE = os.environ.get("BR_MODE", "production")
STEWARDS = Stewards.from_env()
SESSION_SECRET = os.environ.get("BR_SESSION_SECRET", "").encode() or os.urandom(32)
SESSION_S = float(os.environ.get("BR_SESSION_HOURS", "12")) * 3600
SESSION_COOKIE = "br_session"
MCP_TOKEN = os.environ.get("BR_MCP_TOKEN", "")
# reachable without signing in, in production: the page that asks for the PIN, and the health check
OPEN_PATHS = {"/", "/health", "/api/mode", "/api/session"}

state: dict = {}
MCP_CALLS_PER_MIN = 120  # the endpoint is public and the instance is small
_mcp_times: deque = deque(maxlen=MCP_CALLS_PER_MIN)


def build_runtime() -> Runtime:
    video = DATA / "fzj" / "videos" / f"{RUN}.mp4"
    cam = None
    if video.exists():
        cam = CameraZone(video, ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json", RUN[:3])
    rt = Runtime(TRACE, use_model=os.environ.get("BR_USE_MODEL", "1") == "1", camera=cam,
                 seed=int(os.environ.get("BR_SEED", "7")))
    rt.trace.write("start", camera_run=RUN if cam else None, use_model=rt.use_model, mode=MODE,
                   steward_accounts=len(STEWARDS.accounts), signing_key=rt.trace.public_key())
    return rt


def start() -> None:
    rt = build_runtime()
    state["rt"] = rt
    th = threading.Thread(target=rt.loop, daemon=True)
    th.start()
    state["thread"] = th


@asynccontextmanager
async def lifespan(app: FastAPI):
    start()
    yield
    state["rt"].running = False


app = FastAPI(title="Breathing Room", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=WEB), name="static")


def session_cookie(name: str, now: float | None = None) -> str:
    body = f"{name}|{int((time.time() if now is None else now) + SESSION_S)}".encode()
    sig = hmac.new(SESSION_SECRET, body, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(body).decode() + "." + sig


def session_name(value: str | None, now: float | None = None) -> str | None:
    """The steward a sign-in cookie belongs to, if it is genuine and current."""
    try:
        b64, sig = (value or "").split(".")
        body = base64.urlsafe_b64decode(b64.encode())
        name, exp = body.decode().rsplit("|", 1)
    except (ValueError, UnicodeDecodeError):
        return None
    if not hmac.compare_digest(hmac.new(SESSION_SECRET, body, hashlib.sha256).hexdigest(), sig):
        return None
    return name if int(exp) > (time.time() if now is None else now) and name in STEWARDS.accounts else None


def may_read(path: str, cookie: str | None, authorization: str = "") -> bool:
    """Outside the public demonstration, camera frames, evidence crops, zone state, the record and
    the report are for signed-in stewards, and the MCP endpoint for holders of its token."""
    if MODE == "demo" or path in OPEN_PATHS or path.startswith("/static/"):
        return True
    if path == "/mcp":
        return bool(MCP_TOKEN) and hmac.compare_digest(authorization.encode(), f"Bearer {MCP_TOKEN}".encode())
    return session_name(cookie) is not None


@app.middleware("http")
async def headers(request: Request, call_next):
    if not may_read(request.url.path, request.cookies.get(SESSION_COOKIE), request.headers.get("authorization", "")):
        resp = JSONResponse({"detail": "sign in as a steward"}, 401)
    else:
        resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.post("/api/session")
async def sign_in(request: Request):
    """A steward signs in with their PIN; the cookie lasts one shift."""
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    if not STEWARDS:
        raise HTTPException(403, "no steward accounts are configured")
    name, _ = _steward(body, request)
    resp = JSONResponse({"ok": True, "steward": name})
    resp.set_cookie(SESSION_COOKIE, session_cookie(name), max_age=int(SESSION_S), httponly=True, secure=True,
                    samesite="strict")
    state["rt"].trace.write("session", steward=name, event="signed in")
    return resp


@app.delete("/api/session")
def sign_out():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/report")
def evening_report(all: int = 0):
    return HTMLResponse(report.render(state["rt"].trace.path, everything=bool(all), trace=state["rt"].trace), headers={"Cache-Control": "no-store"})


@app.get("/health")
def health():
    """503 when a zone has produced no data for a while: for a monitor outside the instance."""
    h = state["rt"].health()
    return JSONResponse(h, 200 if h["ok"] else 503, headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def api_state():
    rt: Runtime = state["rt"]
    rt.viewed()
    return JSONResponse(rt.snapshot(), headers={"Cache-Control": "no-store"})


@app.get("/api/camera.jpg")
def camera_jpg():
    rt: Runtime = state["rt"]
    if rt.camera is None or rt.camera.jpeg() is None:
        raise HTTPException(404, "no camera frame yet")
    return Response(rt.camera.jpeg(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/evidence/{eid}.jpg")
def evidence_jpg(eid: str):
    rt: Runtime = state["rt"]
    img = rt.camera.evidence_image(eid) if rt.camera is not None else None
    if img is None:
        raise HTTPException(404, "evidence image not kept")
    return Response(img, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


def _client(request: Request) -> str:
    # Caddy on the same host sets X-Forwarded-For; anything else is the direct peer
    peer = request.client.host if request.client else ""
    fwd = request.headers.get("x-forwarded-for", "") if peer in ("127.0.0.1", "::1") else ""
    return fwd.split(",")[0].strip() or peer


def _steward(body: dict, request: Request) -> tuple[str, str]:
    """(who decided, how they were identified). With steward accounts a decision needs a PIN
    and is recorded under that steward's name. Without accounts, only the public demo accepts
    decisions, and records them as unauthenticated."""
    if STEWARDS:
        client = _client(request)
        if STEWARDS.locked(client):
            raise HTTPException(429, "too many wrong PINs; try again in a few minutes")
        name = STEWARDS.identify(str(body.get("pin", "")), client)
        if name is None:
            raise HTTPException(403, "steward PIN required")
        return name, "pin"
    if MODE == "demo":
        return str(body.get("who", "steward"))[:40], "none (public demo)"
    raise HTTPException(403, "no steward accounts are configured: set BR_STEWARDS_FILE")


@app.post("/api/proposals/{pid}/{verb}")
async def decide(pid: str, verb: str, request: Request):
    if verb not in ("approve", "reject"):
        raise HTTPException(404)
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    who, auth = _steward(body, request)
    out = state["rt"].decide(pid, verb == "approve", who=who, auth=auth)
    if not out.get("ok"):
        raise HTTPException(409, out.get("error"))
    return out


@app.post("/api/demo/{action}")
def demo(action: str):
    if MODE != "demo":
        raise HTTPException(404)
    rt: Runtime = state["rt"]
    if action == "bump" and rt.camera is not None:
        rt.camera.bumped = True
        rt.trace.write("demo", action="camera knocked (simulated 3 degree roll)")
        return {"ok": True}
    if action == "unbump" and rt.camera is not None:
        rt.camera.bumped = False
        time.sleep(1.5)  # let the camera thread process a frame taken after the restore
        r = rt.camera.check_camera()  # trust comes back from a measurement, not from the button
        with rt.lock:
            if r["trusted"]:
                for rc in rt.recalibrations:
                    rc["resolved"] = True
        rt.trace.write("demo", action="camera restored to its calibrated position")
        return {"ok": True}
    if action == "restart":
        rt.restart_evening(force=True)
        return {"ok": True}
    if action in ("speed1", "speed4"):
        rt.speed = 1.0 if action == "speed1" else 4.0
        return {"ok": True, "speed": rt.speed}
    raise HTTPException(404)


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Model Context Protocol, Streamable HTTP transport with JSON responses (read-only tools)."""
    now = time.time()
    if len(_mcp_times) == MCP_CALLS_PER_MIN and now - _mcp_times[0] < 60:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": "rate limited"}}, 429,
                            headers={"Retry-After": "10"})
    _mcp_times.append(now)
    try:
        msg = await request.json()
    except ValueError:
        return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}, 400)
    rt: Runtime = state["rt"]
    rt.viewed()
    if isinstance(msg, list):
        out = [r for r in (mcp.handle(rt, CATALOGUE, m, rt.trace) for m in msg if isinstance(m, dict)) if r is not None]
        return JSONResponse(out) if out else Response(status_code=202)
    out = mcp.handle(rt, CATALOGUE, msg, rt.trace) if isinstance(msg, dict) else None
    return JSONResponse(out) if out is not None else Response(status_code=202)


@app.get("/mcp")
def mcp_get():
    return Response(status_code=405, headers={"Allow": "POST"})


@app.get("/api/trace")
def trace_tail(n: int = 120):
    return state["rt"].trace.tail(min(n, 1000))


@app.get("/api/trace/verify")
def trace_verify():
    tr = state["rt"].trace
    ok, n, msg = verify(tr.path)
    return {"ok": ok, "entries": n, "message": msg,
            "signatures": verify_signatures(tr.path, (tr.public_key() or {}).get("public_key")),
            "cloudwatch": compare_cloudwatch(tr)}


@app.get("/api/trace/key")
def trace_key():
    return state["rt"].trace.public_key() or {"signed": False}


@app.get("/api/mode")
def mode(request: Request):
    return {"mode": MODE, "pin_required": bool(STEWARDS) or MODE != "demo", "sign_in_required": MODE != "demo",
            "steward": session_name(request.cookies.get(SESSION_COOKIE))}
