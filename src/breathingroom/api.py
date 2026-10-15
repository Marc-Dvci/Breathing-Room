"""HTTP API and steward console.

    uvicorn breathingroom.api:app --host 0.0.0.0 --port 8000

Environment:
    BR_DATA          folder with the Jülich data (default ../data next to the repository)
    BR_CAMERA_RUN    rated entrance run replayed by camera E1 (default 150_q_56_h0)
    BR_USE_MODEL     1 = agent on the Bedrock API endpoint, 0 = deterministic planner
    BR_TRACE         trace file (default ./var/trace.jsonl)
    BR_STEWARD_PIN   if set, approvals need this PIN
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .runtime import ROOT, CameraZone, Runtime
from .trace import verify

WEB = Path(__file__).parent / "web"
DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
RUN = os.environ.get("BR_CAMERA_RUN", "150_q_56_h0")
TRACE = Path(os.environ.get("BR_TRACE", ROOT / "var" / "trace.jsonl"))
PIN = os.environ.get("BR_STEWARD_PIN", "")

state: dict = {}


def build_runtime() -> Runtime:
    video = DATA / "fzj" / "videos" / f"{RUN}.mp4"
    cam = None
    if video.exists():
        cam = CameraZone(video, ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json", RUN[:3])
    rt = Runtime(TRACE, use_model=os.environ.get("BR_USE_MODEL", "1") == "1", camera=cam,
                 seed=int(os.environ.get("BR_SEED", "7")))
    rt.trace.write("start", camera_run=RUN if cam else None, use_model=rt.use_model)
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


@app.middleware("http")
async def headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["X-Frame-Options"] = "DENY"
    return resp


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health():
    rt: Runtime = state["rt"]
    return {"ok": True, "running": rt.running, "zones": list(rt.zones())}


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


def _pin_ok(body: dict) -> None:
    if PIN and body.get("pin") != PIN:
        raise HTTPException(403, "steward PIN required")


@app.post("/api/proposals/{pid}/{verb}")
async def decide(pid: str, verb: str, request: Request):
    if verb not in ("approve", "reject"):
        raise HTTPException(404)
    body = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    _pin_ok(body)
    out = state["rt"].decide(pid, verb == "approve", who=str(body.get("who", "steward"))[:40])
    if not out.get("ok"):
        raise HTTPException(409, out.get("error"))
    return out


@app.post("/api/demo/{action}")
def demo(action: str):
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


@app.get("/api/trace")
def trace_tail(n: int = 120):
    return state["rt"].trace.tail(min(n, 1000))


@app.get("/api/trace/verify")
def trace_verify():
    ok, n, msg = verify(state["rt"].trace.path)
    return {"ok": ok, "entries": n, "message": msg}
