"""The live system: sources, zones, the tier policy, agent episodes, steward decisions and
the verification that closes the loop.

Two zones run side by side in the demonstration:

- "Entrance E1": a real camera recording (Jülich 2018 entrance experiment, overhead GoPro)
  replayed through the vision pipeline on every tick: lens undistortion, head detection,
  DIS optical flow, head-plane homography, Helbing fields, and heads counted across the
  gate line. Its drift watchdog runs ALIKED + LightGlue against the calibration frame, and its
  second counter is DM-Count on the OpenCV 5 DNN engine.
- "Zone C - Station Road": the venue twin (JuPedSim), where approved actions change the
  crowd and the verification step checks that the density follows the forecast.

Each zone runs on its own thread. Heavy work (decoding, flow, simulation) happens outside
the runtime lock; the zone's state is committed under it, so the API always reads a
consistent snapshot.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import twin as tw
from .agent import Episode, Proposal, VenueCatalogue, investigate
from .calib import FloorGrid
from .fields import FieldBuilder, KERNEL_R
from .flow import Flow

from .physics import Action, Line, forecast, level_of_service
from .policy import Thresholds, Tier, TierPolicy, ZoneReading
from .trace import Trace

ROOT = Path(__file__).resolve().parents[2]
VERIFY_AFTER_S = 90.0
TWIN_WARMUP_S = float(os.environ.get("BR_TWIN_WARMUP_S", "90"))  # skip the quiet start of the evening
TWIN_CYCLE_S = float(os.environ.get("BR_TWIN_CYCLE_S", "900"))  # the evening restarts after this, when idle
PROPOSAL_TTL_S = 240.0  # a proposal nobody decided on goes stale: the crowd has moved on
IDLE_PAUSE_S = float(os.environ.get("BR_IDLE_PAUSE_S", "0"))  # >0: pause when no console has polled for this long
DRIFT_EVERY_S = float(os.environ.get("BR_DRIFT_EVERY_S", "15"))  # watchdog period
# ALIKED + LightGlue need about 1.6 GB while they run and DM-Count about 0.7 GB. On a small
# instance each call runs in a fresh child process, so the memory goes back to the system.
ISOLATE_MODELS = os.environ.get("BR_ISOLATE_MODELS", "0") == "1"


def _drift_job(ref, live, lens_x, H_u2w, probe) -> dict:
    """Child-process drift check (BR_ISOLATE_MODELS=1)."""
    from .datasets.fzj_camera import Lens
    from .drift import DriftWatch

    lens = Lens(lens_x)

    def to_floor(px):
        return cv2.perspectiveTransform(lens.undistort(px).reshape(-1, 1, 2), H_u2w).reshape(-1, 2)

    return DriftWatch(ref, to_floor, probe).check(live).to_json()


def _count_job(und, mask) -> tuple[float, float]:
    """Child-process DM-Count (BR_ISOLATE_MODELS=1)."""
    from .density import DensityModel

    r = DensityModel("sha", "new", max_side=1280)(und, roi_mask=mask)
    return r.count, r.seconds


def _isolated(fn, *args):
    """Run `fn` in a fresh child process that exits afterwards."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result(timeout=180)

# DM-Count count / head count inside the E1 zone, over 809 frames of the six calibrated
# Jülich runs (eval/results/test_a_meta.json): 2.5th percentile 0.27, 97.5th 0.99. DM-Count
# undercounts on this overhead view; a ratio outside the band means one of the two counters
# is not seeing the crowd (occlusion, lighting, a moved camera).
E1_COUNT_RATIO_BAND = (0.25, 1.0)


def helbing_points(pos: np.ndarray, vel: np.ndarray, R: float = KERNEL_R) -> tuple[np.ndarray, np.ndarray]:
    """Per-person density and pressure from positions and velocities (exact, no image)."""
    if len(pos) < 2:
        return np.zeros(len(pos)), np.zeros(len(pos))
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    w = np.exp(-d2 / R**2) / (np.pi * R**2)
    rho = w.sum(1)
    Vm = (w @ vel) / w.sum(1, keepdims=True)
    var = (w * ((vel[None] - Vm[:, None]) ** 2).sum(-1)).sum(1) / w.sum(1)
    return rho, rho * var


@dataclass
class ZoneState:
    zone: str
    t: float = 0.0
    rho: float = 0.0
    pressure: float = 0.0
    los: str = "A"
    n: float = 0.0
    area_m2: float = 1.0
    outflow_pps: float = 0.0
    minutes_to_act: float | None = None
    trusted: bool = True
    trust_reason: str = ""
    tier: str = "NORMAL"
    reasons: list[str] = field(default_factory=list)
    source: str = ""
    hist_t: deque = field(default_factory=lambda: deque(maxlen=900))
    hist_n: deque = field(default_factory=lambda: deque(maxlen=900))
    hist_out: deque = field(default_factory=lambda: deque(maxlen=900))
    hist_rho: deque = field(default_factory=lambda: deque(maxlen=900))
    hist_p: deque = field(default_factory=lambda: deque(maxlen=900))

    def push(self):
        self.hist_t.append(self.t)
        self.hist_n.append(self.n)
        self.hist_out.append(self.outflow_pps)
        self.hist_rho.append(self.rho)
        self.hist_p.append(self.pressure)

    def clear_history(self):
        for d in (self.hist_t, self.hist_n, self.hist_out, self.hist_rho, self.hist_p):
            d.clear()

    def to_json(self, history: int = 600) -> dict:
        return {
            "zone": self.zone, "t": self.t, "rho": round(self.rho, 2), "pressure": round(self.pressure, 4),
            "los": self.los, "n": round(self.n, 1), "area_m2": self.area_m2, "outflow_pps": round(self.outflow_pps, 2),
            "minutes_to_act": None if self.minutes_to_act is None else round(self.minutes_to_act, 1),
            "trusted": self.trusted, "trust_reason": self.trust_reason, "tier": self.tier, "reasons": self.reasons,
            "source": self.source,
            "history": {"t": list(self.hist_t)[-history:], "rho": [round(x, 3) for x in list(self.hist_rho)[-history:]],
                        "pressure": [round(x, 4) for x in list(self.hist_p)[-history:]]},
        }


class TwinZone:
    """Zone C of the venue twin."""

    name = "Zone C - Station Road"
    exit_width_m = tw.MAIN_EXIT_WIDTH

    def __init__(self, seed: int = 0, state_lock=None, warmup_s: float = 0.0):
        self.seed = seed
        self.venue = tw.Venue(seed=seed)
        self.state = ZoneState(self.name, area_m2=tw.ZONE_C_AREA, source="venue twin (JuPedSim)")
        self.last_obs: tw.Observation | None = None
        self.p_window: deque = deque(maxlen=3)
        self.sim_lock = threading.Lock()
        self.state_lock = state_lock or threading.RLock()
        self.timing = deque(maxlen=50)
        for _ in range(int(warmup_s)):
            self.tick(1.0)

    def tick(self, seconds: float = 1.0) -> None:
        t0 = time.perf_counter()
        with self.sim_lock:
            self.venue.step(seconds)
            o = self.venue.observe()
        x0, x1, y0, y1 = tw.ZONE_C
        inz = (o.positions[:, 0] >= x0) & (o.positions[:, 0] < x1) & (o.positions[:, 1] >= y0) & (o.positions[:, 1] < y1) \
            if len(o.positions) else np.zeros(0, bool)
        p90 = None
        if inz.sum() >= 3:
            _, p = helbing_points(o.positions[inz], o.velocities[inz])
            p90 = float(np.percentile(p, 90))
        with self.state_lock:
            self.last_obs = o
            if p90 is not None:
                self.p_window.append(p90)
            s = self.state
            s.t = o.t
            s.n = float(o.n_zone)
            s.rho = o.n_zone / tw.ZONE_C_AREA
            s.pressure = float(np.mean(self.p_window)) if self.p_window else 0.0
            s.outflow_pps = o.outflow_pps
            s.los = level_of_service(s.rho)
            s.push()
        self.timing.append(time.perf_counter() - t0)

    def history(self) -> dict:
        """The inputs of a forecast, copied: an episode compares every action on the same data."""
        with self.state_lock:
            s = self.state
            return {"t": np.array(s.hist_t), "n": np.array(s.hist_n), "out": np.array(s.hist_out),
                    "width": tw.MAIN_EXIT_WIDTH + (tw.SIDE_WIDTH if self.venue.side_open else 0.0)}

    def forecast(self, action: Action | None = None, threshold: float = 4.5, hist: dict | None = None):
        h = hist or self.history()
        return forecast(self.state.zone, h["t"], h["n"], h["out"], self.state.area_m2, h["width"], threshold=threshold,
                        horizon_s=600, action=action, window_s=45, min_points=40)

    def apply(self, action: dict) -> str:
        k = action["kind"]
        with self.sim_lock:
            if k == "meter_entry":
                gates = [action["target"]] if action["target"] in tw.GATES else list(tw.GATES)
                for g in gates:
                    self.venue.meter(g, action["rate_per_min"] / 60.0 / len(gates))
                return f"metering {', '.join(gates)} at {action['rate_per_min']} people/min"
            if k == "hold_entry":
                gates = [action["target"]] if action["target"] in tw.GATES else list(tw.GATES)
                for g in gates:
                    self.venue.meter(g, 0.0)
                return f"holding {', '.join(gates)}"
            if k == "open_exit":
                self.venue.open_side()
                return "second exit door open"
        return "no change"

    def floorplan(self) -> dict:
        o = self.last_obs
        pts = [] if o is None else np.round(o.positions, 2).tolist()
        rho = []
        if o is not None and len(o.positions) >= 2:
            r, _ = helbing_points(o.positions, o.velocities)
            rho = np.round(r, 2).tolist()
        return {"outline": tw.polygon_outline(), "agents": pts, "agent_rho": rho,
                "backlog": o.backlog if o else {}, "side_open": self.venue.side_open,
                "meters": {g: (None if v is None else round(v * 60)) for g, v in self.venue.gate_limit_pps.items()},
                "arrivals_per_min": round((o.arrivals_pps if o else 0) * 60)}


class ReplayReader:
    """Reads a recording forwards and loops. Frames are decoded in order with grab/read:
    seeking a threaded FFmpeg decoder on every frame costs ~75 ms per frame and can abort
    the decoder, while sequential decoding costs ~2 to 3 ms."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._open()
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 25.0

    def _open(self):
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {self.path}")
        self.pos = 0

    def read(self, idx: int) -> np.ndarray:
        if idx < self.pos:
            self.cap.release()
            self._open()
        while self.pos < idx:
            self.cap.grab()
            self.pos += 1
        ok, img = self.cap.read()
        self.pos += 1
        if not ok:
            raise RuntimeError(f"cannot read frame {idx} of {self.path.name}")
        return img


class CameraZone:
    """Entrance E1: recorded camera footage replayed through the vision pipeline."""

    name = "Entrance E1"
    exit_width_m = 0.5  # the Jülich entrance gate

    def __init__(self, video: Path, calib_json: Path, run: str, camera_name: str = "cam-E1", state_lock=None):
        from .datasets import fzj_camera as fc  # lens + homography recovered for the Jülich camera

        self.fc = fc
        lens, runs = fc.load(calib_json)
        self.lens = lens
        self.run = run
        self.H_w2u = np.array(runs[run]["H_world_to_undist"])
        self.H_u2w = np.linalg.inv(self.H_w2u)
        self.reader = ReplayReader(video)
        self.W, self.H, self.n_frames = self.reader.width, self.reader.height, self.reader.n_frames
        self.fps = self.reader.fps
        self.m1, self.m2 = cv2.initUndistortRectifyMap(lens.K, lens.D, None, lens.Kn, (self.W, self.H), cv2.CV_16SC2)
        self.flow = Flow("medium", 960)
        self.camera_name = camera_name
        # floor grid over the area in front of the gate (run frame of reference, head plane)
        self.grid = FloorGrid(-3.5, -2.5, 3.5, 7.0, cell=0.1)
        self.builder = FieldBuilder(self.grid, self.H_u2w, (self.W, self.H))
        self.zone_rect = (-2.8, 2.8, -1.8, 2.2)  # 5.6 m x 4 m in front of the entrance
        self.area = (self.zone_rect[1] - self.zone_rect[0]) * (self.zone_rect[3] - self.zone_rect[2])
        # people leave through the gate at y = -1.8 m; heads are counted across a line 0.4 m before it
        self.gate_line = Line("entrance gate", (-0.9, -1.4), (1.3, -1.4))
        self.prev_heads_m: np.ndarray | None = None
        self.state = ZoneState(self.name, area_m2=self.area, source=f"recorded camera (Jülich run {run})")
        self.state_lock = state_lock or threading.RLock()
        self.frame_idx = 0
        self.window: deque = deque(maxlen=10)  # 2 s at 5 Hz
        self.flux_window: deque = deque(maxlen=25)  # crossings per 0.2 s step, over 5 s
        self.last_jpeg: bytes | None = None
        self.bumped = False
        self.drift = None
        self.counter = None
        self.ref_frame = self.reader.read(0)
        self.last_raw = self.ref_frame
        self.last_und = None
        self.last_heads = np.zeros((0, 2))
        self.frame_lock = threading.Lock()
        self.drift_lock = threading.Lock()
        self.timing = deque(maxlen=50)
        self.loops = 0
        self.last_drift: dict | None = None

    def _knock(self, img: np.ndarray) -> np.ndarray:
        """A simulated knock to the camera: 3 degrees of roll and a shift."""
        M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 3.0, 1.0)
        M[:, 2] += (40, -25)
        return cv2.warpAffine(img, M, (self.W, self.H), borderMode=cv2.BORDER_REFLECT)

    def _read(self, idx: int) -> np.ndarray:
        img = self.reader.read(idx)
        return self._knock(img) if self.bumped else img

    def tick(self, seconds: float = 1.0) -> None:
        """Process `seconds` of footage at 5 Hz field samples (flow over 2-frame gaps)."""
        t0 = time.perf_counter()
        steps = int(seconds * 5)
        restarted = False
        for _ in range(steps):
            f = self.frame_idx
            if f + 2 >= self.n_frames - 1:  # end of the recording: start it again
                f = 0
                restarted = True
                self.window.clear()
                self.flux_window.clear()
                self.prev_heads_m = None
            a = self._read(f)
            b = self._read(f + 2)
            ua = cv2.remap(a, self.m1, self.m2, cv2.INTER_LINEAR)
            ub = cv2.remap(b, self.m1, self.m2, cv2.INTER_LINEAR)
            heads = self.lens.undistort(self.fc.detect_caps(a))
            fl = self.flow(self.flow.prepare(ua), self.flow.prepare(ub))
            fr = self.builder.build_from_points(f / self.fps, heads, fl, 2 / self.fps)
            self.window.append(fr)
            self.flux_window.append(self._crossings(heads))
            self.frame_idx = f + 5
            last_raw, last_und, last_heads = a, ua, heads
        rho = np.mean([w.rho for w in self.window], axis=0)
        P = np.mean([w.pressure for w in self.window], axis=0)
        x0, x1, y0, y1 = self.zone_rect
        mass = self.window[-1].mass
        cx = self.grid.centres().reshape(*self.grid.shape, 2)
        inz = (cx[..., 0] >= x0) & (cx[..., 0] < x1) & (cx[..., 1] >= y0) & (cx[..., 1] < y1)
        occupied = inz & (rho > 1.0)
        with self.frame_lock:
            self.last_raw, self.last_und, self.last_heads = last_raw, last_und, last_heads
        with self.state_lock:
            s = self.state
            if restarted:
                self.loops += 1
                s.clear_history()
            s.t += seconds
            s.n = float(mass[inz].sum())
            s.rho = s.n / self.area
            s.pressure = float(np.percentile(P[occupied], 90)) if occupied.sum() > 5 else 0.0
            s.outflow_pps = float(np.sum(self.flux_window)) / (len(self.flux_window) * 0.2) if self.flux_window else 0.0
            s.los = level_of_service(s.rho)
            s.push()
        self.timing.append(time.perf_counter() - t0)
        self._render(last_und, last_heads, P)

    def _crossings(self, heads_und: np.ndarray, max_step_m: float = 0.5) -> int:
        """Heads that crossed the gate line since the previous sample (0.2 s earlier): each head
        is matched to its nearest predecessor (Hungarian assignment on floor distance)."""
        cur = cv2.perspectiveTransform(np.asarray(heads_und, float).reshape(-1, 1, 2), self.H_u2w).reshape(-1, 2) \
            if len(heads_und) else np.zeros((0, 2))
        prev, self.prev_heads_m = self.prev_heads_m, cur
        if prev is None or not len(prev) or not len(cur):
            return 0
        d = np.linalg.norm(prev[:, None] - cur[None], axis=2)
        r, c = linear_sum_assignment(d)
        ok = d[r, c] <= max_step_m
        (xa, ya), (xb, _) = self.gate_line.a, self.gate_line.b
        p, q = prev[r[ok]], cur[c[ok]]
        crossed = (p[:, 1] >= ya) & (q[:, 1] < ya) & (q[:, 0] >= xa) & (q[:, 0] <= xb)
        return int(crossed.sum())

    def _render(self, und, heads, P):
        """Annotated frame: pushing-index heat on the head plane, heads, zone outline, gate line."""
        img = cv2.resize(und, (960, 720), interpolation=cv2.INTER_AREA)
        sx = 960 / self.W
        Pn = np.clip(P / 0.25, 0, 1)  # 0.25 /s^2 is the 95th percentile of pushing rows in test A
        heat = (Pn * 255).astype(np.uint8)
        G = self.H_w2u.copy()
        A = np.array([[1 / self.grid.cell, 0, -self.grid.x0 / self.grid.cell - 0.5],
                      [0, 1 / self.grid.cell, -self.grid.y0 / self.grid.cell - 0.5], [0, 0, 1]])
        M = np.diag([sx, sx, 1]) @ G @ np.linalg.inv(A)
        warped = cv2.warpPerspective(heat, M, (960, 720), flags=cv2.INTER_LINEAR)
        col = cv2.applyColorMap(warped, cv2.COLORMAP_INFERNO)
        alpha = (warped.astype(np.float32) / 255.0 * 0.6)[..., None]
        img = (img * (1 - alpha) + col * alpha).astype(np.uint8)
        for p in heads:
            cv2.circle(img, (int(p[0] * sx), int(p[1] * sx)), 5, (255, 255, 255), 1, cv2.LINE_AA)
        x0, x1, y0, y1 = self.zone_rect
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
        pp = cv2.perspectiveTransform(poly.reshape(-1, 1, 2), G).reshape(-1, 2) * sx
        cv2.polylines(img, [pp.astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
        gl = cv2.perspectiveTransform(np.array([self.gate_line.a, self.gate_line.b], float).reshape(-1, 1, 2), G).reshape(-1, 2) * sx
        cv2.line(img, tuple(gl[0].astype(int)), tuple(gl[1].astype(int)), (120, 255, 120), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.frame_lock:
                self.last_jpeg = buf.tobytes()

    def history(self) -> dict:
        with self.state_lock:
            s = self.state
            t, n, out = np.array(s.hist_t), np.array(s.hist_n), np.array(s.hist_out)
            if len(t) < 3:
                t, n, out = np.array([s.t - 1, s.t]), np.array([s.n, s.n]), np.array([s.outflow_pps] * 2)
        return {"t": t, "n": n, "out": out, "width": self.exit_width_m}

    def forecast(self, action: Action | None = None, threshold: float = 4.5, hist: dict | None = None):
        h = hist or self.history()
        return forecast(self.state.zone, h["t"], h["n"], h["out"], self.state.area_m2, h["width"], threshold=threshold,
                        horizon_s=600, action=action, window_s=20, min_points=18)

    def to_floor(self, px: np.ndarray) -> np.ndarray:
        """Original image pixels to head-plane metres: lens undistortion, then the homography."""
        u = self.lens.undistort(px)
        return cv2.perspectiveTransform(u.reshape(-1, 1, 2), self.H_u2w).reshape(-1, 2)

    def check_camera(self) -> dict:
        from .drift import DriftWatch

        corners = np.array([[x, y] for x in self.zone_rect[:2] for y in self.zone_rect[2:]], float)
        probe = self.lens.distort(cv2.perspectiveTransform(corners.reshape(-1, 1, 2), self.H_w2u).reshape(-1, 2))
        with self.frame_lock:
            live = self.last_raw
        with self.drift_lock:
            if ISOLATE_MODELS:
                r = _isolated(_drift_job, self.ref_frame, live, self.lens.x, self.H_u2w, probe)
            else:
                if self.drift is None:
                    self.drift = DriftWatch(self.ref_frame, self.to_floor, probe)
                r = self.drift.check(live).to_json()
        with self.state_lock:
            self.state.trusted, self.state.trust_reason = r["trusted"], r["reason"]
            self.last_drift = {**r, "t": self.state.t}
        return r

    def cross_check_count(self) -> dict:
        """A second, independent count: DM-Count (OpenCV 5 DNN) over the zone, against the
        head detector on the same frame."""
        from .density import DensityModel

        with self.frame_lock:
            und, heads = self.last_und, self.last_heads
        if und is None:
            return {"error": "no frame processed yet"}
        x0, x1, y0, y1 = self.zone_rect
        poly = cv2.perspectiveTransform(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float).reshape(-1, 1, 2),
                                        self.H_w2u).reshape(-1, 2)
        mask = np.zeros(und.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        if ISOLATE_MODELS:
            count, secs = _isolated(_count_job, und, mask)
        else:
            if self.counter is None:
                self.counter = DensityModel("sha", "new", max_side=1280)
            r = self.counter(und, roi_mask=mask)
            count, secs = r.count, r.seconds
        w = cv2.perspectiveTransform(np.asarray(heads, float).reshape(-1, 1, 2), self.H_u2w).reshape(-1, 2) \
            if len(heads) else np.zeros((0, 2))
        n_heads = int(((w[:, 0] >= x0) & (w[:, 0] < x1) & (w[:, 1] >= y0) & (w[:, 1] < y1)).sum())
        ratio = count / max(n_heads, 1)
        lo, hi = E1_COUNT_RATIO_BAND
        agree = n_heads >= 3 and lo <= ratio <= hi
        return {"count_primary": n_heads, "count_second": round(count, 1), "ratio": round(ratio, 2),
                "agreement_band": [lo, hi], "agree": bool(agree), "second_counter_ms": round(secs * 1000),
                "note": "head detector vs DM-Count (OpenCV 5 DNN); band measured on six Jülich runs"}

    def jpeg(self) -> bytes | None:
        with self.frame_lock:
            return self.last_jpeg


def forecast_skill(path: Path = ROOT / "eval" / "results" / "test_b_twin.json") -> dict[str, float]:
    """Mean amount (people/m2) by which the twin's peak exceeded the forecast peak, per action,
    over the test B seeds (eval/test_b_twin.py). Proposals rank actions on peak + this error."""
    try:
        peaks = json.loads(path.read_text())["summary"]["peak_by_action"]
    except (OSError, KeyError, ValueError):
        return {}
    return {k: round(max(0.0, v["simulated_mean"] - v["predicted_mean"]), 2) for k, v in peaks.items()}


SKILL = forecast_skill()
_TWIN_SKILL_KEYS = {("meter_entry", "both gates"): "meter both gates to 150/min", ("meter_entry", "gate A"): "meter gate A to 60/min",
                    ("meter_entry", "gate B"): "meter gate A to 60/min", ("open_exit", "second exit door"): "open second exit door",
                    ("hold_entry", "both gates"): "hold both gates", ("none", ""): "no action"}


def _with_skill(zone_actions: list[dict]) -> list[dict]:
    for a in zone_actions:
        key = _TWIN_SKILL_KEYS.get((a["kind"], a.get("target", "")))
        if key in SKILL:
            a["forecast_error"] = SKILL[key]
    return zone_actions


CATALOGUE = VenueCatalogue({
    TwinZone.name: _with_skill([
        {"kind": "meter_entry", "target": "gate A", "rate_per_min": 60, "share": 0.5, "camera": "cam-C1"},
        {"kind": "meter_entry", "target": "gate B", "rate_per_min": 60, "share": 0.5},
        {"kind": "meter_entry", "target": "both gates", "rate_per_min": 150},
        {"kind": "open_exit", "target": "second exit door", "width_m": tw.SIDE_WIDTH},
        {"kind": "hold_entry", "target": "both gates"},
    ]),
    CameraZone.name: [
        {"kind": "send_steward", "target": "entrance gate", "camera": "cam-E1"},
        {"kind": "announce", "target": "please stop pushing, the gate is open"},
        {"kind": "meter_entry", "target": "queue feed", "rate_per_min": 30},
        {"kind": "hold_entry", "target": "queue feed"},
    ],
})


class Runtime:
    def __init__(self, trace_path: Path, use_model: bool = True, camera: CameraZone | None = None, seed: int = 0,
                 twin_warmup_s: float = TWIN_WARMUP_S):
        self.trace = Trace(trace_path)
        self.use_model = use_model
        self.policy = TierPolicy(Thresholds())
        self.lock = threading.RLock()
        self.seed = seed
        self.twin_warmup_s = twin_warmup_s
        self.twin = TwinZone(seed=seed, state_lock=self.lock, warmup_s=twin_warmup_s)
        self.camera = camera
        if camera is not None:
            camera.state_lock = self.lock
        self.proposals: dict[str, Proposal] = {}
        self.episodes: dict[str, Episode] = {}
        self.active_episode: dict[str, str] = {}
        self.running = False
        self.speed = 1.0
        self.applied: list[dict] = []
        self.recalibrations: list[dict] = []
        self.evenings = 1
        self.last_view = time.time()
        self.paused_since: float | None = None

    # -- zones -----------------------------------------------------------------------
    def zones(self) -> dict:
        z = {self.twin.name: self.twin}
        if self.camera is not None:
            z[self.camera.name] = self.camera
        return z

    def tools_for(self, zone_name: str) -> dict:
        frozen = {zone_name: self.zones()[zone_name].history()}  # every forecast in the episode uses this

        def get_zone_state(zone: str) -> dict:
            z = self.zones()[zone]
            fc = z.forecast()
            with self.lock:
                s = z.state
                return {"zone": zone, "source": s.source, "density": round(s.rho, 2), "level_of_service": s.los,
                        "people": round(s.n), "area_m2": s.area_m2, "pushing_index": round(s.pressure, 4),
                        "tier": s.tier, "trend_per_min": round(fc.net_rate_pps * 60, 1),
                        "outflow_per_min": round(s.outflow_pps * 60, 1),
                        "minutes_to_4_5": None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1)}

        def check_camera(camera: str) -> dict:
            z = self.zones()[zone_name]
            if isinstance(z, CameraZone):
                return {"camera": camera, **z.check_camera()}
            return {"camera": camera, "trusted": True, "floor_shift_m": 0.0,
                    "note": "simulated zone: positions come from the twin, no camera to drift"}

        def cross_check_count(zone: str) -> dict:
            z = self.zones()[zone]
            if isinstance(z, CameraZone):
                return z.cross_check_count()
            with self.lock:
                n = self.twin.state.n
            return {"count_primary": round(n), "count_second": round(n), "agree": True,
                    "note": "simulated zone: the count is exact"}

        def measure_exit_flow(zone: str) -> dict:
            z = self.zones()[zone]
            with self.lock:
                out = list(z.state.hist_out)[-30:]
            width = z.exit_width_m + (tw.SIDE_WIDTH if isinstance(z, TwinZone) and self.twin.venue.side_open else 0.0)
            how = "heads counted across the gate line" if isinstance(z, CameraZone) else "people through the exits"
            return {"outflow_per_min": round(float(np.mean(out)) * 60, 1) if out else 0.0,
                    "exit_width_m": width, "exit_capacity_per_min": round(width * 1.3 * 60, 1), "measured_as": how}

        def _forecast(zone: str, action: Action):
            z = self.zones()[zone]
            if zone not in frozen:
                frozen[zone] = z.history()
            return z.forecast(action, hist=frozen[zone])

        return {"get_zone_state": get_zone_state, "check_camera": check_camera, "cross_check_count": cross_check_count,
                "measure_exit_flow": measure_exit_flow, "_forecast": _forecast}

    # -- the loop --------------------------------------------------------------------
    def evaluate(self, name: str) -> None:
        """Tier policy for one zone, and an episode when it reaches WARN."""
        z = self.zones()[name]
        fc = z.forecast()
        with self.lock:
            s = z.state
            s.minutes_to_act = None if fc.seconds_to_threshold is None else fc.seconds_to_threshold / 60
            d = self.policy.decide(ZoneReading(name, s.t, s.rho, s.pressure, s.minutes_to_act, s.trusted, s.trust_reason))
            if d.tier.name != s.tier:
                self.trace.write("tier", zone=name, t=round(s.t, 1), tier=d.tier.name, reasons=d.reasons,
                                 rho=round(s.rho, 2), pressure=round(s.pressure, 4))
            s.tier, s.reasons = d.tier.name, d.reasons
            if d.tier >= Tier.WARN and name not in self.active_episode and not self._pending(name):
                self._start_episode(name, d.tier, d.reasons)

    def tick(self) -> None:
        """One second on every zone, serially (tests and evaluation use this)."""
        for name, z in self.zones().items():
            z.tick(1.0)
            self.evaluate(name)
        with self.lock:
            self._verify()

    def _watchdog(self) -> None:
        """Checks the camera every DRIFT_EVERY_S. Losing trust opens an episode straight away,
        even while a crowd proposal for the zone is waiting: that proposal rests on this camera."""
        cam = self.camera
        while self.running:
            time.sleep(DRIFT_EVERY_S)
            if self._idle() or cam is None:
                continue
            try:
                was = cam.state.trusted
                r = cam.check_camera()
                if was and not r["trusted"]:
                    self.trace.write("drift", zone=cam.name, camera=cam.camera_name, **r)
                    with self.lock:
                        if cam.name not in self.active_episode:
                            self._start_episode(cam.name, Tier.WARN, [r["reason"] or "camera check failed"])
                elif not was and r["trusted"]:
                    self.trace.write("drift", zone=cam.name, camera=cam.camera_name, **r)
            except Exception as e:
                self.trace.write("error", where="watchdog", error=f"{type(e).__name__}: {e}")

    def _pending(self, zone: str) -> bool:
        return any(p.zone == zone and p.status in ("pending", "approved") and not p.verify.get("done") for p in self.proposals.values())

    def _start_episode(self, zone: str, tier: Tier, reasons: list[str]) -> None:
        ep = Episode(zone, tier, reasons, self.tools_for(zone), CATALOGUE, self.trace, clock=lambda: self.zones()[zone].state.t)
        self.episodes[ep.id] = ep
        self.active_episode[zone] = ep.id

        def work():
            try:
                investigate(ep, use_model=self.use_model)
            finally:
                with self.lock:
                    if ep.proposal:
                        if zone == CameraZone.name:
                            ep.proposal.replay = True
                        self.proposals[ep.proposal.id] = ep.proposal
                        self.trace.write("proposal", **ep.proposal.to_json())
                    if ep.recalibration:
                        self.recalibrations.append({**ep.recalibration, "zone": zone, "episode": ep.id, "t": time.time()})
                    self.active_episode.pop(zone, None)

        threading.Thread(target=work, daemon=True).start()

    def decide(self, proposal_id: str, approve: bool, who: str = "steward") -> dict:
        with self.lock:
            p = self.proposals.get(proposal_id)
            if p is None or p.status != "pending":
                return {"ok": False, "error": "no pending proposal with that id"}
            p.status = "approved" if approve else "rejected"
            p.decided_by = who
            z = self.zones()[p.zone]
            result = ""
            if approve and isinstance(z, TwinZone):
                result = z.apply(p.action)
                p.verify = {"due_t": z.state.t + VERIFY_AFTER_S, "start_t": z.state.t, "start_rho": z.state.rho,
                            "done": False}
                self.applied.append({"proposal": p.id, "zone": p.zone, "t": z.state.t, "result": result})
            elif approve:
                p.verify = {"done": True, "outcome": "decision recorded; the recording cannot change"}
            else:
                p.verify = {"done": True, "outcome": "rejected by the steward"}
            self.trace.write("decision", proposal=p.id, zone=p.zone, approved=approve, by=who, applied=result)
            return {"ok": True, "status": p.status, "applied": result}

    def _verify(self) -> None:
        """After an approved action, compare the measured density with the forecast curve.
        Proposals nobody decided on expire after PROPOSAL_TTL_S."""
        for p in list(self.proposals.values()):
            if p.status == "pending" and self.zones()[p.zone].state.t - p.created_t > PROPOSAL_TTL_S:
                p.status = "expired"
                p.verify = {"done": True, "outcome": f"no decision within {PROPOSAL_TTL_S / 60:.0f} minutes"}
                self.trace.write("expired", proposal=p.id, zone=p.zone)
                continue
            v = p.verify
            if p.status != "approved" or not v or v.get("done"):
                continue
            z = self.zones()[p.zone]
            if z.state.t < v["due_t"]:
                continue
            elapsed = z.state.t - v["start_t"]
            pred = float(np.interp(elapsed, p.predicted["curve_t"], p.predicted["curve_rho"]))
            obs = z.state.rho
            tol = max(0.4, 0.15 * pred)
            ok = obs <= pred + tol
            v.update({"done": True, "elapsed_s": round(elapsed), "predicted_rho": round(pred, 2), "observed_rho": round(obs, 2),
                      "outcome": "working as predicted" if ok else "not working: escalated"})
            self.trace.write("verify", proposal=p.id, zone=p.zone, **{k: v[k] for k in ("elapsed_s", "predicted_rho", "observed_rho", "outcome")})
            if not ok:
                self._start_episode(p.zone, Tier.ACT, [f"approved action {p.action['kind']} did not bend the curve: "
                                                       f"observed {obs:.2f}/m2 against {pred:.2f} predicted"])

    def viewed(self) -> None:
        """A console polled the state. After a long idle pause the evening starts again, so
        whoever arrives sees it from the beginning."""
        now = time.time()
        if self.paused_since is not None and now - self.paused_since > 300:
            self.paused_since = None
            self.restart_evening(force=True)
        self.last_view = now

    def _idle(self) -> bool:
        if IDLE_PAUSE_S <= 0:
            return False
        idle = time.time() - self.last_view > IDLE_PAUSE_S
        if idle and self.paused_since is None:
            self.paused_since = time.time()
            self.trace.write("paused", reason=f"no console for {IDLE_PAUSE_S:.0f} s")
        elif not idle:
            self.paused_since = None
        return idle

    def _maybe_restart_evening(self) -> None:
        if self.twin.state.t >= TWIN_CYCLE_S:
            self.restart_evening()

    def restart_evening(self, force: bool = False) -> None:
        """The twin's evening ends after TWIN_CYCLE_S; a new one starts once nothing is open."""
        name = self.twin.name
        with self.lock:
            if not force and (name in self.active_episode or
                              any(p.zone == name and p.status == "pending" for p in self.proposals.values())):
                return
        self.seed += 1
        fresh = TwinZone(seed=self.seed, state_lock=self.lock, warmup_s=self.twin_warmup_s)
        with self.lock:
            self.twin = fresh
            self.policy._held.pop(name, None)
            for p in list(self.proposals.values()):
                if p.zone == name:
                    del self.proposals[p.id]
            self.applied = [a for a in self.applied if a["zone"] != name]
            self.evenings += 1
            if force:
                for p in list(self.proposals.values()):
                    if p.status == "pending":
                        del self.proposals[p.id]
                self.recalibrations.clear()
            self.trace.write("scenario", zone=name, event="a new evening starts", seed=self.seed)

    def snapshot(self) -> dict:
        fp = self.twin.floorplan()
        with self.lock:
            return {
                "zones": {n: z.state.to_json() for n, z in self.zones().items()},
                "floorplan": fp,
                "proposals": [p.to_json() for p in sorted(self.proposals.values(), key=lambda p: -p.created_t)][:6],
                "episodes": [{"id": e.id, "zone": e.zone, "planner": e.planner, "calls": e.calls[-14:],
                              "done": e.done, "active": self.active_episode.get(e.zone) == e.id}
                             for e in list(self.episodes.values())[-4:]],
                "recalibrations": self.recalibrations[-4:],
                "applied": self.applied[-6:],
                "speed": self.speed,
                "evening": self.evenings,
                "camera_timing_ms": None if self.camera is None or not self.camera.timing
                else round(float(np.mean(self.camera.timing)) * 1000),
                "twin_timing_ms": round(float(np.mean(self.twin.timing)) * 1000) if self.twin.timing else None,
            }

    def _zone_loop(self, name: str) -> None:
        while self.running:
            if self._idle():
                time.sleep(1.0)
                continue
            t0 = time.perf_counter()
            try:
                z = self.zones()[name]
                z.tick(1.0)
                self.evaluate(name)
                with self.lock:
                    self._verify()
                if name == self.twin.name:
                    self._maybe_restart_evening()
            except Exception as e:
                self.trace.write("error", where=f"tick {name}", error=f"{type(e).__name__}: {e}")
                time.sleep(1.0)
            dt = time.perf_counter() - t0
            speed = self.speed if name == self.twin.name else 1.0
            time.sleep(max(0.0, 1.0 / speed - dt))

    def loop(self) -> None:
        """Run every zone on its own thread until `running` is cleared."""
        self.running = True
        threads = [threading.Thread(target=self._zone_loop, args=(n,), daemon=True, name=f"zone:{n}") for n in self.zones()]
        if self.camera is not None:
            threads.append(threading.Thread(target=self._watchdog, daemon=True, name="watchdog"))
        for th in threads:
            th.start()
        for th in threads:
            th.join()
