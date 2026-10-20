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

from . import jobs
from . import twin as tw
from .agent import Episode, Proposal, VenueCatalogue, investigate
from .calib import FloorGrid
from .density import DEFAULT_ENGINE
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
SCREEN_EVERY_S = float(os.environ.get("BR_SCREEN_EVERY_S", "3"))  # ECC screen period
CONFIRM_EVERY_S = float(os.environ.get("BR_CONFIRM_EVERY_S", "600"))  # ALIKED + LightGlue even when the screen is steady
# ALIKED + LightGlue need about 1.6 GB while they run and DM-Count about 0.7 GB. On a small
# instance each call runs in a fresh child process, so the memory goes back to the system.
ISOLATE_MODELS = os.environ.get("BR_ISOLATE_MODELS", "0") == "1"
WORK_SCALE = float(os.environ.get("BR_WORK_SCALE", "0.5"))  # camera frames are processed at this fraction of full size
# One heavy model at a time: two at once do not fit in 2 GB and push the instance into swap.
HEAVY = threading.Semaphore(1)
# The second counter sees only the zone's bounding box, at half the undistorted resolution.
COUNT_SCALE = 0.5
# Camera crops shown on proposal cards are kept in memory only, at most this many, this long:
# long enough for a steward to decide (a proposal expires after PROPOSAL_TTL_S) and to look back.
EVIDENCE_KEEP = 24
EVIDENCE_TTL_S = float(os.environ.get("BR_EVIDENCE_TTL_S", "900"))
# A zone whose source has produced nothing for this long is not trusted: its numbers are stale.
FEED_STALE_S = float(os.environ.get("BR_FEED_STALE_S", "10"))


def zone_crop(und: np.ndarray, H_w2u: np.ndarray, rect: tuple, scale: float = COUNT_SCALE):
    """The zone's bounding box in the undistorted image, resized by `scale`, and the zone's
    polygon as a mask over it. The second counter runs on this crop only."""
    x0, x1, y0, y1 = rect
    poly = cv2.perspectiveTransform(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float).reshape(-1, 1, 2),
                                    H_w2u).reshape(-1, 2)
    h, w = und.shape[:2]
    bx0, by0 = np.clip(np.floor(poly.min(0)).astype(int), 0, [w, h])
    bx1, by1 = np.clip(np.ceil(poly.max(0)).astype(int), 0, [w, h])
    crop = und[by0:by1, bx0:bx1]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [bx0, by0]).astype(np.int32)], 1)
    size = (max(16, int(round(crop.shape[1] * scale / 16)) * 16), max(16, int(round(crop.shape[0] * scale / 16)) * 16))
    return (cv2.resize(crop, size, interpolation=cv2.INTER_AREA),
            cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))


def _isolated(fn, *args):
    """Run `fn` in a fresh child process that exits afterwards."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with HEAVY, ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result(timeout=180)

# DM-Count count / head count inside the E1 zone, as `cross_check_count` computes them, over 398
# frames of the six calibrated Jülich runs (eval/results/test_e1_count_band.json): 2.5th
# percentile 0.22, 97.5th 0.79, r = 0.91. DM-Count undercounts on this overhead view; a ratio
# outside the band means one of the two counters is not seeing the crowd (occlusion, lighting,
# a moved camera).
E1_COUNT_RATIO_BAND = (0.22, 0.79)


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
    rho_peak: float | None = None  # densest spot in the zone (camera zones), people/m^2
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
            "zone": self.zone, "t": self.t, "rho": round(self.rho, 2),
            "rho_peak": None if self.rho_peak is None else round(self.rho_peak, 1), "pressure": round(self.pressure, 4),
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
    zone_rect_default = (-2.8, 2.8, -1.8, 2.2)  # 5.6 m x 4 m in front of the entrance (x0, x1, y0, y1)

    def __init__(self, video: Path, calib_json: Path, run: str, camera_name: str = "cam-E1", state_lock=None,
                 work_scale: float = WORK_SCALE, grid: tuple | None = None):
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
        # Frames are reduced once with area averaging, then undistorted and searched for heads at
        # that size; DIS flow runs at 960 px wide either way. Heads are returned in full-size
        # undistorted pixels, which is what the homography expects.
        self.ws = work_scale
        self.work_size = (int(round(self.W * work_scale)), int(round(self.H * work_scale)))
        self.S = fc.scaled_k(np.eye(3), work_scale)  # full-size undistorted px -> working px
        self.m1, self.m2 = cv2.initUndistortRectifyMap(fc.scaled_k(lens.K, work_scale), lens.D, None,
                                                       fc.scaled_k(lens.Kn, work_scale), self.work_size, cv2.CV_16SC2)
        self.flow = Flow("medium", 960)
        self.camera_name = camera_name
        # floor grid over the area in front of the gate (run frame of reference, head plane)
        self.grid = FloorGrid(*(grid or (-3.5, -2.5, 3.5, 7.0)), cell=0.1)
        self.builder = FieldBuilder(self.grid, self.H_u2w, (self.W, self.H))
        self.zone_rect = self.zone_rect_default
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
        self.last_screen: dict | None = None
        self.last_confirm_t = 0.0  # wall clock of the last ALIKED + LightGlue check
        self.last_P: np.ndarray | None = None
        self.last_rho: np.ndarray | None = None
        self.evidence: dict[str, tuple[float, bytes]] = {}  # id -> (wall time kept, JPEG), in memory only
        self.screen = None
        self.screen_frames = [self.reader.read(int(i)) for i in np.linspace(0, min(1000, self.n_frames - 3), 9)]

    def probe_px(self) -> np.ndarray:
        """The zone's corners in the original image: the points the watchdog follows."""
        corners = np.array([[x, y] for x in self.zone_rect[:2] for y in self.zone_rect[2:]], float)
        return self.lens.distort(cv2.perspectiveTransform(corners.reshape(-1, 1, 2), self.H_w2u).reshape(-1, 2))

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
            ua, heads = self.prepare(a)
            ub, _ = self.prepare(b, heads=False)
            fl = self.flow(self.flow.prepare(ua), self.flow.prepare(ub))
            fr = self.builder.build_from_points(f / self.fps, heads, fl, 2 / self.fps)
            self.last_field = (f, fr)
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
            self.last_P, self.last_rho = P, rho
        with self.state_lock:
            s = self.state
            if restarted:
                self.loops += 1
                s.clear_history()
            s.t += seconds
            s.n = float(mass[inz].sum())
            s.rho = s.n / self.area
            s.rho_peak = float(rho[inz].max()) if inz.any() else None
            s.pressure = float(np.percentile(P[occupied], 90)) if occupied.sum() > 5 else 0.0
            s.outflow_pps = float(np.sum(self.flux_window)) / (len(self.flux_window) * 0.2) if self.flux_window else 0.0
            s.los = level_of_service(s.rho)
            s.push()
        self.timing.append(time.perf_counter() - t0)
        self._render(last_und, last_heads, P)

    def prepare(self, img: np.ndarray, heads: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """Undistorted working frame, and the heads in full-size undistorted pixels."""
        small = img if self.ws == 1.0 else cv2.resize(img, self.work_size, interpolation=cv2.INTER_AREA)
        und = cv2.remap(small, self.m1, self.m2, cv2.INTER_LINEAR)
        if not heads:
            return und, None
        c = self.fc.detect_caps(small, scale=self.ws)
        full = (c + 0.5) / self.ws - 0.5 if len(c) else c  # working px -> original px
        return und, self.lens.undistort(full)

    def count_crop(self, und: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """What the second counter sees: the zone's bounding box at COUNT_SCALE of full size."""
        return zone_crop(und, self.S @ self.H_w2u, self.zone_rect, COUNT_SCALE / self.ws)

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
        img = und if und.shape[1] == 960 else cv2.resize(und, (960, 720), interpolation=cv2.INTER_AREA)
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

    def screen_camera(self) -> dict:
        """ECC screen of the latest frame against the calibration frame (tens of ms)."""
        from .drift import ShiftScreen

        if self.screen is None:
            self.screen = ShiftScreen(self.screen_frames, self.to_floor, self.probe_px())
            self.screen_frames = []
        with self.frame_lock:
            live = self.last_raw
        r = self.screen.check(live).to_json()
        self.last_screen = {**r, "t": self.state.t}
        return r

    def confirm_camera(self) -> dict:
        """ALIKED + LightGlue (OpenCV 5 features module) against the calibration frame."""
        from .drift import DriftWatch

        probe = self.probe_px()
        with self.frame_lock:
            live = self.last_raw
        with self.drift_lock:
            if ISOLATE_MODELS:
                r = _isolated(jobs.drift_job, self.ref_frame, live, self.lens.x, self.H_u2w, probe)
            else:
                if self.drift is None:
                    self.drift = DriftWatch(self.ref_frame, self.to_floor, probe)
                with HEAVY:
                    r = self.drift.check(live).to_json()
        self.last_confirm_t = time.time()
        return {"method": "ALIKED + LightGlue", **r}

    def check_camera(self, force: bool = False) -> dict:
        """Two stages. The ECC screen runs first; a steady screen on a trusted camera is the
        answer, and a screen that measures a shift beyond the tolerance withdraws trust at once.
        A smaller shift, a low correlation, a camera already flagged, or `force` sends the frame
        to ALIKED + LightGlue, which decides; only that check restores trust."""
        sc = self.screen_camera()
        with self.state_lock:
            trusted_now = self.state.trusted
        if sc["verdict"] == "steady" and trusted_now and not force:
            r = {"trusted": True, "floor_shift_m": sc["floor_shift_m"], "reason": "", "method": "ECC screen",
                 "screen": sc, "escalated": False}
        elif sc["verdict"] == "moved" and sc["floor_shift_m"] > self.screen.tol and trusted_now and not force:
            # trust is withdrawn on the screen's measurement at once; only the features check gives it back
            r = {"trusted": False, "floor_shift_m": sc["floor_shift_m"], "method": "ECC screen", "screen": sc,
                 "escalated": False,
                 "reason": f"camera moved: calibration points shifted {sc['floor_shift_m']:.2f} m on the floor (ECC screen)"}
        else:
            try:
                full = self.confirm_camera()
            except Exception as e:
                msg = f"features check failed ({type(e).__name__}: {str(e)[:120]})"
                if sc["verdict"] == "steady" and trusted_now:
                    # a periodic confirmation that could not run: the steady screen stands
                    full = {"method": "ECC screen", "trusted": True, "floor_shift_m": sc["floor_shift_m"], "reason": "",
                            "note": msg}
                else:  # the screen saw motion and nothing could confirm it: do not trust the camera
                    full = {"method": "ALIKED + LightGlue", "trusted": False, "floor_shift_m": None,
                            "reason": msg + "; camera unverified"}
            r = {**full, "screen": sc, "escalated": True}
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
        crop, mask = self.count_crop(und)
        if ISOLATE_MODELS:
            count, secs = _isolated(jobs.count_job, crop, mask)
        else:
            if self.counter is None:
                self.counter = DensityModel("sha", max_side=4096)
            with HEAVY:
                r = self.counter(crop, roi_mask=mask)
            count, secs = r.count, r.seconds
        w = cv2.perspectiveTransform(np.asarray(heads, float).reshape(-1, 1, 2), self.H_u2w).reshape(-1, 2) \
            if len(heads) else np.zeros((0, 2))
        n_heads = int(((w[:, 0] >= x0) & (w[:, 0] < x1) & (w[:, 1] >= y0) & (w[:, 1] < y1)).sum())
        ratio = count / max(n_heads, 1)
        lo, hi = E1_COUNT_RATIO_BAND
        agree = n_heads >= 3 and lo <= ratio <= hi
        return {"count_primary": n_heads, "count_second": round(count, 1), "ratio": round(ratio, 2),
                "agreement_band": [lo, hi], "agree": bool(agree), "second_counter_ms": round(secs * 1000), "dnn_engine": DEFAULT_ENGINE,
                "note": "head detector vs DM-Count (OpenCV 5 DNN); band measured on six Jülich runs"}

    def jpeg(self) -> bytes | None:
        with self.frame_lock:
            return self.last_jpeg

    def locate_pushing(self, threshold: float = Thresholds().warn_pressure, min_area_m2: float = 0.25) -> dict:
        """Where in the zone the crowd is pushing. The pushing index (2 s mean) is thresholded on
        the floor grid and split into connected regions (`connectedComponentsWithStats`); each
        region is placed relative to the gate and cut out of the camera image as evidence for
        the steward. Sides are as people see them walking towards the gate."""
        with self.frame_lock:
            P, rho, und = self.last_P, self.last_rho, self.last_und
        if P is None or und is None:
            return {"error": "no frame processed yet"}
        x0, x1, y0, y1 = self.zone_rect
        c = self.grid.centres().reshape(*self.grid.shape, 2)
        inz = (c[..., 0] >= x0) & (c[..., 0] < x1) & (c[..., 1] >= y0) & (c[..., 1] < y1)
        m = ((P >= threshold) & inz).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cell_a = self.grid.cell ** 2
        gate_y, gate_x = self.zone_rect[2], (self.gate_line.a[0] + self.gate_line.b[0]) / 2
        spots = []
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA] * cell_a
            if area < min_area_m2:
                continue
            sel = lab == i
            k = np.argmax(np.where(sel, P, -1))
            px, py = c.reshape(-1, 2)[k]
            lateral = float(px - gate_x)
            side = "centre" if abs(lateral) < 0.6 else ("left" if lateral > 0 else "right")
            spots.append({"area_m2": round(float(area), 2), "peak_pushing_index": round(float(P[sel].max()), 3),
                          "density": round(float(rho[sel].max()), 1), "people": round(float(rho[sel].sum() * cell_a), 1),
                          "metres_before_gate": round(float(py - gate_y), 1), "side": side,
                          "_box": (c[..., 0][sel].min(), c[..., 0][sel].max(), c[..., 1][sel].min(), c[..., 1][sel].max())})
        spots.sort(key=lambda h: -h["peak_pushing_index"])
        spots = spots[:3]
        for h in spots:
            h["evidence_image"] = self._evidence(und, P, h.pop("_box"))
        return {"hotspots": spots, "threshold": threshold,
                "note": "regions of the zone where the pushing index is above the alert threshold, largest first; "
                        "side as seen walking towards the gate"}

    def _evidence(self, und: np.ndarray, P: np.ndarray, box: tuple) -> str:
        """Camera crop around a hotspot, with the pushing heat and the hotspot outlined."""
        import uuid

        bx0, bx1, by0, by1 = box
        pad = 0.8
        poly_m = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], float)
        view_m = np.array([[bx0 - pad, by0 - pad], [bx1 + pad, by0 - pad], [bx1 + pad, by1 + pad], [bx0 - pad, by1 + pad]], float)
        G = self.S @ self.H_w2u  # floor metres -> working px
        img = und.copy()
        Pn = np.clip(P / 0.25, 0, 1)
        A = np.array([[1 / self.grid.cell, 0, -self.grid.x0 / self.grid.cell - 0.5],
                      [0, 1 / self.grid.cell, -self.grid.y0 / self.grid.cell - 0.5], [0, 0, 1]])
        warped = cv2.warpPerspective((Pn * 255).astype(np.uint8), G @ np.linalg.inv(A), (img.shape[1], img.shape[0]))
        col = cv2.applyColorMap(warped, cv2.COLORMAP_INFERNO)
        alpha = (warped.astype(np.float32) / 255.0 * 0.5)[..., None]
        img = (img * (1 - alpha) + col * alpha).astype(np.uint8)
        hp = cv2.perspectiveTransform(poly_m.reshape(-1, 1, 2), G).reshape(-1, 2)
        cv2.polylines(img, [hp.astype(np.int32)], True, (255, 255, 255), 3, cv2.LINE_AA)
        vp = cv2.perspectiveTransform(view_m.reshape(-1, 1, 2), G).reshape(-1, 2)
        h, w = img.shape[:2]
        x0, y0 = np.clip(vp.min(0).astype(int), 0, [w - 1, h - 1])
        x1, y1 = np.clip(vp.max(0).astype(int), 1, [w, h])
        crop = img[y0:y1, x0:x1]
        s = 480 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        eid = "ev_" + uuid.uuid4().hex[:10]
        with self.frame_lock:
            self.evidence[eid] = (time.time(), buf.tobytes())
            for old in list(self.evidence)[:-EVIDENCE_KEEP]:
                del self.evidence[old]
        return eid

    def evidence_image(self, eid: str) -> bytes | None:
        """A crop is served while it is younger than EVIDENCE_TTL_S, and never written to disk."""
        with self.frame_lock:
            now = time.time()
            for old in [k for k, (t, _) in self.evidence.items() if now - t > EVIDENCE_TTL_S]:
                del self.evidence[old]
            kept = self.evidence.get(eid)
        return kept[1] if kept else None


RESULTS = ROOT / "eval" / "results"


def forecast_skill(path: Path = RESULTS / "test_b_twin.json") -> dict[str, dict[str, float]]:
    """Per action, the mean and the largest amount (people/m2) by which the twin's peak exceeded
    the forecast peak over the test B seeds (eval/test_b_twin.py). Proposals rank actions on peak
    + the mean error; the venue-wide rule (choice.py) judges safety on peak + the largest."""
    try:
        seeds = [s["actions"] for s in json.loads(path.read_text())["seeds"].values() if s.get("actions")]
    except (OSError, KeyError, ValueError, AttributeError):
        return {}
    out = {}
    for k in seeds[0] if seeds else []:
        d = [s[k]["simulated_peak"] - s[k]["predicted_peak"] for s in seeds if k in s]
        out[k] = {"mean": round(max(0.0, float(np.mean(d))), 2), "max": round(max(0.0, float(np.max(d))), 2)}
    return out


def queue_skill(path: Path = RESULTS / "test_b_policy.json") -> dict[str, float]:
    """Per action that holds people back, the largest ratio of the queue the twin built before
    the gates to the queue forecast, on the reference venue only (eval/test_b_policy.py)."""
    try:
        runs = json.loads(path.read_text())["venues"]["reference venue"].values()
    except (OSError, KeyError, ValueError, AttributeError):
        return {}
    out: dict[str, float] = {}
    for r in runs:
        for k, a in r.get("actions", {}).items():
            if a.get("predicted_held", 0) >= 1:
                out[k] = round(max(out.get(k, 1.0), a["simulated_held"] / a["predicted_held"]), 2)
    return out


SKILL = forecast_skill()
QUEUE_SKILL = queue_skill()
_TWIN_SKILL_KEYS = {("meter_entry", "both gates"): "meter both gates to 150/min", ("meter_entry", "gate A"): "meter gate A to 60/min",
                    ("meter_entry", "gate B"): "meter gate A to 60/min", ("open_exit", "second exit door"): "open second exit door",
                    ("hold_entry", "both gates"): "hold both gates", ("none", ""): "no action"}


def _with_skill(zone_actions: list[dict]) -> list[dict]:
    for a in zone_actions:
        key = _TWIN_SKILL_KEYS.get((a["kind"], a.get("target", "")))
        if key in SKILL:
            a["forecast_error"] = SKILL[key]["mean"]
            a["forecast_error_max"] = SKILL[key]["max"]
        if key in QUEUE_SKILL:
            a["queue_factor"] = QUEUE_SKILL[key]
    return zone_actions


# Waiting areas are venue inputs: the space before each entrance where the people an action
# holds back stand. The demonstration assumes 60 m2 before each gate of Station Road (half the
# street's 6 m width, 20 m deep) and 40 m2 of corridor feeding the E1 entrance.
GATE_AREA_M2 = 60.0
CATALOGUE = VenueCatalogue({
    TwinZone.name: _with_skill([
        {"kind": "meter_entry", "target": "gate A", "rate_per_min": 60, "share": 0.5, "camera": "cam-C1",
         "holding_area_m2": GATE_AREA_M2},
        {"kind": "meter_entry", "target": "gate B", "rate_per_min": 60, "share": 0.5, "holding_area_m2": GATE_AREA_M2},
        {"kind": "meter_entry", "target": "both gates", "rate_per_min": 150, "holding_area_m2": 2 * GATE_AREA_M2},
        {"kind": "open_exit", "target": "second exit door", "width_m": tw.SIDE_WIDTH},
        {"kind": "hold_entry", "target": "both gates", "holding_area_m2": 2 * GATE_AREA_M2},
    ]),
    CameraZone.name: [
        {"kind": "send_steward", "target": "entrance gate", "camera": "cam-E1"},
        {"kind": "announce", "target": "please stop pushing, the gate is open"},
        {"kind": "meter_entry", "target": "queue feed", "rate_per_min": 30, "holding_area_m2": 40.0},
        {"kind": "hold_entry", "target": "queue feed", "holding_area_m2": 40.0},
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
        self.last_ok: dict[str, float] = {}  # wall time of each zone's last processed second
        self.feed_lost: dict[str, str] = {}  # zone -> why its numbers are stale

    def feed_status(self, name: str) -> str:
        lost = self.feed_lost.get(name)
        return f"lost: {lost}" if lost else "live"

    def _feed(self, name: str, ok: bool, error: str = "") -> None:
        """Record whether a zone's source produced its last second. A zone silent for FEED_STALE_S
        is marked lost: the tier policy treats it as untrusted, a proposal cannot rest on it, and
        /health reports it so a monitor outside the instance can raise an alarm."""
        now = time.time()
        if ok:
            self.last_ok[name] = now
            if self.feed_lost.pop(name, None) is not None:
                self.trace.write("feed", zone=name, status="restored")
            return
        silent = now - self.last_ok.get(name, now)
        if silent >= FEED_STALE_S and name not in self.feed_lost:
            self.feed_lost[name] = f"no data for {silent:.0f} s ({error})" if error else f"no data for {silent:.0f} s"
            self.trace.write("feed", zone=name, status="lost", reason=self.feed_lost[name])
            self.evaluate(name)

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
                        "outflow_per_min": round(s.outflow_pps * 60, 1), "feed": self.feed_status(zone),
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

        def locate_pushing(zone: str) -> dict:
            z = self.zones()[zone]
            if isinstance(z, CameraZone):
                return z.locate_pushing()
            return {"hotspots": [], "note": "simulated zone: the twin models crowd numbers, not pushing"}

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
                "locate_pushing": locate_pushing, "measure_exit_flow": measure_exit_flow, "_forecast": _forecast}

    # -- the loop --------------------------------------------------------------------
    def evaluate(self, name: str) -> None:
        """Tier policy for one zone, and an episode when it reaches WARN."""
        z = self.zones()[name]
        fc = z.forecast()
        with self.lock:
            s = z.state
            s.minutes_to_act = None if fc.seconds_to_threshold is None else fc.seconds_to_threshold / 60
            lost = self.feed_lost.get(name)
            d = self.policy.decide(ZoneReading(name, s.t, s.rho, s.pressure, s.minutes_to_act, s.trusted and not lost,
                                               f"feed lost, {lost}" if lost else s.trust_reason, s.rho_peak))
            if d.tier.name != s.tier:
                self.trace.write("tier", zone=name, t=round(s.t, 1), tier=d.tier.name, reasons=d.reasons,
                                 rho=round(s.rho, 2), pressure=round(s.pressure, 4))
            s.tier, s.reasons = d.tier.name, d.reasons
            # stale numbers are shown as such and reported by /health; the assistant does not investigate them
            if d.tier >= Tier.WARN and not lost and name not in self.active_episode and not self._pending(name):
                self._start_episode(name, d.tier, d.reasons)

    def tick(self) -> None:
        """One second on every zone, serially (tests and evaluation use this)."""
        for name, z in self.zones().items():
            z.tick(1.0)
            self.evaluate(name)
        with self.lock:
            self._verify()

    def _watchdog(self) -> None:
        """Screens the camera every SCREEN_EVERY_S with ECC and confirms with ALIKED + LightGlue
        when the screen is not steady, and every CONFIRM_EVERY_S regardless. Losing trust opens
        an episode straight away, even while a crowd proposal for the zone is waiting: that
        proposal rests on this camera."""
        cam = self.camera
        while self.running:
            time.sleep(SCREEN_EVERY_S)
            if self._idle() or cam is None:
                continue
            try:
                was = cam.state.trusted
                r = cam.check_camera(force=time.time() - cam.last_confirm_t > CONFIRM_EVERY_S)
                if r["escalated"] and r["trusted"] == was and r["screen"]["verdict"] != "steady":
                    self.trace.write("drift_screen", zone=cam.name, camera=cam.camera_name, screen=r["screen"],
                                     confirmed=r["method"], trusted=r["trusted"])
                if r.get("note"):
                    self.trace.write("error", where="watchdog", error=r["note"])
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

    def decide(self, proposal_id: str, approve: bool, who: str = "steward", auth: str = "pin") -> dict:
        with self.lock:
            p = self.proposals.get(proposal_id)
            if p is None or p.status != "pending":
                return {"ok": False, "error": "no pending proposal with that id"}
            p.status = "approved" if approve else "rejected"
            p.decided_by = who
            z = self.zones()[p.zone]
            result = ""
            if approve and isinstance(z, TwinZone):
                # the zone has moved on since the proposal: forecast the approved action again from
                # its current state, and judge the action against that forecast
                a = p.action
                spec = next((x for x in CATALOGUE.actions.get(p.zone, [])
                             if x["kind"] == a["kind"] and x.get("target", "") == a.get("target", "")), {})
                fc = z.forecast(Action(a["kind"], a.get("target", ""), rate_pps=a.get("rate_per_min", 0) / 60.0,
                                       width_m=a.get("width_m", 0.0), share=spec.get("share", 1.0)))
                p.predicted_at_proposal = dict(p.predicted)
                p.predicted = {"peak_density": round(fc.peak_rho, 2), "curve_t": fc.curve_t,
                               "curve_rho": [round(x, 3) for x in fc.curve_rho],
                               "minutes_to_threshold": None if fc.seconds_to_threshold is None
                               else round(fc.seconds_to_threshold / 60, 1), "at_approval": True}
                result = z.apply(p.action)
                p.verify = {"due_t": z.state.t + VERIFY_AFTER_S, "start_t": z.state.t, "start_rho": z.state.rho,
                            "done": False}
                self.applied.append({"proposal": p.id, "zone": p.zone, "t": z.state.t, "result": result})
            elif approve:
                p.verify = {"done": True, "outcome": "decision recorded; the recording cannot change"}
            else:
                p.verify = {"done": True, "outcome": "rejected by the steward"}
            self.trace.write("decision", proposal=p.id, zone=p.zone, approved=approve, by=who, auth=auth, applied=result,
                             predicted_peak=p.predicted.get("peak_density"))
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
            # the action was chosen as safe with the largest error its forecast made in testing, so it
            # is judged with that error too; a zone at the ACT density is never "working"
            err = float(p.action.get("forecast_error_max") or 0.0)
            allowed = pred + max(tol, err)
            ok = obs <= allowed and obs < self.policy.th.act_rho
            outcome = ("working as predicted" if obs <= pred + tol else
                       "working, within its tested forecast error" if ok else "not working: escalated")
            v.update({"done": True, "elapsed_s": round(elapsed), "predicted_rho": round(pred, 2), "observed_rho": round(obs, 2),
                      "allowed_rho": round(allowed, 2), "outcome": outcome})
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

    def health(self) -> dict:
        """For a monitor outside the instance: every zone has produced data within FEED_STALE_S,
        or the console is paused because nobody is watching it."""
        now = time.time()
        if self.paused_since is not None:
            return {"ok": True, "running": self.running, "paused": True, "zones": {n: "paused" for n in self.zones()}}
        zones = {}
        for n in self.zones():
            silent = now - self.last_ok.get(n, now)
            zones[n] = (f"lost: {self.feed_lost[n]}" if n in self.feed_lost
                        else f"stale: no data for {silent:.0f} s" if silent >= FEED_STALE_S else "live")
        return {"ok": self.running and all(v == "live" for v in zones.values()), "running": self.running,
                "paused": False, "zones": zones}

    def _zone_loop(self, name: str) -> None:
        fails = 0
        self.last_ok.setdefault(name, time.time())
        while self.running:
            if self._idle():
                time.sleep(1.0)
                continue
            t0 = time.perf_counter()
            try:
                z = self.zones()[name]
                z.tick(1.0)
                self._feed(name, True)
                fails = 0
                self.evaluate(name)
                with self.lock:
                    self._verify()
                if name == self.twin.name:
                    self._maybe_restart_evening()
            except Exception as e:
                fails += 1
                if fails == 1 or fails % 60 == 0:  # the first failure, then once a minute while it lasts
                    self.trace.write("error", where=f"tick {name}", error=f"{type(e).__name__}: {e}", failures=fails)
                self._feed(name, False, f"{type(e).__name__}: {e}")
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
