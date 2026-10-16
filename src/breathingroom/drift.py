"""Camera drift watchdog with the OpenCV 5 features module: ALIKED keypoints and the
LightGlue matcher.

At calibration time a reference frame is stored. Periodically the live frame is matched
against it; the homography between the two (USAC MAGSAC++) moves the calibration points,
and their displacement on the floor, in metres, says whether the camera's numbers can still
be trusted. A bumped or re-aimed camera is flagged before its density readings drift.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calib import apply_h
from .density import MODELS_DIR

ALIKED_ONNX = "aliked-n16rot-top1k-640.onnx"
LIGHTGLUE_ONNX = "aliked_lightglue.onnx"


@dataclass
class DriftCheck:
    trusted: bool
    floor_shift_m: float
    image_shift_px: float
    matches: int
    inliers: int
    seconds: float
    reason: str = ""

    def to_json(self) -> dict:
        return {"trusted": self.trusted, "floor_shift_m": round(self.floor_shift_m, 3),
                "image_shift_px": round(self.image_shift_px, 1), "matches": self.matches,
                "inliers": self.inliers, "ms": round(self.seconds * 1000), "reason": self.reason}


class DriftWatch:
    def __init__(self, reference_bgr: np.ndarray, to_floor, probe_px: np.ndarray,
                 mask: np.ndarray | None = None, tolerance_m: float = 0.25, size: int = 640,
                 models_dir: Path = MODELS_DIR):
        """`to_floor` maps image pixels to floor metres: a 3x3 homography, or a function
        (for a lens with distortion)."""
        self.aliked = cv2.ALIKED_create(str(models_dir / ALIKED_ONNX))
        self.lg = cv2.LightGlueMatcher_create(str(models_dir / LIGHTGLUE_ONNX))
        self.to_floor = (lambda p: apply_h(to_floor, p)) if isinstance(to_floor, np.ndarray) else to_floor
        self.probe = np.asarray(probe_px, float).reshape(-1, 2)  # e.g. the clicked calibration points
        self.tol = tolerance_m
        self.size = size
        self.mask = mask
        self.ref = self._features(reference_bgr)

    def _features(self, bgr):
        h, w = bgr.shape[:2]
        s = self.size / max(h, w)
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        m = None
        if self.mask is not None:
            m = cv2.resize(self.mask.astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        k, d = self.aliked.detectAndCompute(img, m)
        pts = np.float32([p.pt for p in k]) / s
        return k, d, pts, (img.shape[1], img.shape[0]), s

    def check(self, bgr: np.ndarray) -> DriftCheck:
        t0 = time.perf_counter()
        k1, d1, p1, sz1, s1 = self.ref
        k2, d2, p2, sz2, s2 = self._features(bgr)
        if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
            return DriftCheck(False, float("inf"), float("inf"), 0, 0, time.perf_counter() - t0, "too few features")
        self.lg.setPairInfo(p1 * s1, p2 * s2, sz1, sz2)
        mt = self.lg.match(d1, d2)
        if len(mt) < 8:
            return DriftCheck(False, float("inf"), float("inf"), len(mt), 0, time.perf_counter() - t0, "too few matches")
        a = np.float32([p1[m.queryIdx] for m in mt])
        b = np.float32([p2[m.trainIdx] for m in mt])
        G, inl = cv2.findHomography(a, b, cv2.USAC_MAGSAC, 3.0)  # reference image -> live image
        if G is None:
            return DriftCheck(False, float("inf"), float("inf"), len(mt), 0, time.perf_counter() - t0, "no consistent motion")
        moved = apply_h(G, self.probe)
        img_shift = float(np.linalg.norm(moved - self.probe, axis=1).max())
        # a point that appears at `moved` is now read by the stale calibration as H(moved)
        floor_shift = float(np.linalg.norm(self.to_floor(moved) - self.to_floor(self.probe), axis=1).max())
        ok = floor_shift <= self.tol
        reason = "" if ok else f"camera moved: calibration points shifted {floor_shift:.2f} m on the floor"
        return DriftCheck(ok, floor_shift, img_shift, len(mt), int(inl.sum()), time.perf_counter() - t0, reason)


@dataclass
class ScreenCheck:
    floor_shift_m: float
    correlation: float
    seconds: float
    verdict: str  # "steady", "moved" or "unsure"

    def to_json(self) -> dict:
        return {"method": "ECC screen", "verdict": self.verdict,
                "floor_shift_m": None if not np.isfinite(self.floor_shift_m) else round(self.floor_shift_m, 3),
                "correlation": round(self.correlation, 3), "ms": round(self.seconds * 1000, 1)}


class ShiftScreen:
    """A cheap first check of the camera: `cv.findTransformECC` aligns the live frame to the
    calibration frame on the pixels that stay still in the recording (barriers, floor, walls),
    coarse to fine. It costs tens of milliseconds, so it can
    run every few seconds. The motion model is a full homography, so a zoom or a tilt is measured
    as well as a roll or a shift. A large shift, or a low correlation (lighting change, a scene
    the model does not explain), sends the camera to the ALIKED + LightGlue check."""

    def __init__(self, frames_bgr: list[np.ndarray], to_floor, probe_px: np.ndarray, tolerance_m: float = 0.25,
                 width: int = 480, min_correlation: float = 0.85):
        self.to_floor = (lambda p: apply_h(to_floor, p)) if isinstance(to_floor, np.ndarray) else to_floor
        self.probe = np.asarray(probe_px, float).reshape(-1, 2)
        self.tol = tolerance_m
        self.min_cc = min_correlation
        h, w = frames_bgr[0].shape[:2]
        self.s = width / w
        self.size = (width, int(round(h * self.s)))
        g = np.stack([self._grey(f) for f in frames_bgr]).astype(np.float32)
        med = np.median(g, 0)
        # still pixels: within 12 grey levels of the median in at least 90% of the frames
        self.mask = ((np.abs(g - med) < 12).mean(0) >= 0.9).astype(np.uint8) * 255
        self.ref = self._grey(frames_bgr[0])
        self.pyr = [(cv2.resize(self.ref, None, fx=1 / 2**k, fy=1 / 2**k, interpolation=cv2.INTER_AREA),
                     cv2.resize(self.mask, (self.size[0] >> k, self.size[1] >> k), interpolation=cv2.INTER_NEAREST))
                    for k in (2, 1, 0)]

    def _grey(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), self.size, interpolation=cv2.INTER_AREA)

    def check(self, bgr: np.ndarray) -> ScreenCheck:
        t0 = time.perf_counter()
        live = self._grey(bgr)
        warp = np.eye(3)  # reference -> live, at working size
        cc = 0.0
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        for (ref, mask), k in zip(self.pyr, (2, 1, 0)):
            D = np.diag([1 / 2**k, 1 / 2**k, 1.0])
            lv = cv2.resize(live, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            w = (D @ warp @ np.linalg.inv(D)).astype(np.float32)
            try:
                cc, w = cv2.findTransformECC(ref, lv, w, cv2.MOTION_HOMOGRAPHY, crit, mask, 5)
            except cv2.error:
                return ScreenCheck(float("inf"), 0.0, time.perf_counter() - t0, "unsure")
            warp = np.linalg.inv(D) @ w.astype(np.float64) @ D
        S = np.diag([self.s, self.s, 1.0])
        G = np.linalg.inv(S) @ warp @ S  # reference -> live, full-size pixels
        moved = apply_h(G, self.probe)
        shift = float(np.linalg.norm(self.to_floor(moved) - self.to_floor(self.probe), axis=1).max())
        verdict = "unsure" if cc < self.min_cc else ("moved" if shift > self.tol / 2 else "steady")
        return ScreenCheck(shift, float(cc), time.perf_counter() - t0, verdict)
