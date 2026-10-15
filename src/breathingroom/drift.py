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
