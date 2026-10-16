"""Crowd counting through the OpenCV 5 DNN engine.

DM-Count density maps (ONNX, exported by tools/export_dmcount.py). The map has 1/8 of the
input resolution and integrates to the number of people; summing it over the floor cells of
a PixelMap gives people per cell, and dividing by the cell's visible area gives people/m^2.

Three trainings are exported: SHB (sparse street scenes), SHA and QNRF (dense crowds).
eval/test_d_counting.py scores them against annotated heads on oblique street CCTV (CroHD).
In the console, DM-Count is the second, independent counter behind the head detector.
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0

ENGINES = {
    "new": cv2.dnn.ENGINE_NEW,
    "classic": cv2.dnn.ENGINE_CLASSIC,
    "ort": cv2.dnn.ENGINE_ORT,
    "auto": cv2.dnn.ENGINE_AUTO,
}
# Measured with bench/bench_stages.py: on AWS Graviton the classic engine runs DM-Count about five
# times faster than the new graph engine, with the same count; on x86 the new engine is faster.
DEFAULT_ENGINE = os.environ.get("BR_DNN_ENGINE") or ("classic" if platform.machine().lower() in ("aarch64", "arm64") else "new")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DensityResult:
    density: np.ndarray  # (h/8, w/8) people per map pixel
    count: float
    seconds: float
    input_size: tuple[int, int]


class DensityModel:
    """One DM-Count network on the OpenCV DNN engine."""

    def __init__(self, name: str = "sha", engine: str = DEFAULT_ENGINE, max_side: int = 1280):
        self.name = name
        self.path = MODELS_DIR / f"dmcount_{name}.onnx"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing; run tools/export_dmcount.py")
        self.engine = engine
        self.net = cv2.dnn.readNetFromONNX(str(self.path), ENGINES[engine])
        self.max_side = max_side

    def _input_size(self, w: int, h: int) -> tuple[int, int]:
        s = min(1.0, self.max_side / max(w, h))
        # VGG pools 4 times: keep both sides multiples of 16
        return max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16)

    def __call__(self, bgr: np.ndarray, roi_mask: np.ndarray | None = None) -> DensityResult:
        h, w = bgr.shape[:2]
        iw, ih = self._input_size(w, h)
        img = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img, 1.0, (iw, ih), swapRB=True)  # NCHW RGB float
        blob = (blob - MEAN.reshape(1, 3, 1, 1)) / STD.reshape(1, 3, 1, 1)
        t0 = time.perf_counter()
        self.net.setInput(blob.astype(np.float32))
        out = self.net.forward()
        dt = time.perf_counter() - t0
        dens = out.reshape(out.shape[-2], out.shape[-1]).astype(np.float32)
        if roi_mask is not None:
            m = cv2.resize(roi_mask.astype(np.uint8), dens.shape[::-1], interpolation=cv2.INTER_NEAREST)
            dens = dens * (m > 0)
        return DensityResult(dens, float(dens.sum()), dt, (iw, ih))


def resample_density(dens: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Resize a density map to `size_wh` keeping its total (people are conserved)."""
    total = float(dens.sum())
    out = cv2.resize(dens, size_wh, interpolation=cv2.INTER_LINEAR)
    s = float(out.sum())
    return out * (total / s) if s > 0 else out
