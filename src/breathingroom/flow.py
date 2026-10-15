"""Dense optical flow with OpenCV DIS (Kroeger et al., ECCV 2016)."""

from __future__ import annotations

import cv2
import numpy as np

PRESETS = {
    "ultrafast": cv2.DISOpticalFlow_PRESET_ULTRAFAST,
    "fast": cv2.DISOpticalFlow_PRESET_FAST,
    "medium": cv2.DISOpticalFlow_PRESET_MEDIUM,
}


class Flow:
    """DIS flow between two frames at a working width, returned at that working resolution."""

    def __init__(self, preset: str = "medium", width: int = 960):
        self.dis = cv2.DISOpticalFlow_create(PRESETS[preset])
        self.width = width

    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.width / w
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.width, int(round(h * s))), interpolation=cv2.INTER_AREA)

    def __call__(self, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        return self.dis.calc(g0, g1, None)
