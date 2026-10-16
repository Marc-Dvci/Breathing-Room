"""The Jülich 2018crowdqueue overview camera: lens model, cap detector, calibration file.

The calibration is produced by eval/fzj_calibrate.py and stored in
eval/calib/fzj_2018crowdqueue.json."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def detect_caps(bgr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Centroids (px) of the orange caps every participant wears. `scale` is the image's size
    relative to the original video (0.5 for a half-size frame); sizes are scaled with it."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 140, 120), (20, 255, 255))
    if scale >= 0.75:  # at half size an opening erodes the smallest caps; the area test removes specks
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = (area > 150 * scale**2) & (area < 6000 * scale**2)
    return cent[1:][keep]


def scaled_k(K: np.ndarray, scale: float) -> np.ndarray:
    """A camera matrix for the image resized by `scale` with area averaging (pixel centres move)."""
    S = np.array([[scale, 0, 0.5 * scale - 0.5], [0, scale, 0.5 * scale - 0.5], [0, 0, 1]])
    return S @ K


class Lens:
    """Pinhole + radial distortion; `undistort` maps original pixels into the undistorted view."""

    def __init__(self, x):
        self.x = np.asarray(x, float)
        self.K = np.array([[x[0], 0, x[1]], [0, x[0], x[2]], [0, 0, 1]])
        self.D = np.array([x[3], x[4], 0, 0, x[5]])
        self.Kn = np.array([[x[6], 0, x[7]], [0, x[6], x[8]], [0, 0, 1]])

    def undistort(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.D, P=self.Kn).reshape(-1, 2)

    def distort(self, pts):
        """Undistorted pixels back to original pixels."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        n = (np.c_[p, np.ones(len(p))] @ np.linalg.inv(self.Kn).T)[:, :2]
        out, _ = cv2.projectPoints(np.c_[n, np.ones(len(n))].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D)
        return out.reshape(-1, 2)




def load(path: str | Path) -> tuple[Lens, dict]:
    d = json.loads(Path(path).read_text())
    return Lens(d["lens"]), d["runs"]
