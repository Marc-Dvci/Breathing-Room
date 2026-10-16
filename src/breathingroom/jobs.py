"""The two heavy model calls, as functions a fresh child process can run (BR_ISOLATE_MODELS=1).

This module imports only OpenCV, NumPy and the two model wrappers, so a child process starts
without loading the venue twin, the agent or SciPy."""

from __future__ import annotations

import cv2


def drift_job(ref, live, lens_x, H_u2w, probe) -> dict:
    """ALIKED + LightGlue check of `live` against `ref`."""
    from .datasets.fzj_camera import Lens
    from .drift import DriftWatch

    lens = Lens(lens_x)

    def to_floor(px):
        return cv2.perspectiveTransform(lens.undistort(px).reshape(-1, 1, 2), H_u2w).reshape(-1, 2)

    return DriftWatch(ref, to_floor, probe).check(live).to_json()


def count_job(crop, mask) -> tuple[float, float]:
    """DM-Count over a zone crop."""
    from .density import DensityModel

    r = DensityModel("sha", max_side=max(crop.shape[:2]))(crop, roi_mask=mask)
    return r.count, r.seconds
