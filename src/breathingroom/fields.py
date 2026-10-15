"""Crowd fields on a metric floor grid: density, velocity, velocity variance and pressure.

Inputs per frame are a density map (people per map pixel, from DM-Count) and an optical
flow field over the same image. Every map pixel is carried to metres through the camera's
head-plane homography, so each person-mass lands where that person's head is, and every
flow vector becomes a velocity in m/s.

The fields follow Helbing, Johansson and Al-Abideen (2007), "Dynamics of crowd disasters:
an empirical study", Phys. Rev. E 75, 046109:

    rho(r)   = sum_j f(r - r_j)                               people / m^2
    V(r)     = sum_j f(r - r_j) v_j / rho(r)                  m / s
    Var(r)   = sum_j f(r - r_j) |v_j - V(r)|^2 / rho(r)       m^2 / s^2
    P(r)     = rho(r) * Var(r)                                "crowd pressure", 1 / s^2

with the Gaussian kernel f(d) = exp(-|d|^2 / R^2) / (pi R^2), R = 0.7 m. On a grid of cell
area a, sum_j f(r - r_j) is a Gaussian blur of the per-cell people mass divided by a, so the
whole computation is three bincounts and three cv.GaussianBlur calls per frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .calib import FloorGrid, apply_h

KERNEL_R = 0.7  # metres (Helbing et al. 2007)


@dataclass
class FieldFrame:
    t: float
    rho: np.ndarray  # (rows, cols) people / m^2
    V: np.ndarray  # (rows, cols, 2) m / s
    var: np.ndarray  # (rows, cols) m^2 / s^2
    mass: np.ndarray  # (rows, cols) people per cell (unsmoothed)

    @property
    def pressure(self) -> np.ndarray:
        return self.rho * self.var


class FieldBuilder:
    """Turns people-mass (a density map or detected heads) plus a flow field into FieldFrames
    on a fixed floor grid."""

    def __init__(self, grid: FloorGrid, H_img2plane: np.ndarray, image_size: tuple[int, int],
                 map_shape: tuple[int, int] | None = None, kernel_r: float = KERNEL_R):
        self.grid = grid
        self.H = H_img2plane
        self.image_size = image_size  # (w, h) of the image the homography refers to
        rows, cols = grid.shape
        self.n = rows * cols
        self.sigma_cells = (kernel_r / np.sqrt(2)) / grid.cell
        self.area = grid.cell**2
        self.map_shape = map_shape
        if map_shape is not None:
            mh, mw = map_shape
            xs = (np.arange(mw) + 0.5) * image_size[0] / mw
            ys = (np.arange(mh) + 0.5) * image_size[1] / mh
            X, Y = np.meshgrid(xs, ys)
            self.map_pix = np.column_stack([X.ravel(), Y.ravel()])

    def velocities(self, pix: np.ndarray, flow: np.ndarray, dt: float, dense: bool = False) -> np.ndarray:
        """Velocity (m/s) of image points `pix` (image px) from a flow field of any resolution.

        `dense=True` means `pix` are the density-map pixel centres: the flow is area-averaged
        onto the map grid. Otherwise it is sampled bilinearly at each point."""
        fh, fw = flow.shape[:2]
        sx, sy = fw / self.image_size[0], fh / self.image_size[1]
        if dense:
            mh, mw = self.map_shape
            f = cv2.resize(flow, (mw, mh), interpolation=cv2.INTER_AREA).reshape(-1, 2)
        else:
            mx = (pix[:, 0] * sx - 0.5).astype(np.float32).reshape(-1, 1)
            my = (pix[:, 1] * sy - 0.5).astype(np.float32).reshape(-1, 1)
            f = cv2.remap(flow, mx, my, cv2.INTER_LINEAR).reshape(-1, 2)
        f = f / np.array([sx, sy])  # flow in image px
        w0 = apply_h(self.H, pix)
        w1 = apply_h(self.H, pix + f)
        return (w1 - w0) / dt

    def _blur(self, a: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(a.astype(np.float32), (0, 0), self.sigma_cells, borderType=cv2.BORDER_CONSTANT)

    def build(self, t: float, pix: np.ndarray, mass: np.ndarray, vel: np.ndarray) -> FieldFrame:
        """Fields from people-mass at image points: map pixel centres with density-map values,
        or detected heads with mass 1 each."""
        rows, cols = self.grid.shape
        world = apply_h(self.H, pix)
        cell = self.grid.index(world)
        ok = (cell >= 0) & np.isfinite(vel[:, 0]) & (mass > 0)
        c, w, v = cell[ok], mass[ok].astype(np.float64), vel[ok]
        m = np.bincount(c, weights=w, minlength=self.n).reshape(rows, cols)
        mvx = np.bincount(c, weights=w * v[:, 0], minlength=self.n).reshape(rows, cols)
        mvy = np.bincount(c, weights=w * v[:, 1], minlength=self.n).reshape(rows, cols)
        mvv = np.bincount(c, weights=w * (v**2).sum(1), minlength=self.n).reshape(rows, cols)
        bm, bx, by, bvv = (self._blur(a) for a in (m, mvx, mvy, mvv))
        rho = bm / self.area
        safe = np.maximum(bm, 1e-6)
        V = np.dstack([bx / safe, by / safe])
        var = np.maximum(bvv / safe - (V**2).sum(2), 0.0)
        low = bm < 1e-3
        V[low] = 0
        var[low] = 0
        return FieldFrame(t, rho.astype(np.float32), V.astype(np.float32), var.astype(np.float32), m.astype(np.float32))

    def build_from_map(self, t: float, density: np.ndarray, flow: np.ndarray, dt: float) -> FieldFrame:
        vel = self.velocities(self.map_pix, flow, dt, dense=True)
        return self.build(t, self.map_pix, density.reshape(-1), vel)

    def build_from_points(self, t: float, heads: np.ndarray, flow: np.ndarray, dt: float) -> FieldFrame:
        heads = np.asarray(heads, np.float64).reshape(-1, 2)
        vel = self.velocities(heads, flow, dt) if len(heads) else np.zeros((0, 2))
        return self.build(t, heads, np.ones(len(heads)), vel)


class SpaceTimeWindow:
    """Sliding time window of field frames; Helbing's averages are over space and time."""

    def __init__(self, seconds: float = 2.0):
        self.seconds = seconds
        self.frames: deque[FieldFrame] = deque()

    def push(self, fr: FieldFrame) -> None:
        self.frames.append(fr)
        while self.frames and fr.t - self.frames[0].t > self.seconds:
            self.frames.popleft()

    def mean(self, attr: str) -> np.ndarray:
        return np.mean([getattr(f, attr) for f in self.frames], axis=0)

    def pressure(self) -> np.ndarray:
        return np.mean([f.pressure for f in self.frames], axis=0)


def sample(grid: FloorGrid, field: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a grid field at floor points (m)."""
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    mx = ((xy[:, 0] - grid.x0) / grid.cell - 0.5).astype(np.float32)
    my = ((xy[:, 1] - grid.y0) / grid.cell - 0.5).astype(np.float32)
    out = cv2.remap(field.astype(np.float32), mx.reshape(-1, 1), my.reshape(-1, 1), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.ravel()
