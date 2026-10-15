"""Camera geometry: the floor homography, the camera pose behind it, and the head plane.

People are measured where their heads are. A head seen by an oblique camera sits about
1.6 m above the floor, so mapping it through the floor homography places the person
metres away from where they stand. This module recovers the camera from the floor
homography (focal length, rotation, height) and builds the homography of the plane the
heads move in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HEAD_HEIGHT_M = 1.60  # height of an adult head centre above the floor


def _as_pts(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1, 2)


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    """Map Nx2 points through a 3x3 homography."""
    p = _as_pts(pts)
    if len(p) == 0:
        return p
    return cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)


@dataclass
class FloorFit:
    """Image-to-floor homography fitted from point correspondences."""

    H: np.ndarray  # image px -> floor metres
    inliers: np.ndarray  # bool mask over the input correspondences
    rms_floor_m: float  # residual on the floor, metres
    rms_image_px: float  # residual back in the image, pixels


def fit_floor(image_pts, floor_pts, threshold_m: float = 0.05) -> FloorFit:
    """Fit the floor homography with USAC MAGSAC++ and report residuals in both planes."""
    img = _as_pts(image_pts)
    flr = _as_pts(floor_pts)
    if len(img) < 4 or len(img) != len(flr):
        raise ValueError("need at least four matching image/floor points")
    H, mask, method = None, None, 0
    # USAC MAGSAC++ first; it can return nothing on a few noise-free points, so fall back to RANSAC
    for method in ((cv2.USAC_MAGSAC, cv2.RANSAC) if len(img) > 4 else (0,)):
        H, mask = cv2.findHomography(img, flr, method, threshold_m, maxIters=10000, confidence=0.9999)
        if H is not None:
            break
    if H is None:
        raise ValueError("homography fit failed; check that the points are not collinear")
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(img), bool)
    if inl.sum() >= 4 and method:
        # Final least-squares refit on the inliers.
        H, _ = cv2.findHomography(img[inl], flr[inl], 0)
    err_f = np.linalg.norm(apply_h(H, img[inl]) - flr[inl], axis=1)
    err_i = np.linalg.norm(apply_h(np.linalg.inv(H), flr[inl]) - img[inl], axis=1)
    return FloorFit(H, inl, float(np.sqrt(np.mean(err_f**2))), float(np.sqrt(np.mean(err_i**2))))


def focal_from_floor(H_img2floor: np.ndarray, principal: tuple[float, float]) -> float | None:
    """Focal length (px) from one plane homography, assuming square pixels and a known
    principal point (Zhang's single-view constraints). None when the view is too close to
    fronto-parallel for the constraints to determine it."""
    cx, cy = principal
    G = np.linalg.inv(H_img2floor)  # floor -> image
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = T @ G
    g1, g2 = G[:, 0], G[:, 1]
    est = []
    den1 = g1[2] * g2[2]
    if abs(den1) > 1e-12:
        f2 = -(g1[0] * g2[0] + g1[1] * g2[1]) / den1
        if f2 > 0:
            est.append(f2)
    den2 = g2[2] ** 2 - g1[2] ** 2
    if abs(den2) > 1e-12:
        f2 = (g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / den2
        if f2 > 0:
            est.append(f2)
    if not est:
        return None
    return float(np.sqrt(np.median(est)))


@dataclass
class Camera:
    """A calibrated camera: intrinsics, pose relative to the floor, and the planes it sees."""

    name: str
    image_size: tuple[int, int]  # (width, height)
    H_floor: np.ndarray  # image -> floor (m)
    K: np.ndarray
    R: np.ndarray  # world -> camera rotation
    t: np.ndarray  # world -> camera translation
    up: float  # +1 or -1: sign of the world z axis that points from the floor to the camera
    rms_floor_m: float = 0.0
    rms_image_px: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def centre(self) -> np.ndarray:
        return (-self.R.T @ self.t).ravel()

    @property
    def height_m(self) -> float:
        return float(abs(self.centre[2]))

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and straight down (0 = overhead)."""
        axis_world = self.R.T @ np.array([0, 0, 1.0])
        down = np.array([0, 0, -self.up])
        return float(np.degrees(np.arccos(np.clip(axis_world @ down, -1, 1))))

    def plane_to_image(self, z: float) -> np.ndarray:
        """Homography from (X, Y) on the horizontal plane at height z to image pixels."""
        r1, r2, r3 = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        G = self.K @ np.column_stack([r1, r2, self.up * z * r3 + self.t.ravel()])
        return G / G[2, 2]

    def image_to_plane(self, z: float) -> np.ndarray:
        H = np.linalg.inv(self.plane_to_image(z))
        return H / H[2, 2]

    @property
    def H_head(self) -> np.ndarray:
        return self.image_to_plane(HEAD_HEIGHT_M)

    def head_error_if_floor_used(self, img_pts) -> np.ndarray:
        """How far (m) each head would land from its true floor position if mapped through
        the floor homography instead of the head plane."""
        return np.linalg.norm(apply_h(self.H_floor, img_pts) - apply_h(self.H_head, img_pts), axis=1)

    # -- persistence ---------------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "name": self.name,
            "image_size": list(self.image_size),
            "H_floor": self.H_floor.tolist(),
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.ravel().tolist(),
            "up": self.up,
            "rms_floor_m": self.rms_floor_m,
            "rms_image_px": self.rms_image_px,
            "height_m": self.height_m,
            "tilt_deg": self.tilt_deg,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Camera":
        return cls(
            name=d["name"],
            image_size=tuple(d["image_size"]),
            H_floor=np.array(d["H_floor"], float),
            K=np.array(d["K"], float),
            R=np.array(d["R"], float),
            t=np.array(d["t"], float).reshape(3, 1),
            up=float(d["up"]),
            rms_floor_m=float(d.get("rms_floor_m", 0)),
            rms_image_px=float(d.get("rms_image_px", 0)),
            notes=list(d.get("notes", [])),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Camera":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def pose_from_floor(H_img2floor: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation, translation and up-sign of the camera from the floor homography and K."""
    G = np.linalg.inv(H_img2floor)
    A = np.linalg.inv(K) @ G
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    if t[2] < 0:  # the floor must lie in front of the camera
        r1, r2, t = -r1, -r2, -t
    R0 = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    t = t.reshape(3, 1)
    centre = -R.T @ t
    up = 1.0 if centre[2, 0] >= 0 else -1.0
    return R, t, up


def calibrate(
    name: str,
    image_size: tuple[int, int],
    image_pts,
    floor_pts,
    K: np.ndarray | None = None,
    focal_px: float | None = None,
) -> Camera:
    """Build a Camera from clicked floor points of known position.

    The focal length is recovered from the homography when the view is oblique enough;
    pass `K` or `focal_px` for overhead views where it is not observable."""
    fit = fit_floor(image_pts, floor_pts)
    w, h = image_size
    notes = []
    if K is None:
        f = focal_px or focal_from_floor(fit.H, (w / 2, h / 2))
        if f is None or not (0.2 * w < f < 20 * w):
            raise ValueError(
                "focal length is not observable from this view (close to overhead); pass focal_px or K"
            )
        if focal_px is None:
            notes.append(f"focal length {f:.0f} px recovered from the floor homography")
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    R, t, up = pose_from_floor(fit.H, K)
    cam = Camera(name, (w, h), fit.H, K, R, t, up, fit.rms_floor_m, fit.rms_image_px, notes)
    return cam


@dataclass
class FloorGrid:
    """A metric grid on the floor plan, in cells of `cell` metres."""

    x0: float
    y0: float
    x1: float
    y1: float
    cell: float = 0.5

    @property
    def shape(self) -> tuple[int, int]:  # (rows, cols)
        return (int(np.ceil((self.y1 - self.y0) / self.cell)), int(np.ceil((self.x1 - self.x0) / self.cell)))

    def index(self, xy: np.ndarray) -> np.ndarray:
        """Flat cell index for Nx2 floor points, -1 outside the grid."""
        xy = _as_pts(xy)
        rows, cols = self.shape
        c = np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64)
        r = np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64)
        ok = (c >= 0) & (c < cols) & (r >= 0) & (r < rows)
        out = np.full(len(xy), -1, np.int64)
        out[ok] = r[ok] * cols + c[ok]
        return out

    def centres(self) -> np.ndarray:
        rows, cols = self.shape
        cx = self.x0 + (np.arange(cols) + 0.5) * self.cell
        cy = self.y0 + (np.arange(rows) + 0.5) * self.cell
        X, Y = np.meshgrid(cx, cy)
        return np.column_stack([X.ravel(), Y.ravel()])


@dataclass
class PixelMap:
    """Precomputed lookup from a (possibly downscaled) image to floor-grid cells on one plane.

    Built once per calibration so per-frame aggregation is a single bincount."""

    grid: FloorGrid
    scale: float  # map pixel -> original pixel multiplier
    cell_of_pixel: np.ndarray  # (h, w) int64, -1 = outside grid
    visible_area: np.ndarray  # (n_cells,) m^2 of each cell that the camera sees

    @classmethod
    def build(cls, H_img2plane: np.ndarray, image_size, grid: FloorGrid, scale: float = 1.0,
              mask: np.ndarray | None = None, samples_per_cell: int = 6) -> "PixelMap":
        w, h = image_size
        mw, mh = int(round(w / scale)), int(round(h / scale))
        xs = (np.arange(mw) + 0.5) * scale
        ys = (np.arange(mh) + 0.5) * scale
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        cells = grid.index(apply_h(H_img2plane, pts)).reshape(mh, mw)
        if mask is not None:
            m = cv2.resize(mask.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
            cells[~m] = -1
        # Visible area: sample each cell finely, project into the image, keep in-bounds samples.
        rows, cols = grid.shape
        s = samples_per_cell
        off = (np.arange(s) + 0.5) / s * grid.cell
        ox, oy = np.meshgrid(off, off)
        sub = np.column_stack([ox.ravel(), oy.ravel()])
        corners = np.column_stack([
            np.repeat(grid.x0 + np.arange(cols) * grid.cell, 1)[None, :].repeat(rows, 0).ravel(),
            (grid.y0 + np.arange(rows) * grid.cell)[:, None].repeat(cols, 1).ravel(),
        ])
        allp = (corners[:, None, :] + sub[None, :, :]).reshape(-1, 2)
        img = apply_h(np.linalg.inv(H_img2plane), allp)
        inb = (img[:, 0] >= 0) & (img[:, 0] < w) & (img[:, 1] >= 0) & (img[:, 1] < h)
        if mask is not None:
            ix = np.clip(img[:, 0].astype(int), 0, w - 1)
            iy = np.clip(img[:, 1].astype(int), 0, h - 1)
            mm = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            inb &= mm[iy, ix]
        frac = inb.reshape(len(corners), -1).mean(axis=1)
        return cls(grid, scale, cells, frac * grid.cell**2)

    def accumulate(self, values: np.ndarray) -> np.ndarray:
        """Sum a per-pixel map (same size as the lookup) into floor cells."""
        if values.shape != self.cell_of_pixel.shape:
            values = cv2.resize(values.astype(np.float32), self.cell_of_pixel.shape[::-1], interpolation=cv2.INTER_AREA)
            # keep the total mass when downscaling a density map
            values = values * (values.size and 1.0)
        c = self.cell_of_pixel.ravel()
        ok = c >= 0
        n = self.grid.shape[0] * self.grid.shape[1]
        return np.bincount(c[ok], weights=values.ravel()[ok], minlength=n)
