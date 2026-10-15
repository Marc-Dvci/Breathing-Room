"""Calibrate the Jülich 2018crowdqueue overview camera for every rated run.

The archive gives head trajectories in metres but no camera parameters. Two public pieces
make the camera recoverable:

1. Alia et al. (VCNN4PuDe) publish run 050 as an undistorted video together with the same
   trajectories in pixels. Matching those to the metric tracks by person id gives the
   head-plane homography of the undistorted view (and shows the frame numbers coincide).
2. SIFT matches between the original and the undistorted frames of run 050 give the GoPro
   lens model (pinhole + radial k1, k2, k3), fitted with robust least squares.

Each run then gets its own head-plane homography, refined from the run-050 one by matching
the orange caps (HSV segmentation) to the projected heads one-to-one, because the camera
can move between sessions.

    python eval/fzj_calibrate.py            # all rated runs
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.datasets import fzj  # noqa: E402
from breathingroom.datasets.fzj_camera import Lens, detect_caps  # noqa: E402,F401

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
OUT = ROOT / "eval" / "calib"
RATED = ["030", "040", "050", "060", "110", "120", "150", "160", "230", "240", "250", "260", "270", "280"]


def fit_lens() -> tuple[Lens, dict]:
    co = cv2.VideoCapture(str(DATA / "fzj/videos/050_c_45_h0.mp4"))
    cu = cv2.VideoCapture(str(DATA / "alia/50-undistorted.mp4"))
    sift = cv2.SIFT_create(8000)
    P, Q = [], []
    for f in (100, 400, 700):
        co.set(cv2.CAP_PROP_POS_FRAMES, f)
        cu.set(cv2.CAP_PROP_POS_FRAMES, f)
        a, b = co.read()[1], cu.read()[1]
        ka, da = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
        kb, db = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
        good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
        P += [ka[m.queryIdx].pt for m in good]
        Q += [kb[m.trainIdx].pt for m in good]
    P, Q = np.array(P), np.array(Q)
    x = np.array([900, 960, 720, -0.3, 0.1, 0, 900, 960, 720.0])
    res = lambda x, P, Q: (Lens(x).undistort(P) - Q).ravel()  # noqa: E731
    x = least_squares(res, x, args=(P, Q), loss="soft_l1", f_scale=3).x
    ok = np.linalg.norm(Lens(x).undistort(P) - Q, axis=1) < 5
    x = least_squares(res, x, args=(P[ok], Q[ok]), loss="soft_l1", f_scale=2).x
    e = np.linalg.norm(Lens(x).undistort(P) - Q, axis=1)
    return Lens(x), {"matches": int(len(P)), "median_px": float(np.median(e)), "inlier_3px": float((e < 3).mean())}


def homography_050() -> tuple[np.ndarray, dict]:
    """World (m) -> undistorted pixels for run 050, from Alia et al.'s pixel tracks."""
    A = np.loadtxt(DATA / "alia/trajectory-ROI.csv", delimiter=",")
    T = fzj.load_trajectories(DATA / "fzj/traj/050_c_45_h0.txt")
    W = {(int(i), int(f)): p for i, f, p in zip(T.ids, T.frames, T.xy)}
    src, dst = [], []
    for i, f, x, y in A:
        if (int(i), int(f)) in W:
            src.append(W[int(i), int(f)])
            dst.append((x, y))
    src, dst = np.array(src), np.array(dst)
    H, m = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
    e = np.linalg.norm(cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2) - dst, axis=1)
    return H, {"pairs": int(len(src)), "median_px": float(np.median(e)), "inliers": float(m.mean())}


def _assign(H, samples, gate):
    src, dst, err = [], [], []
    for xy, caps in samples:
        proj = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), H).reshape(-1, 2)
        d = np.linalg.norm(proj[:, None] - caps[None], axis=2)
        r, c = linear_sum_assignment(d)
        ok = d[r, c] < gate
        src.append(xy[r[ok]])
        dst.append(caps[c[ok]])
        err.append(d[r, c][ok])
    return np.concatenate(src), np.concatenate(dst), np.concatenate(err)


def _splat(pts, shape, cell, origin, sigma=1.5):
    img = np.zeros(shape, np.float32)
    q = np.floor((pts - origin) / cell).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 0] < shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < shape[0])
    np.add.at(img, (q[ok, 1], q[ok, 0]), 1.0)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def rigid_init(samples, H0, cell: float = 0.05, size: int = 512) -> np.ndarray:
    """Each corridor set-up was surveyed in its own world frame. Find the rigid transform
    (8 axis orientations, translation by FFT correlation over all frames) that carries this
    run's frame into the run-050 frame, and return the resulting world->undistorted H."""
    Hinv = np.linalg.inv(H0)
    caps_w = [cv2.perspectiveTransform(c.reshape(-1, 1, 2), Hinv).reshape(-1, 2) for _, c in samples]
    allc = np.concatenate(caps_w)
    origin = np.median(allc, 0) - size * cell / 2
    cap_f = [np.fft.rfft2(_splat(c, (size, size), cell, origin)) for c in caps_w]
    best = (-1.0, None)
    for k in range(4):
        for flip in (1, -1):
            a = k * np.pi / 2
            Rf = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([1, flip])
            allw = np.concatenate([xy for xy, _ in samples]) @ Rf.T
            shift0 = np.median(allc, 0) - np.median(allw, 0)
            acc = 0
            for (xy, _), cf in zip(samples, cap_f):
                acc = acc + cf * np.conj(np.fft.rfft2(_splat(xy @ Rf.T + shift0, (size, size), cell, origin)))
            corr = np.fft.irfft2(acc, s=(size, size))
            i = int(corr.argmax())
            if corr.flat[i] > best[0]:
                dy, dx = np.unravel_index(i, corr.shape)
                d = np.array([dx if dx < size / 2 else dx - size, dy if dy < size / 2 else dy - size]) * cell
                best = (float(corr.flat[i]), (Rf, shift0 + d))
    Rf, t = best[1]
    G = np.array([[Rf[0, 0], Rf[0, 1], t[0]], [Rf[1, 0], Rf[1, 1], t[1]], [0, 0, 1.0]])
    return H0 @ G


def refine_run(run: str, lens: Lens, H0: np.ndarray, step: int = 25) -> dict:
    tf = next((DATA / "fzj/traj").glob(f"{run}_*.txt"))
    T = fzj.load_trajectories(tf)
    cap = cv2.VideoCapture(str(DATA / "fzj/videos" / f"{tf.stem}.mp4"))
    samples = []
    for fr in range(0, T.n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if not ok:
            break
        _, xy = T.at(fr)
        caps = detect_caps(img)
        if len(xy) >= 3 and len(caps) >= 3:
            samples.append((xy, lens.undistort(caps)))
    _, _, e0 = _assign(H0, samples, np.inf)
    H = H0.copy() if np.median(e0) < 20 else rigid_init(samples, H0)
    for gate in (120, 80, 50, 30, 20, 15):
        src, dst, _ = _assign(H, samples, gate)
        Hn, _ = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0)
        if Hn is not None:
            H = Hn
    src, dst, e = _assign(H, samples, 15)
    return {
        "run": tf.stem,
        "H_world_to_undist": H.tolist(),
        "pairs": int(len(src)),
        "median_px": float(np.median(e)),
        "p90_px": float(np.percentile(e, 90)),
        "median_px_with_050_homography": float(np.median(e0)),
        "frames": len(samples),
    }


def main(runs=RATED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lens, lens_q = fit_lens()
    H0, h_q = homography_050()
    print("lens:", lens_q)
    print("run 050 homography:", h_q)
    summary = {"lens": lens.x.tolist(), "lens_fit": lens_q, "homography_050_fit": h_q, "runs": {}}
    for r in runs:
        res = refine_run(r, lens, H0)
        summary["runs"][r] = res
        print(f"{res['run']}: {res['pairs']} pairs, median {res['median_px']:.1f} px "
              f"(p90 {res['p90_px']:.1f}; run-050 homography alone {res['median_px_with_050_homography']:.1f})")
    (OUT / "fzj_2018crowdqueue.json").write_text(json.dumps(summary, indent=2))


def load() -> tuple[Lens, dict]:
    from breathingroom.datasets.fzj_camera import load as _load

    return _load(OUT / "fzj_2018crowdqueue.json")


if __name__ == "__main__":
    main(sys.argv[1:] or RATED)
