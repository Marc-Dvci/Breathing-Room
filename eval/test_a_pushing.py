"""Test A: does the pressure index read from video pick out the people raters saw pushing?

Ground truth: per-person, per-second ratings of the Jülich 2018 entrance experiments
(DOI 10.34735/ped.2022.4; 1 falling behind, 2 walking, 3 mild pushing, 4 strong pushing).
A row is one rated person at one second. Positive = rating 3 or 4.

For every row the same two quantities are computed twice:
- from the published trajectories (the reference: exact positions and velocities), and
- from the video alone, through the Breathing Room pipeline (lens undistortion, DM-Count
  density on the OpenCV 5 DNN engine, DIS optical flow, head-plane homography, Helbing
  fields), sampled at the person's position.

Both use Helbing's kernel (R = 0.7 m) and a 2 s space-time window. Nothing is trained: the
index is a physical formula and AUROC needs no threshold.

    python eval/test_a_pushing.py [runs...]
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from breathingroom.calib import FloorGrid  # noqa: E402
from breathingroom.datasets import fzj  # noqa: E402
from breathingroom.density import DensityModel  # noqa: E402
from breathingroom.fields import KERNEL_R, FieldBuilder, sample  # noqa: E402
from breathingroom.flow import Flow  # noqa: E402
import fzj_calibrate as fc  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data")) / "fzj"
OUT = ROOT / "eval" / "results"
CALIBRATED = ["030", "040", "050", "060", "150", "160"]  # median reprojection < 6 px
STEP = 5  # field sample every 5 frames (5 Hz)
FLOW_GAP = 2  # flow between frame f and f + 2 (0.08 s)
DENSITY_EVERY = 10  # DM-Count every 10 frames (0.4 s), held in between
HALF_WINDOW = 25  # +-1 s around the rated second


def trajectory_features(T: fzj.Trajectories, R: float = KERNEL_R) -> dict:
    """{(pid, frame): (rho, var)} at every STEP-th frame from the trajectories."""
    V = T.velocities(12)
    out = {}
    for f in range(0, T.n_frames, STEP):
        lo, hi = np.searchsorted(T.frames, f), np.searchsorted(T.frames, f, "right")
        ids, xy, v = T.ids[lo:hi], T.xy[lo:hi], V[lo:hi]
        ok = ~np.isnan(v[:, 0])
        ids, xy, v = ids[ok], xy[ok], v[ok]
        if len(ids) < 2:
            continue
        d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
        w = np.exp(-d2 / R**2) / (np.pi * R**2)
        rho = w.sum(1)
        Vm = (w @ v) / w.sum(1, keepdims=True)
        var = (w * ((v[None] - Vm[:, None]) ** 2).sum(-1)).sum(1) / w.sum(1)
        for i, p in enumerate(ids):
            out[int(p), f] = (rho[i], var[i])
    return out


def video_fields(run_stem: str, lens: fc.Lens, H_w2u: np.ndarray, grid: FloorGrid, n_frames: int,
                 density_name: str = "sha", engine: str = "new") -> tuple[dict, dict]:
    """Video-only fields at 5 Hz, twice: people-mass from DM-Count (the generic counter) and
    from detected heads (the lab's orange caps), both with the same DIS motion."""
    cap = cv2.VideoCapture(str(DATA / "videos" / f"{run_stem}.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m1, m2 = cv2.initUndistortRectifyMap(lens.K, lens.D, None, lens.Kn, (W, H), cv2.CV_16SC2)
    dm = DensityModel(density_name, engine, max_side=1280)
    flow = Flow("medium", 960)
    H_u2w = np.linalg.inv(H_w2u)
    builder = None
    frames: dict[int, tuple[np.ndarray, ...]] = {}
    stats = {"density_s": [], "flow_s": [], "count_video": [], "count_heads": [], "frame": []}
    heads = {}
    dens = None
    prev = {}
    f = 0
    while f < n_frames + FLOW_GAP:
        ok, img = cap.read()
        if not ok:
            break
        need = (f % STEP == 0) or ((f - FLOW_GAP) % STEP == 0 and f >= FLOW_GAP)
        if need:
            und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
            g = flow.prepare(und)
            if f % STEP == 0:
                prev[f] = g
                heads[f] = lens.undistort(fc.detect_caps(img))
                if f % DENSITY_EVERY == 0:
                    r = dm(und)
                    dens = r.density
                    stats["density_s"].append(r.seconds)
                    stats["count_video"].append(r.count)
                    stats["count_heads"].append(len(heads[f]))
                    stats["frame"].append(f)
                    if builder is None:
                        builder = FieldBuilder(grid, H_u2w, (W, H), dens.shape)
            if (f - FLOW_GAP) % STEP == 0 and (f - FLOW_GAP) in prev and dens is not None:
                f0 = f - FLOW_GAP
                t0 = time.perf_counter()
                fl = flow(prev.pop(f0), g)
                stats["flow_s"].append(time.perf_counter() - t0)
                dt = FLOW_GAP / fzj.FPS
                a = builder.build_from_map(f0 / fzj.FPS, dens, fl, dt)
                b = builder.build_from_points(f0 / fzj.FPS, heads.pop(f0), fl, dt)
                frames[f0] = (a.rho, a.pressure, b.rho, b.pressure)
        f += 1
    return frames, stats


def run_one(run: str, lens: fc.Lens, calib: dict) -> list[dict]:
    tf = next((DATA / "traj").glob(f"{run}_*.txt"))
    T = fzj.load_trajectories(tf)
    rat = fzj.load_ratings(next((DATA / "ratings" / "1) Rating data").glob(f"{run} - jointversion.txt")), T.n_frames)
    H_w2u = np.array(calib[run]["H_world_to_undist"])
    lo, hi = T.xy.min(0) - 1.0, T.xy.max(0) + 1.0
    grid = FloorGrid(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), cell=0.1)
    t0 = time.perf_counter()
    vf, stats = video_fields(tf.stem, lens, H_w2u, grid, T.n_frames)
    print(f"  {tf.stem}: video fields in {time.perf_counter() - t0:.0f} s, "
          f"DM-Count {np.mean(stats['density_s']):.2f} s/frame, DIS {np.mean(stats['flow_s']) * 1000:.0f} ms/pair")
    tr = trajectory_features(T)
    pos = {(int(i), int(fr)): p for i, fr, p in zip(T.ids, T.frames, T.xy)}
    rows = []
    for pid, labels in rat.items():
        for f in range(25, T.n_frames - 25, 25):
            if f >= len(labels) or labels[f] == 0:
                continue
            win = range(f - HALF_WINDOW, f + HALF_WINDOW + 1, STEP)
            tr_v = [tr[pid, k] for k in win if (pid, k) in tr]
            vid = []
            for k in win:
                if (pid, k) in pos and k in vf:
                    xy = pos[pid, k][None]
                    vid.append([sample(grid, g, xy)[0] for g in vf[k]])
            if len(tr_v) < 5 or len(vid) < 5:
                continue
            tr_v, vid = np.array(tr_v), np.array(vid)
            rows.append({
                "run": run, "pid": pid, "frame": f, "rating": int(labels[f]),
                "rho_traj": float(tr_v[:, 0].mean()), "P_traj": float((tr_v[:, 0] * tr_v[:, 1]).mean()),
                "rho_video_dm": float(vid[:, 0].mean()), "P_video_dm": float(vid[:, 1].mean()),
                "rho_video_heads": float(vid[:, 2].mean()), "P_video_heads": float(vid[:, 3].mean()),
            })
    counts = [(fr, c, h, len(T.at(fr)[0]))
              for fr, c, h in zip(stats["frame"], stats["count_video"], stats["count_heads"])]
    return rows, {"density_s": float(np.mean(stats["density_s"])), "flow_ms": float(np.mean(stats["flow_s"]) * 1000),
                  "counts": counts}


def main(runs=CALIBRATED) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lens, calib = fc.load()
    all_rows, meta = [], {}
    for r in runs:
        rows, m = run_one(r, lens, calib)
        all_rows += rows
        meta[r] = m
        with open(OUT / "test_a_rows.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            wr.writeheader()
            wr.writerows(all_rows)
        (OUT / "test_a_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(all_rows)} rows written to {OUT / 'test_a_rows.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:] or CALIBRATED)
