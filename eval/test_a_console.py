"""Test A through the console's own code: the pushing index that `CameraZone` computes live,
scored against the trained raters' labels on the six calibrated Jülich runs.

eval/test_a_pushing.py builds the same fields with its own loop (and with DM-Count as a second
source). This script runs `CameraZone.tick` itself, at the working scale the console uses, so
the AUROC below is the AUROC of the deployed pipeline. Rows, windows and labels are the same as
in test A: one rated person at one second, positive = rating 3 or 4, the index averaged over
+-1 s at the person's position from the published trajectories.

    python eval/test_a_console.py [--work-scale 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.datasets import fzj  # noqa: E402
from breathingroom.fields import sample  # noqa: E402
from breathingroom.runtime import CameraZone  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data")) / "fzj"
CALIB = ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json"
OUT = ROOT / "eval" / "results"
RUNS = ["030", "040", "050", "060", "150", "160"]
BANDS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 9)]
HALF_WINDOW = 25


def run_rows(run: str, work_scale: float) -> tuple[list[dict], dict]:
    tf = next((DATA / "traj").glob(f"{run}_*.txt"))
    T = fzj.load_trajectories(tf)
    rat = fzj.load_ratings(next((DATA / "ratings" / "1) Rating data").glob(f"{run} - jointversion.txt")), T.n_frames)
    lo, hi = T.xy.min(0) - 1.0, T.xy.max(0) + 1.0
    cz = CameraZone(DATA / "videos" / f"{tf.stem}.mp4", CALIB, run, work_scale=work_scale,
                    grid=(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
    fields, t0 = {}, time.perf_counter()
    while cz.frame_idx + 2 < min(cz.n_frames - 1, T.n_frames):
        before = cz.frame_idx
        cz.tick(0.2)  # one field sample
        if cz.frame_idx < before:
            break
        f, fr = cz.last_field
        fields[f] = (fr.rho, fr.pressure)
    secs = time.perf_counter() - t0
    pos = {(int(i), int(k)): p for i, k, p in zip(T.ids, T.frames, T.xy)}
    rows = []
    for pid, labels in rat.items():
        for f in range(25, T.n_frames - 25, 25):
            if f >= len(labels) or labels[f] == 0:
                continue
            vid = [[sample(cz.grid, g, pos[pid, k][None])[0] for g in fields[k]]
                   for k in range(f - HALF_WINDOW, f + HALF_WINDOW + 1, 5) if (pid, k) in pos and k in fields]
            if len(vid) < 5:
                continue
            v = np.array(vid)
            rows.append({"run": run, "pid": pid, "frame": f, "rating": int(labels[f]),
                         "rho": float(v[:, 0].mean()), "P": float(v[:, 1].mean())})
    return rows, {"samples": len(fields), "seconds": round(secs, 1), "ms_per_camera_second": round(secs / len(fields) * 5000)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-scale", type=float, default=0.5)
    args = ap.parse_args()
    rows, meta = [], {}
    for r in RUNS:
        rr, m = run_rows(r, args.work_scale)
        rows += rr
        meta[r] = m
        print(f"{r}: {len(rr)} rows, {m}", flush=True)
    y = np.array([r["rating"] >= 3 for r in rows], int)
    P = np.array([r["P"] for r in rows])
    rho = np.array([r["rho"] for r in rows])
    # bands on the reference density from test A, matched by (run, pid, frame)
    ref = {}
    ref_csv = OUT / "test_a_rows.csv"
    if ref_csv.exists():
        import csv

        with open(ref_csv) as fh:
            for x in csv.DictReader(fh):
                ref[(x["run"].zfill(3), int(x["pid"]), int(x["frame"]))] = float(x["rho_traj"])
    band_rho = np.array([ref.get((r["run"], r["pid"], r["frame"]), np.nan) for r in rows])
    bands = {}
    for a, b in BANDS:
        m = (band_rho >= a) & (band_rho < b)
        bands[f"{a}-{b}"] = {"n": int(m.sum()), "auc": float(roc_auc_score(y[m], P[m])) if 0 < y[m].mean() < 1 else None}
    out = {"work_scale": args.work_scale, "rows": len(rows), "pushing_share": float(y.mean()),
           "auc_pressure": float(roc_auc_score(y, P)), "auc_density": float(roc_auc_score(y, rho)),
           "auc_pressure_by_density_band": bands, "runs": meta}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"test_a_console_ws{args.work_scale:g}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
