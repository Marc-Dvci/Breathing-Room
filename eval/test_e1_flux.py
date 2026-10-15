"""Outflow at the E1 gate line from video, against the trajectories.

The console measures E1's outflow by counting detected heads that cross a line 0.4 m
before the entrance gate (heads matched between samples 0.2 s apart), averaged over 5 s.
The reference is the number of tracked heads crossing the same line, per second, from the
published trajectories of run 150. Both are compared per second (5 s averages) and in total.

    python eval/test_e1_flux.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.datasets import fzj  # noqa: E402
from breathingroom.runtime import CameraZone  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
OUT = ROOT / "eval" / "results"
RUN = "150_q_56_h0"


def main() -> None:
    cam = CameraZone(DATA / "fzj" / "videos" / f"{RUN}.mp4", ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json", RUN[:3])
    seconds = (cam.n_frames - 4) // 25
    video, counted = [], 0
    for _ in range(seconds):
        cam.tick(1.0)
        video.append(cam.state.outflow_pps)
        counted += sum(list(cam.flux_window)[-5:])
    T = fzj.load_trajectories(DATA / "fzj" / "traj" / f"{RUN}.txt")
    y_line, (xa, xb) = cam.gate_line.a[1], (cam.gate_line.a[0], cam.gate_line.b[0])
    cross_t = []
    order = np.lexsort((T.frames, T.ids))
    ids, fr, xy = T.ids[order], T.frames[order], T.xy[order]
    for pid in np.unique(ids):
        m = ids == pid
        y, x, f = xy[m, 1], xy[m, 0], fr[m]
        k = np.flatnonzero((y[:-1] >= y_line) & (y[1:] < y_line) & (x[1:] >= xa) & (x[1:] <= xb))
        if len(k):
            cross_t.append(f[k[0] + 1] / T.fps)
    cross_t = np.array(cross_t)
    per_s = np.array([((cross_t >= s) & (cross_t < s + 1)).sum() for s in range(seconds)], float)
    ref = np.array([per_s[max(0, s - 4):s + 1].mean() for s in range(seconds)])  # same 5 s average
    v = np.array(video)
    ok = np.arange(seconds) >= 5  # after the first 5 s average has filled
    out = {"run": RUN, "seconds": seconds, "line_y_m": y_line,
           "total_crossings_reference": int(len(cross_t)),
           "total_counted_from_video": int(counted),
           "mean_pps_reference": round(float(ref[ok].mean()), 3), "mean_pps_video": round(float(v[ok].mean()), 3),
           "pearson_r": round(float(np.corrcoef(ref[ok], v[ok])[0, 1]), 3),
           "mae_pps": round(float(np.abs(ref[ok] - v[ok]).mean()), 3),
           "video": [round(x, 3) for x in v], "reference": [round(x, 3) for x in ref]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_e1_flux.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: x for k, x in out.items() if k not in ("video", "reference")}, indent=1))


if __name__ == "__main__":
    main()
