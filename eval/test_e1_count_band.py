"""The agreement band of E1's two counters, at the console's settings.

The console's `cross_check_count` counts the zone twice: heads from the head detector, and
DM-Count (OpenCV 5 DNN engine) on the zone's bounding box at half the undistorted resolution.
Both come from `CameraZone.prepare` and `CameraZone.count_crop`, the console's own code.
On this overhead view DM-Count undercounts, so agreement is a band on the ratio
DM-Count / heads, measured here over the six calibrated Jülich runs: frames every 0.8 s
with at least 3 heads in the zone. The band is the 2.5th to 97.5th percentile.

    python eval/test_e1_count_band.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.density import DensityModel  # noqa: E402
from breathingroom.runtime import COUNT_SCALE, CameraZone  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data")) / "fzj"
CALIB = ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json"
OUT = ROOT / "eval" / "results"
RUNS = ["030", "040", "050", "060", "150", "160"]
EVERY = 20  # frames (0.8 s)


def main() -> None:
    dm = DensityModel("sha", "new", max_side=4096)
    rect = CameraZone.zone_rect_default
    rows = []
    for run in RUNS:
        video = next((DATA / "videos").glob(f"{run}_*.mp4"))
        cz = CameraZone(video, CALIB, run)
        cap = cv2.VideoCapture(str(video))
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f % EVERY == 0:
                und, heads = cz.prepare(img)
                w = cv2.perspectiveTransform(heads.reshape(-1, 1, 2), cz.H_u2w).reshape(-1, 2) if len(heads) else np.zeros((0, 2))
                x0, x1, y0, y1 = rect
                n = int(((w[:, 0] >= x0) & (w[:, 0] < x1) & (w[:, 1] >= y0) & (w[:, 1] < y1)).sum())
                if n >= 3:
                    crop, mask = cz.count_crop(und)
                    t0 = time.perf_counter()
                    r = dm(crop, roi_mask=mask)
                    rows.append({"run": run, "frame": f, "heads": n, "dmcount": round(r.count, 2),
                                 "ratio": round(r.count / n, 3), "ms": round((time.perf_counter() - t0) * 1000)})
            f += 1
        last = [r for r in rows if r["run"] == run]
        print(f"{run}: {len(last)} frames, median ratio {np.median([r['ratio'] for r in last]):.2f}", flush=True)
    ratio = np.array([r["ratio"] for r in rows])
    out = {"frames": len(rows), "count_scale": COUNT_SCALE, "zone_rect_m": list(rect),
           "band": [round(float(np.percentile(ratio, 2.5)), 2), round(float(np.percentile(ratio, 97.5)), 2)],
           "median_ratio": round(float(np.median(ratio)), 3),
           "pearson_r": round(float(np.corrcoef([r["heads"] for r in rows], [r["dmcount"] for r in rows])[0, 1]), 3),
           "median_ms": float(np.median([r["ms"] for r in rows])), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_e1_count_band.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
