"""Test E: the camera drift watchdog (ALIKED + LightGlue, OpenCV 5 features module).

Live frames come from the Jülich run 150 recording, at 12 moments across the run, so the
crowd under the camera differs from the calibration frame (frame 0). Each frame is checked
as recorded (false alarms) and after synthetic knocks of known size: a roll of 0.5 to 4
degrees about the image centre plus a proportional shift. The true floor displacement of
the calibration points is known for each knock, so detection is scored against it: the
watchdog should flag knocks that move the floor more than its 0.25 m tolerance and pass
the rest.

    python eval/test_e_drift.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.drift import DriftWatch  # noqa: E402
from breathingroom.runtime import CameraZone  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
OUT = ROOT / "eval" / "results"
ROLLS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def knock(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    M[:, 2] += (13 * deg, -8 * deg)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), np.vstack([M, [0, 0, 1]])


def main() -> None:
    cam = CameraZone(DATA / "fzj" / "videos" / "150_q_56_h0.mp4", ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json", "150")
    corners = np.array([[x, y] for x in cam.zone_rect[:2] for y in cam.zone_rect[2:]], float)
    probe = cam.lens.distort(cv2.perspectiveTransform(corners.reshape(-1, 1, 2), cam.H_w2u).reshape(-1, 2))
    dw = DriftWatch(cam.ref_frame, cam.to_floor, probe)
    rows = []
    for f in np.linspace(50, cam.n_frames - 10, 12).astype(int):
        img = cam.reader.read(int(f))
        for deg in ROLLS:
            live, M = knock(img, deg)
            moved = cv2.perspectiveTransform(probe.reshape(-1, 1, 2), M).reshape(-1, 2)
            true_shift = float(np.linalg.norm(cam.to_floor(moved) - cam.to_floor(probe), axis=1).max())
            r = dw.check(live)
            rows.append({"frame": int(f), "roll_deg": deg, "true_floor_shift_m": round(true_shift, 3),
                         "measured_floor_shift_m": round(r.floor_shift_m, 3), "trusted": r.trusted,
                         "inliers": r.inliers, "matches": r.matches, "ms": round(r.seconds * 1000)})
        print(f"frame {f}: " + " ".join(f"{x['roll_deg']}:{x['measured_floor_shift_m']:.2f}/{x['true_floor_shift_m']:.2f}"
                                         for x in rows[-len(ROLLS):]), flush=True)
    tol = dw.tol
    should = [r for r in rows if r["true_floor_shift_m"] > tol]
    shouldnt = [r for r in rows if r["true_floor_shift_m"] <= tol]
    err = [abs(r["measured_floor_shift_m"] - r["true_floor_shift_m"]) for r in rows]
    by_roll = {str(d): {"true_shift_m": round(float(np.median([r["true_floor_shift_m"] for r in rows if r["roll_deg"] == d])), 3),
                        "flagged": sum(not r["trusted"] for r in rows if r["roll_deg"] == d),
                        "of": sum(1 for r in rows if r["roll_deg"] == d)} for d in ROLLS}
    out = {"tolerance_m": tol, "checks": len(rows),
           "detected": sum(not r["trusted"] for r in should), "should_detect": len(should),
           "false_alarms": sum(not r["trusted"] for r in shouldnt), "should_pass": len(shouldnt),
           "median_abs_error_m": round(float(np.median(err)), 3), "p90_abs_error_m": round(float(np.percentile(err, 90)), 3),
           "median_ms": float(np.median([r["ms"] for r in rows])), "by_roll": by_roll, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_e_drift.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
