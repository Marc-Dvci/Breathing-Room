"""Test D: counting real crowds on oblique street CCTV with DM-Count on the OpenCV 5 DNN engine.

Data: CroHD / HT21 training sequences (Sundararaman et al., CVPR 2021; CC BY-NC-SA 3.0),
four dense real scenes filmed from elevated, oblique viewpoints, with every head annotated.
Frames are read straight from the downloaded HT21.zip; the dataset is not redistributed.

For 25 evenly spaced frames per sequence, each DM-Count training (SHA, SHB, QNRF) counts the
whole frame, and the count is compared with the number of annotated heads.

    python eval/test_d_counting.py [--zip path/to/HT21.zip] [--frames 25]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom.density import DensityModel  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
OUT = ROOT / "eval" / "results"
SEQS = ["HT21-01", "HT21-02", "HT21-03", "HT21-04"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(DATA / "ht21" / "HT21.zip"))
    ap.add_argument("--frames", type=int, default=25)
    args = ap.parse_args()
    models = {n: DensityModel(n, "new", max_side=1920) for n in ("sha", "shb", "qnrf")}
    rows = []
    with zipfile.ZipFile(args.zip) as z:
        for seq in SEQS:
            base = f"HT21/train/{seq}/"
            info = z.read(base + "seqinfo.ini").decode()
            length = int(next(x for x in info.splitlines() if x.startswith("seqLength")).split("=")[1])
            gt = np.loadtxt(z.open(base + "gt/gt.txt"), delimiter=",")
            gt = gt[gt[:, 7] == 1]  # class 1: pedestrian head
            for f in np.linspace(1, length, args.frames).astype(int):
                img = cv2.imdecode(np.frombuffer(z.read(f"{base}img1/{f:06d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
                n_gt = int((gt[:, 0] == f).sum())
                r = {"seq": seq, "frame": int(f), "gt": n_gt}
                for name, m in models.items():
                    res = m(img)
                    r[name] = round(res.count, 1)
                    r[f"{name}_ms"] = round(res.seconds * 1000)
                rows.append(r)
                print(json.dumps(r), flush=True)
    summary = {}
    for name in models:
        e = np.array([r[name] - r["gt"] for r in rows], float)
        g = np.array([r["gt"] for r in rows], float)
        per_seq = {}
        for seq in SEQS:
            rs = [r for r in rows if r["seq"] == seq]
            ee = np.array([r[name] - r["gt"] for r in rs], float)
            gg = np.array([r["gt"] for r in rs], float)
            per_seq[seq] = {"mean_gt": round(float(gg.mean()), 1), "mae": round(float(np.abs(ee).mean()), 1),
                            "bias": round(float(ee.mean()), 1), "mape": round(float(np.mean(np.abs(ee) / gg)) * 100, 1)}
        summary[name] = {"mae": round(float(np.abs(e).mean()), 1), "bias": round(float(e.mean()), 1),
                         "mape_pct": round(float(np.mean(np.abs(e) / g)) * 100, 1),
                         "pearson_r": round(float(np.corrcoef(g, g + e)[0, 1]), 3),
                         "median_ms": float(np.median([r[f"{name}_ms"] for r in rows])), "per_sequence": per_seq}
    out = {"frames": len(rows), "summary": summary, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_d_counting.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
