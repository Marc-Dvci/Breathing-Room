"""Exit capacity from the Jülich 2021 bottleneck experiments (DOI 10.34735/ped.2021.9).

For each of the 24 runs, every person's first crossing of the line x = 0.5 m (inside the
bottleneck, which starts at x = 0) is found from the trajectories. The steady flow is the
number of crossings between the 10th and the 90th percentile crossing, divided by the time
between them. A line through the origin, flow = a * width, gives the capacity per metre.

    python eval/test_b_capacity.py
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
from breathingroom.physics import SPECIFIC_FLOW  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data")) / "fzj" / "bneck"
OUT = ROOT / "eval" / "results"
LINE_X = 0.5


def steady_flow(T: fzj.Trajectories) -> tuple[float, int]:
    order = np.lexsort((T.frames, T.ids))
    ids, fr, x = T.ids[order], T.frames[order], T.xy[order, 0]
    cross = []
    for pid in np.unique(ids):
        m = ids == pid
        k = np.flatnonzero((x[m][:-1] < LINE_X) & (x[m][1:] >= LINE_X))
        if len(k):
            cross.append(fr[m][k[0] + 1] / T.fps)
    c = np.sort(cross)
    lo, hi = int(0.1 * len(c)), int(0.9 * len(c))
    return (hi - lo) / max(c[hi] - c[lo], 1e-9), len(c)


def main() -> dict:
    rows = []
    for p in sorted(DATA.glob("*.txt")):
        w = fzj.bottleneck_width(p.name)
        flow, n = steady_flow(fzj.load_trajectories(p))
        rows.append({"run": p.stem, "width_m": w, "people": n, "flow_pps": round(flow, 3),
                     "specific_flow": round(flow / w, 3), "motivation": fzj.bottleneck_motivation(p.name)})
        print(f"{p.stem:28s} w={w:.2f} m  n={n:4d}  flow={flow:.2f}/s  specific={flow / w:.2f}/m/s")
    w = np.array([r["width_m"] for r in rows])
    q = np.array([r["flow_pps"] for r in rows])
    a = float((w @ q) / (w @ w))
    r2 = 1 - ((q - a * w) ** 2).sum() / ((q - q.mean()) ** 2).sum()
    sf = q / w
    out = {"runs": rows, "fit_flow_per_metre": round(a, 3), "r2": round(float(r2), 3),
           "specific_flow_range": [round(float(sf.min()), 2), round(float(sf.max()), 2)],
           "system_value": SPECIFIC_FLOW,
           "runs_below_system_value": int((sf < SPECIFIC_FLOW).sum())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_b_capacity.json").write_text(json.dumps(out, indent=1))
    print(f"flow = {a:.2f} x width (R2 {r2:.2f}); specific flow {sf.min():.2f} to {sf.max():.2f} p/m/s; "
          f"system uses {SPECIFIC_FLOW} ({out['runs_below_system_value']} of {len(rows)} runs below it)")
    return out


if __name__ == "__main__":
    main()
