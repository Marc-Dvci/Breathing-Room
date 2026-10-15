"""Score test A from eval/results/test_a_rows.csv: AUROC overall, by density band, and
the agreement between video-derived and trajectory-derived density."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
BANDS = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 9)]
SOURCES = {"trajectories": ("rho_traj", "P_traj"), "video (heads)": ("rho_video_heads", "P_video_heads"),
           "video (DM-Count)": ("rho_video_dm", "P_video_dm")}


def auc(y, x):
    return float(roc_auc_score(y, x)) if 0 < y.mean() < 1 and len(y) > 30 else None


def bootstrap_auc(d, col, y, n=1000, seed=0):
    """95% interval resampling whole runs and people (clustered bootstrap)."""
    rs = np.random.default_rng(seed)
    groups = d.groupby(["run", "pid"]).indices
    keys = list(groups)
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[i]] for i in rs.integers(0, len(keys), len(keys))])
        a = auc(y[idx], d[col].values[idx])
        if a is not None:
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main() -> dict:
    d = pd.read_csv(RES / "test_a_rows.csv")
    y = (d.rating >= 3).astype(int).values
    out = {"rows": int(len(d)), "people": int(d.groupby(["run", "pid"]).ngroups), "runs": sorted(d.run.astype(str).unique().tolist()),
           "pushing_share": float(y.mean()), "sources": {}}
    for name, (rc, pc) in SOURCES.items():
        if rc not in d:
            continue
        s = {"auc_density": auc(y, d[rc].values), "auc_pressure": auc(y, d[pc].values),
             "auc_pressure_ci95": bootstrap_auc(d, pc, y), "auc_pressure_by_density_band": {}}
        band_rho = d.rho_traj.values  # bands use the reference density so all sources are compared on the same rows
        for a, b in BANDS:
            m = (band_rho >= a) & (band_rho < b)
            s["auc_pressure_by_density_band"][f"{a}-{b}"] = {"n": int(m.sum()), "auc": auc(y[m], d[pc].values[m])}
        if rc != "rho_traj":
            s["density_vs_reference"] = {"pearson": float(np.corrcoef(d[rc], d.rho_traj)[0, 1]),
                                         "mae": float((d[rc] - d.rho_traj).abs().mean()),
                                         "bias": float((d[rc] - d.rho_traj).mean())}
        out["sources"][name] = s
    (RES / "test_a_scores.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = main()
    print(f"{o['rows']} rows, {o['people']} people, runs {o['runs']}, pushing share {o['pushing_share']:.2f}")
    for k, s in o["sources"].items():
        bands = " ".join(f"{b}:{v['auc']:.2f}" if v["auc"] else f"{b}:--" for b, v in s["auc_pressure_by_density_band"].items())
        extra = s.get("density_vs_reference", {})
        print(f"{k:18s} AUROC density {s['auc_density']:.3f}  pressure {s['auc_pressure']:.3f} "
              f"[{s['auc_pressure_ci95'][0]:.3f}, {s['auc_pressure_ci95'][1]:.3f}]  bands {bands}  {json.dumps(extra)}")
    sys.exit(0)
