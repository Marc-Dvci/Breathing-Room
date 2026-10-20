"""Test A, two follow-up questions, from eval/results/test_a_rows.csv.

1. Does the pushing index add to density, or only compete with it? Density alone and the index
   alone are each scored by AUROC; then both together, in a logistic model fitted on five runs
   and scored on the sixth (leave one run out), so no run scores a model that saw it. The 95%
   interval of the gain over density alone resamples whole people (clustered bootstrap).
2. How specific can an alert be? For each video pathway the threshold is set on runs 030/040/050
   at a target share of walking person-seconds flagged, and scored on runs 060/150/160, for the
   index alone (averaged over each person's last 3 s) and with the console's condition that
   pushing counts where the crowd reaches 2/m2. On these overhead runs DM-Count reads density
   2.4/m2 low (eval/results/test_a_scores.json), so that condition, on its own density, removes
   most of its pushing; the index alone is the fair reading of that pathway.
3. What does the console's own threshold do, and how often is a stretch of pushing missed
   altogether? The console alerts at the threshold set at the Youden point on runs 030/040/050
   (policy.Thresholds.warn_pressure; eval/results/test_a_threshold.json), which opens an
   investigation and never acts by itself. Besides person-seconds, each operating point is scored
   on episodes: a person's run of at least 2 consecutive seconds rated pushing is detected if any
   of its seconds is flagged; walking is reported as flagged seconds per minute of walking.

    python eval/test_a_operating.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
CAL, EVAL = {30, 40, 50}, {60, 150, 160}
SOURCES = {"video (heads)": ("rho_video_heads", "P_video_heads"), "video (DM-Count)": ("rho_video_dm", "P_video_dm")}
TARGETS = (0.05, 0.10, 0.15, 0.20)
PACKED = 2.0  # policy.Thresholds.watch_rho: pushing counts where the crowd reaches this density
PERSIST = 3  # seconds
CONSOLE: dict = {}


def loro(d: pd.DataFrame, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(d))
    for r in d.run.unique():
        m = d.run.values == r
        pred[m] = LogisticRegression(max_iter=1000).fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
    return pred


def gain_ci(d: pd.DataFrame, y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    rs = np.random.default_rng(seed)
    groups = list(d.groupby(["run", "pid"]).indices.values())
    out = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rs.integers(0, len(groups), len(groups))])
        if 0 < y[idx].mean() < 1:
            out.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def episodes(d: pd.DataFrame, y: np.ndarray, flag: np.ndarray, runs: set[int]) -> dict:
    """Pushing episodes (2 s or longer, per person) with at least one flagged second, on `runs`,
    and flagged seconds per minute of walking there."""
    hit, delay, walk_flags, walk_s = [], [], 0, 0
    sub = d[d.run.isin(runs)]  # d has a fresh RangeIndex: its index is the row position in y and flag
    for _, idx in sub.groupby(["run", "pid"]).indices.items():
        pos = sub.index.values[idx]
        yy, ff = y[pos], flag[pos]
        for seg in np.split(np.arange(len(yy)), np.flatnonzero(np.diff(yy)) + 1):
            if yy[seg[0]] == 1 and len(seg) >= 2:
                k = np.flatnonzero(ff[seg])
                hit.append(len(k) > 0)
                if len(k):
                    delay.append(int(k[0]))
            elif yy[seg[0]] == 0:
                walk_flags += int(ff[seg].sum())
                walk_s += len(seg)
    return {"pushing_episodes": len(hit), "episodes_detected": float(np.mean(hit)) if hit else None,
            "median_seconds_to_first_flag": float(np.median(delay)) if delay else None,
            "walking_flags_per_min": 60.0 * walk_flags / max(walk_s, 1)}


def main() -> dict:
    global CONSOLE
    CONSOLE = json.loads((RES / "test_a_threshold.json").read_text())
    d = pd.read_csv(RES / "test_a_rows.csv").sort_values(["run", "pid", "frame"]).reset_index(drop=True)
    y = (d.rating >= 3).astype(int).values
    cal, ev = d.run.isin(CAL).values, d.run.isin(EVAL).values
    out = {"rows": int(len(d)), "pushing_share": float(y.mean()), "sources": {}}
    for name, (rc, pc) in SOURCES.items():
        rho, P = d[rc].values, d[pc].values
        both = loro(d, np.c_[rho, P, rho * P], y)
        s = {"auc_density": float(roc_auc_score(y, rho)), "auc_pushing_index": float(roc_auc_score(y, P)),
             "auc_both_leave_one_run_out": float(roc_auc_score(y, both)),
             "gain_over_density_ci95": gain_ci(d, y, both, rho)}
        # the console's alert: the index over the last PERSIST s, where the crowd is packed
        g = d.groupby(["run", "pid"])
        Pm = g[pc].transform(lambda x: x.rolling(PERSIST, min_periods=1).mean()).values
        rm = g[rc].transform(lambda x: x.rolling(PERSIST, min_periods=1).mean()).values
        for rule, score in (("index_3s", Pm), ("index_3s_where_packed", np.where(rm >= PACKED, Pm, -np.inf))):
            neg_cal = score[cal & (y == 0)]
            s[rule] = []
            for target in TARGETS:
                thr = float(np.quantile(neg_cal, 1 - target))
                if not np.isfinite(thr):
                    continue
                flag = score >= thr
                s[rule].append({"target_walking_flagged": target, "threshold": thr,
                                "eval_pushing_flagged": float(flag[ev & (y == 1)].mean()),
                                "eval_walking_flagged": float(flag[ev & (y == 0)].mean()),
                                **episodes(d, y, flag, EVAL)})
        thr = CONSOLE[pc]["threshold"]  # the console's alert threshold, set at the Youden point on CAL
        flag = Pm >= thr
        s["console_threshold"] = {"threshold": thr, "eval_pushing_flagged": float(flag[ev & (y == 1)].mean()),
                                  "eval_walking_flagged": float(flag[ev & (y == 0)].mean()), **episodes(d, y, flag, EVAL)}
        out["sources"][name] = s
    (RES / "test_a_operating.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    o = main()
    for k, s in o["sources"].items():
        print(f"{k:18s} AUROC density {s['auc_density']:.3f}  index {s['auc_pushing_index']:.3f}  both {s['auc_both_leave_one_run_out']:.3f} "
              f"(gain over density {s['auc_both_leave_one_run_out'] - s['auc_density']:+.3f}, 95% {s['gain_over_density_ci95'][0]:+.3f} to {s['gain_over_density_ci95'][1]:+.3f})")
        for rule in ("index_3s", "index_3s_where_packed"):
            for p in s[rule]:
                print(f"   {rule:22s} set for {p['target_walking_flagged']:.0%} walking flagged: held-out pushing flagged "
                      f"{p['eval_pushing_flagged']:.0%}, walking flagged {p['eval_walking_flagged']:.0%}; "
                      f"episodes detected {p['episodes_detected']:.0%} of {p['pushing_episodes']}")
        c = s["console_threshold"]
        print(f"   console threshold {c['threshold']:.3f}: pushing flagged {c['eval_pushing_flagged']:.0%}, walking "
              f"{c['eval_walking_flagged']:.0%}; episodes detected {c['episodes_detected']:.0%} of {c['pushing_episodes']}, "
              f"{c['walking_flags_per_min']:.1f} flagged walking seconds per minute")
