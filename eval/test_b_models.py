"""Test B against a second pedestrian model.

Every other test B scores the forecast against JuPedSim's collision-free speed model, the model
the forecaster's settings were chosen on, so the forecast and its ground truth could share that
model's habits. Here the crowd follows JuPedSim's anticipation velocity model instead: people
anticipate their neighbours' motion and streams that cross keep moving. The forecaster, its
settings, the console's `TierPolicy`, the layouts and the arrivals are unchanged. Reported per
model, on the same evenings: how many reached 4.5/m2, and when WARN and ACT came against it; and,
from the investigation, what opening the second door and metering both gates did to the zone and
to the people held before the gates.

    python eval/test_b_models.py      # after eval/test_b_policy.py; about two hours on one core
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from breathingroom import twin as tw  # noqa: E402
from breathingroom.policy import Tier, TierPolicy, ZoneReading  # noqa: E402
from test_b_policy import ACTION_S, LAYOUTS, TH, fc_at, first_cross, record  # noqa: E402
from test_b_venues import actions  # noqa: E402

OUT = ROOT / "eval" / "results"
SEEDS = (0, 1)
MODEL = "avm"
ACTIONS = ("open second exit door", "meter both gates to 150/min")


def run(L: tw.Layout, seed: int) -> dict:
    t0 = time.perf_counter()
    base = {"t": [], "n": [], "out": [], "held": []}
    record(tw.Venue(seed=seed, layout=L, model=MODEL), 600, base)
    t, rho = np.array(base["t"]), np.array(base["n"]) / L.zone_c_area
    pol = TierPolicy(TH)
    first: dict[str, tuple[int, float]] = {}
    for k in range(45, len(t)):
        f = fc_at(L, base, k)
        mins = None if f.seconds_to_threshold is None else f.seconds_to_threshold / 60
        d = pol.decide(ZoneReading("Zone C", float(t[k]), float(rho[k]), 0.0, mins))
        for tier in (Tier.WARN, Tier.ACT):
            if d.tier >= tier and tier.name not in first:
                first[tier.name] = (k, float(t[k]))
    c45 = first_cross(t, rho, TH.act_rho)
    res = {"peak_rho": float(rho.max()), "cross_4_5_s": c45,
           "warn_s": first["WARN"][1] if "WARN" in first else None, "act_s": first["ACT"][1] if "ACT" in first else None,
           "actions": {}}
    res["warn_lead_s"] = None if c45 is None or res["warn_s"] is None else c45 - res["warn_s"]
    res["act_lead_s"] = None if c45 is None or res["act_s"] is None else c45 - res["act_s"]
    if "WARN" in first:
        k, ta = first["WARN"]
        for name in ACTIONS:
            a = actions(L)[name]
            h = {"t": [], "n": [], "out": [], "held": []}
            record(tw.Venue(seed=seed, layout=L, model=MODEL), ta + ACTION_S, h, act_at=ta, action=a)
            r, tt = np.array(h["n"]) / L.zone_c_area, np.array(h["t"])
            after = tt >= ta
            res["actions"][name] = {"predicted_peak": fc_at(L, base, k, action=a, horizon=ACTION_S).peak_rho,
                                    "simulated_peak": float(r[after].max()),
                                    "simulated_held": float(np.array(h["held"])[after].max())}
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res


def summarise(out: dict) -> dict:
    csm = json.loads((OUT / "test_b_policy.json").read_text())["venues"]
    pairs = [(out["runs"][v][s], csm[v][s]) for v in out["runs"] for s in out["runs"][v]]

    def stat(xs):
        xs = [x for x in xs if x is not None]
        return {"min": min(xs), "median": float(np.median(xs)), "max": max(xs), "n": len(xs)} if xs else None

    s = {}
    for label, i in ((MODEL, 0), ("csm, same evenings", 1)):
        runs = [p[i] for p in pairs]
        crossed = [r for r in runs if r["cross_4_5_s"] is not None]
        s[label] = {"runs": len(runs), "reached_4_5": len(crossed), "peak_rho": stat([r["peak_rho"] for r in runs]),
                    "warn_before_4_5": sum(r["warn_lead_s"] is not None and r["warn_lead_s"] > 0 for r in crossed),
                    "act_before_4_5": sum(r["act_lead_s"] is not None and r["act_lead_s"] > 0 for r in crossed),
                    "warn_lead_s": stat([r["warn_lead_s"] for r in crossed]),
                    "act_lead_s": stat([r["act_lead_s"] for r in crossed]),
                    "warned_without_reaching": sum(r["warn_s"] is not None for r in runs if r["cross_4_5_s"] is None)}
        for name in ACTIONS:
            acts = [r["actions"][name] for r in runs if name in r.get("actions", {})]
            s[label][name] = {"zone_peak": stat([a["simulated_peak"] for a in acts]),
                              "held": stat([a["simulated_held"] for a in acts]),
                              "peak_error": stat([a["simulated_peak"] - a["predicted_peak"] for a in acts])}
    return s


def main() -> dict:
    path = OUT / "test_b_models.json"
    out = {"model": MODEL, "seeds": list(SEEDS), "runs": {}}
    for name, L in LAYOUTS.items():
        out["runs"][name] = {}
        for seed in SEEDS:
            r = run(L, seed)
            out["runs"][name][str(seed)] = r
            print(f"{name:20s} seed {seed}: peak {r['peak_rho']:.2f}, 4.5 at {r['cross_4_5_s']}, WARN {r['warn_s']}, "
                  f"ACT {r['act_s']}, {r['wall_s']:.0f} s", flush=True)
            path.write_text(json.dumps(out))
    out["summary"] = summarise(out)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    return out


if __name__ == "__main__":
    main()
