"""Test B with sudden surges the forecast cannot see coming.

The forecast carries the inflow's trend of the last 45 s forward for two minutes. Its weak
point is a surge that starts without a trend: a train unloading, a second hall letting out.
In the evenings of test B the arrivals ramp up over three minutes, so the trend gives warning.
Here, on the ten layouts of eval/test_b_policy.py, the street is quiet (QUIET_PPS, scaled by
each layout's demand) and the arrivals jump, at SURGE_AT_S and with no ramp, to the full rate
of the test B evenings (FULL_PPS), and the console's `TierPolicy` is replayed as test_b_policy
does:

  pulse  the full rate for 120 s (a train: about 780 people on the reference venue), then quiet
  step   the full rate from SURGE_AT_S to the end (a hall letting out)

Reported per kind: evenings that reached 4.5/m2, how many were warned (WARN) and told to act
(ACT) before it, the lead times, and the time from the start of the surge to WARN, against the
same evening without a surge (eval/results/test_b_policy.json).

    python eval/test_b_surge.py      # after eval/test_b_policy.py; about an hour on one core
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
from test_b_policy import LAYOUTS, SEEDS, TH, fc_at, first_cross, record  # noqa: E402

OUT = ROOT / "eval" / "results"
SURGE_AT_S = 180.0
END_S = 600.0
QUIET_PPS, FULL_PPS = 1.5, 6.5  # arrivals, persons/s on the reference venue: the ends of tw.egress_profile
KINDS = {"pulse": 120.0, "step": None}  # seconds at the full rate (None: to the end)


def surged(L: tw.Layout, duration: float | None):
    def profile(t: float) -> float:
        on = t >= SURGE_AT_S and (duration is None or t < SURGE_AT_S + duration)
        return L.demand * (FULL_PPS if on else QUIET_PPS)
    return profile


def run(L: tw.Layout, seed: int, kind: str) -> dict:
    t0 = time.perf_counter()
    duration = KINDS[kind]
    h = {"t": [], "n": [], "out": [], "held": []}
    record(tw.Venue(seed=seed, layout=L, profile=surged(L, duration)), END_S, h)
    t, rho = np.array(h["t"]), np.array(h["n"]) / L.zone_c_area
    pol = TierPolicy(TH)
    first: dict[str, float] = {}
    for k in range(45, len(t)):
        f = fc_at(L, h, k)
        mins = None if f.seconds_to_threshold is None else f.seconds_to_threshold / 60
        d = pol.decide(ZoneReading("Zone C", float(t[k]), float(rho[k]), 0.0, mins))
        for tier in (Tier.WARN, Tier.ACT):
            if d.tier >= tier and tier.name not in first:
                first[tier.name] = float(t[k])
    c45 = first_cross(t, rho, TH.act_rho)
    # the tier the zone was at when the surge began: a warning already standing covers it
    at_surge = {n: v for n, v in first.items() if v <= SURGE_AT_S}
    res = {"peak_rho": float(rho.max()), "cross_4_5_s": c45, "warn_s": first.get("WARN"), "act_s": first.get("ACT"),
           "warned_before_surge": "WARN" in at_surge,
           "surge_people": round(L.demand * (FULL_PPS - QUIET_PPS) * ((duration or END_S - SURGE_AT_S))),
           "wall_s": round(time.perf_counter() - t0, 1)}
    for tier in ("warn", "act"):
        s = res[f"{tier}_s"]
        res[f"{tier}_lead_s"] = None if c45 is None or s is None else c45 - s
        res[f"{tier}_after_surge_s"] = None if s is None or s < SURGE_AT_S else s - SURGE_AT_S
    res["surge_to_4_5_s"] = None if c45 is None else c45 - SURGE_AT_S
    return res


def summarise(out: dict) -> dict:
    base = json.loads((OUT / "test_b_policy.json").read_text())["venues"]

    def stat(xs):
        xs = [x for x in xs if x is not None]
        return {"min": min(xs), "median": float(np.median(xs)), "max": max(xs), "n": len(xs)} if xs else None

    s = {}
    for kind in KINDS:
        runs = [(v, sd, r) for v, seeds in out["runs"].items() for sd, rs in seeds.items() for k, r in rs.items() if k == kind]
        crossed = [r for _, _, r in runs if r["cross_4_5_s"] is not None]
        base_cross = [base[v][sd] for v, sd, r in runs if base[v][sd]["cross_4_5_s"] is not None]
        s[kind] = {"runs": len(runs), "surge_people": stat([r["surge_people"] for _, _, r in runs]),
                   "peak_rho": stat([r["peak_rho"] for _, _, r in runs]),
                   "reached_4_5": len(crossed), "reached_4_5_without_surge": len(base_cross),
                   "warn_before_4_5": sum(r["warn_lead_s"] is not None and r["warn_lead_s"] > 0 for r in crossed),
                   "act_before_4_5": sum(r["act_lead_s"] is not None and r["act_lead_s"] > 0 for r in crossed),
                   "warn_lead_s": stat([r["warn_lead_s"] for r in crossed]),
                   "act_lead_s": stat([r["act_lead_s"] for r in crossed]),
                   "warned_before_the_surge": sum(r["warned_before_surge"] for r in crossed),
                   "warn_after_surge_s": stat([r["warn_after_surge_s"] for r in crossed]),
                   "act_after_surge_s": stat([r["act_after_surge_s"] for r in crossed]),
                   "surge_to_4_5_s": stat([r["surge_to_4_5_s"] for r in crossed]),
                   "act_lead_without_surge_s": stat([b["act_lead_s"] for b in base_cross])}
    return s


def main() -> dict:
    path = OUT / "test_b_surge.json"
    out = {"surge_at_s": SURGE_AT_S, "end_s": END_S, "quiet_pps": QUIET_PPS, "full_pps": FULL_PPS,
           "kinds": {k: {"seconds_at_full_rate": d} for k, d in KINDS.items()}, "runs": {}}
    for name, L in LAYOUTS.items():
        out["runs"][name] = {}
        for seed in SEEDS:
            out["runs"][name][str(seed)] = {}
            for kind in KINDS:
                r = run(L, seed, kind)
                out["runs"][name][str(seed)][kind] = r
                print(f"{name:20s} seed {seed} {kind:5s}: peak {r['peak_rho']:.2f}, 4.5 at {r['cross_4_5_s']}, "
                      f"WARN {r['warn_s']}, ACT {r['act_s']}, {r['wall_s']:.0f} s", flush=True)
                path.write_text(json.dumps(out))
    out["summary"] = summarise(out)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    return out


if __name__ == "__main__":
    main()
