"""Test B, part 2: minutes-ahead forecasts and the predicted effect of actions, scored
against the JuPedSim venue twin.

1. Lead time. In each baseline run (no action), every second the forecaster sees only the
   past and predicts when Zone C reaches 4.0 people/m^2. Scored on the seconds when the true
   crossing is 1 to 5 minutes ahead.
2. Counterfactuals. At the moment the forecast first says the zone is 5 minutes from 4.0/m^2,
   the same run (same seed, identical up to that moment) is branched once per action. The
   peak density the twin then produces over 5 minutes is compared with the peak the mass
   balance predicted for that action.

    python eval/test_b_twin.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from breathingroom import twin as tw  # noqa: E402
from breathingroom.physics import SPECIFIC_FLOW, Action, forecast  # noqa: E402

OUT = ROOT / "eval" / "results"
THRESHOLD = 4.0
SEEDS = (0, 1, 2, 3, 4)
ACTIONS = {
    "meter both gates to 150/min": Action("meter_entry", "both gates", rate_pps=150 / 60),
    "meter gate A to 60/min": Action("meter_entry", "gate A", rate_pps=60 / 60, share=0.5),
    "open second exit door": Action("open_exit", "second exit door", width_m=tw.SIDE_WIDTH),
    "hold both gates": Action("hold_entry", "both gates"),
}


def record(v: tw.Venue, until: float, hist: dict, act_at: float | None = None, action: str | None = None):
    while v.t < until:
        if act_at is not None and action and v.t >= act_at and not hist.get("acted"):
            apply(v, ACTIONS[action])
            hist["acted"] = v.t
        v.step(1.0)
        o = v.observe()
        hist["t"].append(o.t)
        hist["n"].append(o.n_zone)
        hist["out"].append(o.outflow_pps)


def apply(v: tw.Venue, a: Action):
    if a.kind == "meter_entry":
        gates = [a.target] if a.target in tw.GATES else list(tw.GATES)
        for g in gates:
            v.meter(g, a.rate_pps / len(gates))
    elif a.kind == "hold_entry":
        for g in tw.GATES:
            v.meter(g, 0.0)
    elif a.kind == "open_exit":
        v.open_side()


def fc_at(h, k, action=None, horizon=300):
    return forecast("Zone C", np.array(h["t"][: k + 1]), np.array(h["n"][: k + 1]), np.array(h["out"][: k + 1]),
                    tw.ZONE_C_AREA, tw.MAIN_EXIT_WIDTH, threshold=THRESHOLD, horizon_s=horizon, action=action, window_s=45,
                    min_points=40)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"threshold": THRESHOLD, "specific_flow": SPECIFIC_FLOW, "seeds": {}}
    lead_err = []
    cf_rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        base = {"t": [], "n": [], "out": []}
        record(tw.Venue(seed=seed), 600, base)
        rho = np.array(base["n"]) / tw.ZONE_C_AREA
        t = np.array(base["t"])
        hit = np.flatnonzero(rho >= THRESHOLD)
        cross = float(t[hit[0]]) if len(hit) else None
        pairs, warn_t = [], None
        for k in range(45, len(t)):
            if cross is not None and t[k] >= cross:
                break
            f = fc_at(base, k)
            if f.seconds_to_threshold is not None:
                if cross is not None:
                    pairs.append((cross - t[k], f.seconds_to_threshold))
                if warn_t is None and f.seconds_to_threshold <= 300:
                    warn_t = (k, float(t[k]))
        p = np.array(pairs)
        sel = (p[:, 0] >= 60) & (p[:, 0] <= 300) if len(p) else np.zeros(0, bool)
        if sel.any():
            lead_err += list(np.abs(p[sel, 1] - p[sel, 0]))
        seed_res = {"cross_s": cross, "warning_s": warn_t[1] if warn_t else None,
                    "lead_time_s": (cross - warn_t[1]) if (cross and warn_t) else None,
                    "median_abs_err_s_1_to_5_min_ahead": float(np.median(np.abs(p[sel, 1] - p[sel, 0]))) if sel.any() else None,
                    "pairs": p.tolist(), "baseline_rho": rho.tolist(), "actions": {}}
        if warn_t:
            k, ta = warn_t
            pred_none = fc_at(base, k)
            seed_res["actions"]["no action"] = {"predicted_peak": pred_none.peak_rho,
                                                "simulated_peak": float(rho[k:k + 300].max())}
            for name, a in ACTIONS.items():
                pred = fc_at(base, k, action=a)
                h = {"t": [], "n": [], "out": []}
                record(tw.Venue(seed=seed), ta + 300, h, act_at=ta, action=name)
                r = np.array(h["n"]) / tw.ZONE_C_AREA
                tt = np.array(h["t"])
                sim_peak = float(r[tt >= ta].max())
                seed_res["actions"][name] = {"predicted_peak": pred.peak_rho, "simulated_peak": sim_peak,
                                             "rho_curve": r[tt >= ta].tolist(), "pred_curve": pred.curve_rho,
                                             "pred_t": pred.curve_t}
                cf_rows.append((seed, name, pred.peak_rho, sim_peak))
        seed_res["wall_s"] = time.perf_counter() - t0
        out["seeds"][str(seed)] = seed_res
        print(f"seed {seed}: crossing {cross}, warning {warn_t}, {seed_res['wall_s']:.0f} s")
        for name, r in seed_res["actions"].items():
            print(f"   {name:30s} predicted peak {r['predicted_peak']:.2f}  simulated {r['simulated_peak']:.2f}")
    disruption = {"open_exit": 0, "meter_entry": 1, "hold_entry": 2}
    regrets, per_action = [], {}
    for v in out["seeds"].values():
        acts = {k: r for k, r in v["actions"].items() if k != "no action"}
        if not acts:
            continue
        # the deterministic planner's rule: lowest predicted peak (to 0.1/m2), then least disruption
        chosen = min(acts, key=lambda k: (round(acts[k]["predicted_peak"], 1), disruption[ACTIONS[k].kind]))
        best = min(r["simulated_peak"] for r in acts.values())
        regrets.append(acts[chosen]["simulated_peak"] - best)
        v["chosen"] = chosen
        for k, r in v["actions"].items():
            per_action.setdefault(k, []).append((r["predicted_peak"], r["simulated_peak"]))
    out["summary"] = {
        "median_abs_err_s_1_to_5_min_ahead": float(np.median(lead_err)) if lead_err else None,
        "lead_time_s": [v["lead_time_s"] for v in out["seeds"].values()],
        "peak_by_action": {k: {"predicted_mean": round(float(np.mean([a for a, _ in x])), 2),
                               "simulated_mean": round(float(np.mean([b for _, b in x])), 2),
                               "mean_abs_err": round(float(np.mean([abs(a - b) for a, b in x])), 2)} for k, x in per_action.items()},
        "chosen_action": [v.get("chosen") for v in out["seeds"].values()],
        "chosen_regret_per_m2": [round(r, 2) for r in regrets],
    }
    (OUT / "test_b_twin.json").write_text(json.dumps(out))
    print(json.dumps(out["summary"]))
    return out


if __name__ == "__main__":
    main()
