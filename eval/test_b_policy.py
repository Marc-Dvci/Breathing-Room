"""Test B under the console's own alert policy, with the queue each action moves upstream.

eval/test_b_venues.py scores a warning given 5 minutes before 4.0/m2. The console works
differently: `TierPolicy` opens an investigation at WARN (10 minutes to 4.5/m2 at the current
trend, or 3.5/m2 measured) and raises ACT at 5 minutes to 4.5/m2, or at 4.5/m2 measured. This
test replays the twin through `TierPolicy` itself, with the console's forecast settings, and
reports when WARN and ACT came against when the zone reached 4.5/m2, and how often they came in
evenings that never reached it.

From the moment the investigation opens, every action is simulated as the console would propose
it, and two things are recorded: the zone's peak, and the people the action keeps waiting before
the gates (the twin counts them: metering and holding move part of the crowd upstream).

Two choices are scored on the same simulations:
- the console's venue-wide rule (breathingroom.choice), which judges each action on the zone and
  on the waiting area before the gates it restricts: zone peak plus the largest error that
  action's forecast made on the reference venue, and people held times the largest factor by
  which the queue forecast fell short there, over the waiting area (GATE_DEPTH_M deep across the
  street before each gate, the console's assumption for its own venue);
- the earlier rule, on the zone's peak alone (least disruptive within 0.25/m2 of the lowest peak
  plus mean error), kept for comparison: it meters both gates and holds hundreds outside.
Each is scored on what the twin then did: the zone's peak, the people held, their density in
the waiting area, and the densest of the two places against the best action in hindsight.

Layouts: the reference venue, the seven of test_b_venues, and two with a wide second door, where
opening it competes with metering.

    python eval/test_b_policy.py          # after eval/test_b_twin.py; about an hour on one core
    python eval/test_b_policy.py --score  # score the saved simulations again
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
from breathingroom import choice as rule  # noqa: E402
from breathingroom import twin as tw  # noqa: E402
from breathingroom.physics import forecast  # noqa: E402
from breathingroom.policy import Thresholds, Tier, TierPolicy, ZoneReading  # noqa: E402
from score_test_b import DISRUPTION, choose  # noqa: E402
from test_b_venues import LAYOUTS as VENUES  # noqa: E402
from test_b_venues import actions, apply  # noqa: E402

OUT = ROOT / "eval" / "results"
TH = Thresholds()
SEEDS = (0, 1, 2)
ACTION_S = 300  # each action is simulated, and forecast, over 5 minutes from the investigation
GATE_DEPTH_M = 20.0  # waiting area before each gate: this deep, across half the street (runtime.GATE_AREA_M2 on the reference venue)
KIND = {"open second exit door": "open_exit", "meter both gates to 150/min": "meter_entry",
        "meter gate A to 60/min": "meter_entry", "hold both gates": "hold_entry"}
LAYOUTS = {"reference venue": tw.DEFAULT, **VENUES,
           "wide second door": tw.Layout(length=30, width=8, exit_width=1.6, door_width=2.8, demand=1.6 / 2.4),
           "door wider than exit": tw.Layout(length=25, width=7, exit_width=1.8, door_width=2.2, demand=1.8 / 2.4)}


def record(v: tw.Venue, until: float, h: dict, act_at: float | None = None, action=None) -> None:
    while v.t < until:
        if act_at is not None and action is not None and v.t >= act_at and not h.get("acted"):
            apply(v, action)
            h["acted"] = v.t
        v.step(1.0)
        o = v.observe()
        h["t"].append(o.t)
        h["n"].append(o.n_zone)
        h["out"].append(o.outflow_pps)
        h["held"].append(sum(o.backlog.values()))


def fc_at(L: tw.Layout, h: dict, k: int, action=None, horizon: float = 600):
    """The console's forecast (runtime.TwinZone.forecast): threshold 4.5, 45 s window."""
    return forecast("Zone C", np.array(h["t"][: k + 1]), np.array(h["n"][: k + 1]), np.array(h["out"][: k + 1]),
                    L.zone_c_area, L.exit_width, threshold=TH.act_rho, horizon_s=horizon, action=action,
                    window_s=45, min_points=40)


def first_cross(t: np.ndarray, rho: np.ndarray, level: float) -> float | None:
    k = np.flatnonzero(rho >= level)
    return float(t[k[0]]) if len(k) else None


def run(L: tw.Layout, seed: int) -> dict:
    t0 = time.perf_counter()
    base = {"t": [], "n": [], "out": [], "held": []}
    record(tw.Venue(seed=seed, layout=L), 600, base)
    t, rho = np.array(base["t"]), np.array(base["n"]) / L.zone_c_area
    pol = TierPolicy(TH)
    first: dict[str, tuple[int, float, list[str]]] = {}
    for k in range(45, len(t)):
        f = fc_at(L, base, k)
        mins = None if f.seconds_to_threshold is None else f.seconds_to_threshold / 60
        d = pol.decide(ZoneReading("Zone C", float(t[k]), float(rho[k]), 0.0, mins))
        for tier in (Tier.WARN, Tier.ACT):
            if d.tier >= tier and tier.name not in first:
                first[tier.name] = (k, float(t[k]), d.reasons)
    c45, c55 = first_cross(t, rho, TH.act_rho), first_cross(t, rho, TH.critical_rho)
    res = {"peak_rho": float(rho.max()), "cross_4_5_s": c45, "cross_5_5_s": c55,
           "warn_s": first["WARN"][1] if "WARN" in first else None,
           "warn_reasons": first["WARN"][2] if "WARN" in first else [],
           "act_s": first["ACT"][1] if "ACT" in first else None,
           "held_no_action_peak": float(max(base["held"])), "actions": {}}
    res["warn_lead_s"] = (c45 - res["warn_s"]) if (c45 is not None and res["warn_s"] is not None) else None
    res["act_lead_s"] = (c45 - res["act_s"]) if (c45 is not None and res["act_s"] is not None) else None
    if "WARN" in first:
        k, ta, _ = first["WARN"]
        end = min(k + ACTION_S, len(t) - 1)
        res["actions"]["no action"] = {"predicted_peak": fc_at(L, base, k, horizon=ACTION_S).peak_rho,
                                       "simulated_peak": float(rho[k:end + 1].max()),
                                       "predicted_held": 0.0, "simulated_held": float(max(base["held"][k:end + 1]))}
        for name, a in actions(L).items():
            h = {"t": [], "n": [], "out": [], "held": []}
            record(tw.Venue(seed=seed, layout=L), ta + ACTION_S, h, act_at=ta, action=a)
            r, tt = np.array(h["n"]) / L.zone_c_area, np.array(h["t"])
            f = fc_at(L, base, k, action=a, horizon=ACTION_S)
            after = tt >= ta
            res["actions"][name] = {"predicted_peak": f.peak_rho, "simulated_peak": float(r[after].max()),
                                    "predicted_held": f.held_back_peak,
                                    "simulated_held": float(np.array(h["held"])[after].max())}
    res["wall_s"] = time.perf_counter() - t0
    return res


def waiting_area(L: tw.Layout, name: str) -> float:
    """m2 before the gates an action restricts (both gates for actions that restrict none: the
    queue the gates build on their own waits there too)."""
    per_gate = L.width / 2 * GATE_DEPTH_M
    return per_gate if name == "meter gate A to 60/min" else 2 * per_gate


def venue_rule(acts: dict, L: tw.Layout, err: dict, queue: dict) -> str:
    """The console's choice (breathingroom.choice) from the forecasts made at the investigation."""
    opts = [rule.Option(n, KIND[n], acts[n]["predicted_peak"], err[n]["mean"], err[n]["max"], acts[n]["predicted_held"],
                        queue.get(n, 1.0), waiting_area(L, n) if KIND[n] in ("meter_entry", "hold_entry") else None)
            for n in DISRUPTION]
    return rule.choose(opts).chosen.key


def score(out: dict, ref: dict) -> dict:
    from breathingroom.runtime import forecast_skill, queue_skill

    err = forecast_skill(OUT / "test_b_twin.json")  # mean and largest error per action, reference venue
    queue = queue_skill(OUT / "test_b_policy.json")  # largest queue shortfall per action, reference venue
    runs = [(name, s, r) for name, seeds in out["venues"].items() for s, r in seeds.items()]
    crossed = [r for _, _, r in runs if r["cross_4_5_s"] is not None]
    not_crossed = [r for _, _, r in runs if r["cross_4_5_s"] is None]
    rows = []
    for name, s, r in runs:
        acts = r["actions"]
        if not acts:
            continue
        L = LAYOUTS[name]
        best_peak = min(acts[n]["simulated_peak"] for n in DISRUPTION)
        best = min(DISRUPTION, key=lambda n: (acts[n]["simulated_peak"], DISRUPTION[n]))
        # what each action did to the densest place it touches: the zone or the waiting area
        worst = {n: max(acts[n]["simulated_peak"], acts[n]["simulated_held"] / waiting_area(L, n)) for n in DISRUPTION}
        best_worst = min(worst.values())
        chosen = choose({n: acts[n]["predicted_peak"] + err[n]["mean"] for n in DISRUPTION})  # zone peak only
        console = venue_rule(acts, L, err, queue)
        # the gentlest action whose simulated peak is within 0.05/m2 of the best: the right answer in hindsight
        right = min((n for n in DISRUPTION if acts[n]["simulated_peak"] <= best_peak + 0.05), key=lambda n: DISRUPTION[n])

        def out_of(n):
            return {"zone_peak": round(acts[n]["simulated_peak"], 2), "held": round(acts[n]["simulated_held"]),
                    "waiting_area_m2": waiting_area(L, n),
                    "waiting_density": round(acts[n]["simulated_held"] / waiting_area(L, n), 2),
                    "densest_place": round(worst[n], 2), "regret_densest_place": round(worst[n] - best_worst, 2),
                    "regret_zone_peak": round(acts[n]["simulated_peak"] - best_peak, 2)}

        rows.append({"venue": name, "seed": s, "best": best, "gentlest_near_best": right,
                     "best_densest_place": min(worst, key=worst.get),
                     "venue_rule": console, "venue_rule_outcome": out_of(console),
                     "chosen": chosen, "peak_only_outcome": out_of(chosen),
                     "regret": round(acts[chosen]["simulated_peak"] - best_peak, 2),
                     "held_by_chosen": round(acts[chosen]["simulated_held"]),
                     "held_predicted_for_chosen": round(acts[chosen]["predicted_held"]),
                     "held_without_action": round(acts["no action"]["simulated_held"]),
                     **{f"regret_always_{n}": round(acts[n]["simulated_peak"] - best_peak, 2) for n in DISRUPTION}})

    def outcome(key):
        o = [r[key] for r in rows]
        return {"zone_peak_max": max(x["zone_peak"] for x in o), "zone_peak_below_warn": sum(x["zone_peak"] < TH.warn_rho for x in o),
                "held_median": float(np.median([x["held"] for x in o])), "held_max": max(x["held"] for x in o),
                "waiting_density_max": max(x["waiting_density"] for x in o),
                "densest_place_max": max(x["densest_place"] for x in o),
                "regret_densest_place_mean": round(float(np.mean([x["regret_densest_place"] for x in o])), 2),
                "regret_densest_place_max": max(x["regret_densest_place"] for x in o),
                "regret_zone_peak_mean": round(float(np.mean([x["regret_zone_peak"] for x in o])), 2)}
    metered = [(a["predicted_held"], a["simulated_held"]) for _, _, r in runs for n, a in r["actions"].items()
               if n.startswith(("meter", "hold"))]
    p, q = np.array(metered).T if metered else (np.zeros(0), np.zeros(0))

    def stat(xs):
        return {"min": min(xs), "median": float(np.median(xs)), "max": max(xs)} if xs else None

    return {
        "runs": len(runs), "reached_4_5": len(crossed),
        "warn_before_4_5": sum(r["warn_lead_s"] is not None and r["warn_lead_s"] > 0 for r in crossed),
        "act_before_4_5": sum(r["act_lead_s"] is not None and r["act_lead_s"] > 0 for r in crossed),
        "warn_lead_s": stat([r["warn_lead_s"] for r in crossed if r["warn_lead_s"] is not None]),
        "act_lead_s": stat([r["act_lead_s"] for r in crossed if r["act_lead_s"] is not None]),
        "never_reached_4_5": len(not_crossed),
        "warn_in_those": sum(r["warn_s"] is not None for r in not_crossed),
        "act_in_those": sum(r["act_s"] is not None for r in not_crossed),
        "peak_in_those": stat([r["peak_rho"] for r in not_crossed]),
        "error_priced_from_reference": err, "queue_factor_from_reference": queue,
        "waiting_area": f"{GATE_DEPTH_M:.0f} m deep across half the street before each gate",
        "choice": rows,
        "venue_rule": {"chosen_counts": {n: sum(r["venue_rule"] == n for r in rows) for n in DISRUPTION},
                       "best_densest_place_counts": {n: sum(r["best_densest_place"] == n for r in rows) for n in DISRUPTION},
                       **outcome("venue_rule_outcome")},
        "peak_only_rule": {"chosen_counts": {n: sum(r["chosen"] == n for r in rows) for n in DISRUPTION},
                           **outcome("peak_only_outcome")},
        "chosen_counts": {n: sum(r["chosen"] == n for r in rows) for n in DISRUPTION},
        "gentlest_near_best_counts": {n: sum(r["gentlest_near_best"] == n for r in rows) for n in DISRUPTION},
        "regret_under_0_25": sum(r["regret"] < 0.25 for r in rows),
        "mean_regret": round(float(np.mean([r["regret"] for r in rows])), 2) if rows else None,
        "mean_regret_always": {n: round(float(np.mean([r[f"regret_always_{n}"] for r in rows])), 2) for n in DISRUPTION} if rows else None,
        "held_upstream": {"metering_or_holding_runs": len(metered),
                          "simulated_median": float(np.median(q)) if len(q) else None,
                          "simulated_max": float(q.max()) if len(q) else None,
                          "prediction_mae": float(np.mean(np.abs(p - q))) if len(q) else None,
                          "prediction_bias": float(np.mean(p - q)) if len(q) else None},
    }


def main(only: list[str] | None = None) -> dict:
    ref = json.loads((OUT / "test_b_twin.json").read_text())
    path = OUT / "test_b_policy.json"
    out = {"thresholds": {"warn_minutes": TH.warn_minutes, "act_minutes": TH.act_minutes, "act_rho": TH.act_rho,
                          "warn_rho": TH.warn_rho}, "action_horizon_s": ACTION_S,
           "layouts": {n: {"length": L.length, "width": L.width, "exit_width": L.exit_width, "door_width": L.door_width,
                           "demand": round(L.demand, 3)} for n, L in LAYOUTS.items()}, "venues": {}}
    for name, L in LAYOUTS.items():
        if only and name not in only:
            continue
        out["venues"][name] = {}
        for seed in SEEDS:
            r = run(L, seed)
            out["venues"][name][str(seed)] = r
            print(f"{name:20s} seed {seed}: peak {r['peak_rho']:.2f}, 4.5 at {r['cross_4_5_s']}, WARN {r['warn_s']}, "
                  f"ACT {r['act_s']}, {r['wall_s']:.0f} s", flush=True)
            for a, x in r["actions"].items():
                print(f"   {a:30s} peak predicted {x['predicted_peak']:.2f} simulated {x['simulated_peak']:.2f}   "
                      f"held predicted {x['predicted_held']:.0f} simulated {x['simulated_held']:.0f}", flush=True)
            path.write_text(json.dumps(out))  # partial results survive an interruption
    out["summary"] = score(out, ref)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "choice"}, indent=1))
    return out


def rescore() -> None:
    """Score the saved runs again (after a change to the scoring) without simulating."""
    path = OUT / "test_b_policy.json"
    out = json.loads(path.read_text())
    out["summary"] = score(out, json.loads((OUT / "test_b_twin.json").read_text()))
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "choice"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["--score"]:
        rescore()
    else:
        main(sys.argv[1:] or None)
