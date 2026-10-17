"""Test B on other venues: the forecast and the action choice, unchanged, on seven street layouts
the system was never tuned on.

Each layout changes the street (18 to 45 m long, 5 to 10 m wide), the main exit (1.6 to 4.0 m),
the second door and the surge. The forecaster keeps its window, its specific flow and its
threshold. The action choice prices each action with the forecast error measured on the
reference venue only (eval/results/test_b_twin.json), as the console does, so the errors are
carried to venues they were not measured on. Metering rates scale with the main exit width.

    python eval/test_b_venues.py      # after eval/test_b_twin.py
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
from breathingroom.physics import Action, forecast  # noqa: E402
from score_test_b import DISRUPTION, choose  # noqa: E402

OUT = ROOT / "eval" / "results"
THRESHOLD = 4.0
SEEDS = (0, 1, 2)
LAYOUTS = {  # surge peak per metre of main exit: 2.7 persons/s/m on the reference venue
    "short and narrow": tw.Layout(length=20, width=5, exit_width=2.0, door_width=1.2, demand=2.0 / 2.4),
    "long and wide": tw.Layout(length=40, width=8, exit_width=3.2, door_width=1.8, demand=1.1 * 3.2 / 2.4),
    "narrow exit": tw.Layout(length=25, width=7, exit_width=1.8, door_width=1.5, demand=1.8 / 2.4),
    "long, small exit": tw.Layout(length=35, width=5, exit_width=1.6, door_width=1.0, demand=1.2 * 1.6 / 2.4),
    "plaza, wide exit": tw.Layout(length=30, width=10, exit_width=4.0, door_width=2.0, demand=4.0 / 2.4),
    "short, wide exit": tw.Layout(length=18, width=6, exit_width=3.0, door_width=1.2, demand=0.9 * 3.0 / 2.4),
    "long street": tw.Layout(length=45, width=6, exit_width=2.4, door_width=1.5, demand=1.0),
}


def actions(L: tw.Layout) -> dict[str, Action]:
    r = L.exit_width / tw.MAIN_EXIT_WIDTH
    return {
        "meter both gates to 150/min": Action("meter_entry", "both gates", rate_pps=r * 150 / 60),
        "meter gate A to 60/min": Action("meter_entry", "gate A", rate_pps=r * 60 / 60, share=0.5),
        "open second exit door": Action("open_exit", "second exit door", width_m=L.door_width),
        "hold both gates": Action("hold_entry", "both gates"),
    }


def apply(v: tw.Venue, a: Action):
    if a.kind == "meter_entry":
        gates = [a.target] if a.target in v.gates else list(v.gates)
        for g in gates:
            v.meter(g, a.rate_pps / len(gates))
    elif a.kind == "hold_entry":
        for g in v.gates:
            v.meter(g, 0.0)
    elif a.kind == "open_exit":
        v.open_side()


def record(v: tw.Venue, until: float, hist: dict, act_at: float | None = None, action: Action | None = None):
    while v.t < until:
        if act_at is not None and action and v.t >= act_at and not hist.get("acted"):
            apply(v, action)
            hist["acted"] = v.t
        v.step(1.0)
        o = v.observe()
        hist["t"].append(o.t)
        hist["n"].append(o.n_zone)
        hist["out"].append(o.outflow_pps)


def fc_at(L: tw.Layout, h, k, action=None, horizon=300):
    return forecast("Zone C", np.array(h["t"][: k + 1]), np.array(h["n"][: k + 1]), np.array(h["out"][: k + 1]),
                    L.zone_c_area, L.exit_width, threshold=THRESHOLD, horizon_s=horizon, action=action, window_s=45,
                    min_points=40)


def run(L: tw.Layout, seed: int) -> dict:
    t0 = time.perf_counter()
    base = {"t": [], "n": [], "out": []}
    record(tw.Venue(seed=seed, layout=L), 600, base)
    rho = np.array(base["n"]) / L.zone_c_area
    t = np.array(base["t"])
    hit = np.flatnonzero(rho >= THRESHOLD)
    cross = float(t[hit[0]]) if len(hit) else None
    pairs, warn = [], None
    for k in range(45, len(t)):
        if cross is not None and t[k] >= cross:
            break
        f = fc_at(L, base, k)
        if f.seconds_to_threshold is not None:
            if cross is not None:
                pairs.append((cross - t[k], f.seconds_to_threshold))
            if warn is None and f.seconds_to_threshold <= 300:
                warn = (k, float(t[k]))
    p = np.array(pairs)
    sel = (p[:, 0] >= 60) & (p[:, 0] <= 300) if len(p) else np.zeros(0, bool)
    res = {"cross_s": cross, "warning_s": warn[1] if warn else None,
           "lead_time_s": (cross - warn[1]) if (cross and warn) else None,
           "abs_err_s_1_to_5_min_ahead": np.abs(p[sel, 1] - p[sel, 0]).tolist() if sel.any() else [],
           "peak_rho": float(rho.max()), "actions": {}}
    if warn:
        k, ta = warn
        res["actions"]["no action"] = {"predicted_peak": fc_at(L, base, k).peak_rho,
                                       "simulated_peak": float(rho[k:k + 300].max())}
        for name, a in actions(L).items():
            h = {"t": [], "n": [], "out": []}
            record(tw.Venue(seed=seed, layout=L), ta + 300, h, act_at=ta, action=a)
            r, tt = np.array(h["n"]) / L.zone_c_area, np.array(h["t"])
            res["actions"][name] = {"predicted_peak": fc_at(L, base, k, action=a).peak_rho,
                                    "simulated_peak": float(r[tt >= ta].max())}
    res["wall_s"] = time.perf_counter() - t0
    return res


def score(out: dict, ref: dict) -> dict:
    """The action choice, plain and priced with the reference venue's measured errors."""
    ref_acts = [v["actions"] for v in ref["seeds"].values() if v["actions"]]
    err = {n: max(0.0, float(np.mean([a[n]["simulated_peak"] - a[n]["predicted_peak"] for a in ref_acts]))) for n in DISRUPTION}
    rows, lead, errs = [], [], []
    for name, seeds in out["venues"].items():
        for s, r in seeds.items():
            if r["lead_time_s"] is not None:
                lead.append(r["lead_time_s"])
            errs += r["abs_err_s_1_to_5_min_ahead"]
            acts = r["actions"]
            if not acts:
                continue
            best = min(acts[n]["simulated_peak"] for n in DISRUPTION)
            plain = choose({n: acts[n]["predicted_peak"] for n in DISRUPTION})
            priced = choose({n: acts[n]["predicted_peak"] + err[n] for n in DISRUPTION})
            rows.append({"venue": name, "seed": s, "best": min(DISRUPTION, key=lambda n: acts[n]["simulated_peak"]),
                         "chosen_plain": plain, "regret_plain": round(acts[plain]["simulated_peak"] - best, 2),
                         "chosen_with_error": priced, "regret_with_error": round(acts[priced]["simulated_peak"] - best, 2)})
    runs = [r for seeds in out["venues"].values() for r in seeds.values()]
    return {
        "runs": len(runs), "crossed": sum(r["cross_s"] is not None for r in runs),
        "warned_before_crossing": sum(r["lead_time_s"] is not None and r["lead_time_s"] > 0 for r in runs),
        "lead_time_s_min_max": [min(lead), max(lead)] if lead else None,
        "lead_time_s_median": float(np.median(lead)) if lead else None,
        "median_abs_err_s_1_to_5_min_ahead": float(np.median(errs)) if errs else None,
        "error_priced_from_reference": {n: round(e, 2) for n, e in err.items()},
        "choice": rows,
        "mean_regret_plain": round(float(np.mean([r["regret_plain"] for r in rows])), 2) if rows else None,
        "mean_regret_with_error": round(float(np.mean([r["regret_with_error"] for r in rows])), 2) if rows else None,
        "regret_with_error_under_0.25": sum(r["regret_with_error"] < 0.25 for r in rows),
    }


def main() -> dict:
    ref = json.loads((OUT / "test_b_twin.json").read_text())
    path = OUT / "test_b_venues.json"
    out = {"threshold": THRESHOLD, "layouts": {n: {"length": L.length, "width": L.width, "exit_width": L.exit_width,
                                                   "door_width": L.door_width, "demand": round(L.demand, 3)}
                                               for n, L in LAYOUTS.items()}, "venues": {}}
    for name, L in LAYOUTS.items():
        out["venues"][name] = {}
        for seed in SEEDS:
            r = run(L, seed)
            out["venues"][name][str(seed)] = r
            print(f"{name:18s} seed {seed}: crossing {r['cross_s']}, warning {r['warning_s']}, lead {r['lead_time_s']}, "
                  f"peak {r['peak_rho']:.2f}, {r['wall_s']:.0f} s", flush=True)
            for a, x in r["actions"].items():
                print(f"   {a:30s} predicted {x['predicted_peak']:.2f}  simulated {x['simulated_peak']:.2f}", flush=True)
            path.write_text(json.dumps(out))  # partial results survive an interruption
    out["summary"] = score(out, ref)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out["summary"].items() if k != "choice"}, indent=1))
    return out


if __name__ == "__main__":
    main()
