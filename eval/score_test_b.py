"""Score the action choice of test B with and without the measured forecast error.

For each seed, the error of each action's forecast is estimated on the other seeds only
(leave one seed out), the planner ranks the actions on predicted peak + that error, and the
regret is the simulated peak of the chosen action minus the best simulated peak.

    python eval/score_test_b.py      # after eval/test_b_twin.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "eval" / "results"
DISRUPTION = {"open second exit door": 0, "meter both gates to 150/min": 1, "meter gate A to 60/min": 1, "hold both gates": 2}


def choose(peak: dict[str, float]) -> str:
    """The planner's rule: the least disruptive action within 0.25/m2 of the lowest peak."""
    lowest = min(peak.values())
    return min((n for n in peak if peak[n] <= lowest + 0.25), key=lambda n: (DISRUPTION[n], peak[n]))


def main() -> None:
    d = json.loads((RES / "test_b_twin.json").read_text())
    seeds = {k: v["actions"] for k, v in d["seeds"].items() if v["actions"]}
    rows = []
    for s, acts in seeds.items():
        others = [a for k, a in seeds.items() if k != s]
        err = {n: max(0.0, float(np.mean([o[n]["simulated_peak"] - o[n]["predicted_peak"] for o in others]))) for n in DISRUPTION}
        best = min(acts[n]["simulated_peak"] for n in DISRUPTION)
        plain = choose({n: acts[n]["predicted_peak"] for n in DISRUPTION})
        priced = choose({n: acts[n]["predicted_peak"] + err[n] for n in DISRUPTION})
        rows.append({"seed": s, "chosen_plain": plain, "regret_plain": round(acts[plain]["simulated_peak"] - best, 2),
                     "chosen_with_error": priced, "regret_with_error": round(acts[priced]["simulated_peak"] - best, 2)})
        print(rows[-1])
    out = {"rows": rows, "mean_regret_plain": round(float(np.mean([r["regret_plain"] for r in rows])), 2),
           "mean_regret_with_error": round(float(np.mean([r["regret_with_error"] for r in rows])), 2)}
    (RES / "test_b_choice.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
