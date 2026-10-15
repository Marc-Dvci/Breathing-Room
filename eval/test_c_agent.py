"""Test C: the agent loop, scored on fixed scenarios built from the real pipeline.

Each scenario freezes a zone in a state the tier policy flags, then runs a full episode with
gpt-oss-120b on the Bedrock API endpoint, `--repeats` times:

  twin_filling     Zone C of the venue twin at WARN, filling towards 4.5/m2
  camera_knocked   Entrance E1 replay with the camera knocked (3 degrees, 40 px)
  e1_pushing       Entrance E1 replay with pushing while the density falls
  model_down       twin_filling with an unreachable model endpoint

Scored per episode: finished with a decision; the correct kind of decision (a proposal for
crowd states, a recalibration request for the knocked camera); a proposal cites a forecast of
its own episode; the chosen action's predicted peak is within 0.1/m2 of the lowest compared;
camera checked before proposing; refused tool calls (guards that fired); steps and seconds.

    AWS_PROFILE=otl-agent python eval/test_c_agent.py --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("BR_ISOLATE_MODELS", "1")  # each repeat builds a camera zone: keep the models out of this process
from breathingroom import agent as ag  # noqa: E402
from breathingroom.policy import Tier  # noqa: E402
from breathingroom.runtime import CATALOGUE, CameraZone, Runtime  # noqa: E402
from breathingroom.trace import Trace  # noqa: E402

DATA = Path(os.environ.get("BR_DATA", ROOT.parent / "data"))
OUT = ROOT / "eval" / "results"


def twin_runtime(trace: Trace) -> Runtime:
    rt = Runtime(OUT / "agent_trace.jsonl", use_model=True, camera=None, seed=3, twin_warmup_s=0)
    rt.trace = trace
    while True:  # run the evening until the policy says WARN
        rt.twin.tick(1.0)
        fc = rt.twin.forecast()
        mins = None if fc.seconds_to_threshold is None else fc.seconds_to_threshold / 60
        if mins is not None and mins <= 10 and rt.twin.state.t > 60:
            rt.twin.state.tier = "WARN"
            return rt


def camera_runtime(trace: Trace, knocked: bool) -> Runtime:
    cam = CameraZone(DATA / "fzj" / "videos" / "150_q_56_h0.mp4", ROOT / "eval" / "calib" / "fzj_2018crowdqueue.json", "150")
    rt = Runtime(OUT / "agent_trace.jsonl", use_model=True, camera=cam, seed=3, twin_warmup_s=0)
    rt.trace = trace
    for _ in range(30):  # 30 s of footage: the zone is past its first rush and pushing
        cam.tick(1.0)
    cam.bumped = knocked
    if knocked:
        cam.tick(1.0)
    cam.state.tier = "WARN"
    return rt


def score(ep: ag.Episode, expect: str) -> dict:
    tools = [c["tool"] for c in ep.calls]
    refused = sum(1 for c in ep.calls if isinstance(c["result"], dict) and "error" in c["result"])
    out = {"planner": ep.planner, "steps": len(ep.calls), "refused_calls": refused, "tools": tools,
           "decision": "proposal" if ep.proposal else "recalibration" if ep.recalibration else
           "no_action" if ep.closed_reason else "none"}
    out["finished"] = out["decision"] != "none"
    out["correct_decision"] = out["decision"] == expect
    out["checked_camera_first"] = "check_camera" in tools and (
        "propose_action" not in tools or tools.index("check_camera") < tools.index("propose_action"))
    if ep.proposal:
        p = ep.proposal
        peaks = [a["peak"] for a in p.alternatives if a["kind"] != "none"]
        out["cites_own_forecast"] = p.forecast_id in ep.forecasts
        out["chose_lowest_peak"] = p.predicted["peak_density"] <= min(peaks) + 0.1
        out["action"] = f"{p.action['kind']}:{p.action['target']}"
        out["radio"] = p.radio
        out["compared"] = len(peaks)
    return out


def run(name: str, make, expect: str, repeats: int, trace: Trace, model_down: bool = False) -> list[dict]:
    rows = []
    for i in range(repeats):
        rt = make()
        zone = CameraZone.name if rt.camera is not None else rt.twin.name
        ep = ag.Episode(zone, Tier.WARN, [f"test scenario {name}"], rt.tools_for(zone), CATALOGUE, trace,
                        clock=lambda: 0.0)
        t0 = time.perf_counter()
        if model_down:
            saved = ag.MANTLE_URL
            ag.MANTLE_URL = "https://127.0.0.1:9/v1"
            try:
                ag.investigate(ep, use_model=True)
            finally:
                ag.MANTLE_URL = saved
        else:
            ag.investigate(ep, use_model=True)
        r = {"scenario": name, "repeat": i, "seconds": round(time.perf_counter() - t0, 1), **score(ep, expect)}
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "tools"}), flush=True)
    return rows


def main() -> None:
    (OUT / "test_c_agent_trace.jsonl").unlink(missing_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    trace = Trace(OUT / "test_c_agent_trace.jsonl")
    rows = []
    rows += run("twin_filling", lambda: twin_runtime(trace), "proposal", args.repeats, trace)
    rows += run("camera_knocked", lambda: camera_runtime(trace, True), "recalibration", args.repeats, trace)
    rows += run("e1_pushing", lambda: camera_runtime(trace, False), "proposal", args.repeats, trace)
    rows += run("model_down", lambda: twin_runtime(trace), "proposal", min(2, args.repeats), trace, model_down=True)
    summary = {}
    for name in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == name]
        props = [r for r in rs if r["decision"] == "proposal"]
        summary[name] = {
            "episodes": len(rs),
            "planner": sorted({r["planner"] for r in rs}),
            "finished": sum(r["finished"] for r in rs),
            "correct_decision": sum(r["correct_decision"] for r in rs),
            "checked_camera_first": sum(r["checked_camera_first"] for r in rs),
            "proposals_citing_own_forecast": sum(r.get("cites_own_forecast", False) for r in props),
            "proposals_lowest_peak": sum(r.get("chose_lowest_peak", False) for r in props),
            "refused_calls_total": sum(r["refused_calls"] for r in rs),
            "median_steps": float(np.median([r["steps"] for r in rs])),
            "median_seconds": float(np.median([r["seconds"] for r in rs])),
            "actions": sorted({r.get("action", "") for r in props}),
        }
    (OUT / "test_c_agent.json").write_text(json.dumps({"model": ag.MODEL_ID, "summary": summary, "episodes": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
