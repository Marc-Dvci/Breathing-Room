import numpy as np

from breathingroom.agent import Episode, VenueCatalogue, run_deterministic
from breathingroom.physics import forecast
from breathingroom.policy import Tier
from breathingroom.trace import Trace

CAT = VenueCatalogue({"Zone C": [
    {"kind": "meter_entry", "target": "gate A", "rate_per_min": 60, "camera": "cam-1"},
    {"kind": "open_exit", "target": "side", "width_m": 2.4},
    {"kind": "hold_entry", "target": "gates"},
]})


def tools(trusted=True):
    t = np.arange(0, 120.0)
    n = 200 + 1.2 * t
    out = np.full_like(t, 3.0)

    def _forecast(zone, action):
        return forecast(zone, t, n, out, 90, 2.4, threshold=4.5, action=action, window_s=45)

    return {
        "get_zone_state": lambda zone: {"density": 3.8, "zone": zone},
        "check_camera": lambda camera: {"trusted": trusted, "floor_shift_m": 0.0 if trusted else 0.6,
                                        "reason": "" if trusted else "camera moved"},
        "cross_check_count": lambda zone: {"agree": True},
        "measure_exit_flow": lambda zone: {"outflow_per_min": 180},
        "_forecast": _forecast,
    }


def episode(tmp_path, trusted=True):
    return Episode("Zone C", Tier.WARN, ["test"], tools(trusted), CAT, Trace(tmp_path / "t.jsonl"), clock=lambda: 0.0)


def test_deterministic_planner_proposes_lowest_peak_with_cited_forecast(tmp_path):
    ep = run_deterministic(episode(tmp_path))
    p = ep.proposal
    assert p is not None and p.forecast_id in ep.forecasts
    peaks = {a["kind"]: a["peak"] for a in p.alternatives}
    assert "none" in peaks  # the steward always sees the no-action baseline
    assert p.predicted["peak_density"] == min(v for k, v in peaks.items() if k != "none")
    assert [c["tool"] for c in ep.calls][:2] == ["get_zone_state", "check_camera"]


def test_untrusted_camera_asks_for_recalibration_and_proposes_nothing(tmp_path):
    ep = run_deterministic(episode(tmp_path, trusted=False))
    assert ep.proposal is None and ep.recalibration["camera"] == "cam-1"


def test_guards_refuse_uncited_or_unlisted_actions(tmp_path):
    ep = episode(tmp_path)
    assert "error" in ep.call("propose_action", {"zone": "Zone C", "forecast_id": "fc_made_up", "radio_message": "x", "rationale": ""})
    assert "error" in ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit", "target": "fire door"})
    ok = ep.call("forecast_action", {"zone": "Zone C", "kind": "meter_entry", "target": "gate A", "rate_per_min": 60})
    long = " ".join(["word"] * 60)
    assert "error" in ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"], "radio_message": long, "rationale": ""})
    assert ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"], "radio_message": "Zone C filling, meter gate A.", "rationale": "r"})["ok"]


def test_radio_numbers_must_come_from_tools(tmp_path):
    ep = episode(tmp_path)
    ep.call("get_zone_state", {"zone": "Zone C"})  # density 3.8
    ok = ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})  # target filled in: one open_exit
    assert "forecast_id" in ok and ep.forecast_actions[ok["forecast_id"]]["target"] == "side"
    bad = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"],
                                     "radio_message": "Zone C at 3.8 per square metre, stable for 10 minutes.", "rationale": "r"})
    assert "error" in bad and "10" in bad["error"]
    good = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"],
                                      "radio_message": "Zone C at 3.8 per square metre. Open the side passage.", "rationale": "r"})
    assert good.get("ok")


def test_approval_forecasts_the_action_again_and_applies_it(tmp_path):
    from breathingroom.agent import Episode as Ep
    from breathingroom.runtime import CATALOGUE, Runtime

    rt = Runtime(tmp_path / "t.jsonl", use_model=False, camera=None, seed=1, twin_warmup_s=120)
    zone = rt.twin.name
    ep = run_deterministic(Ep(zone, Tier.WARN, ["test"], rt.tools_for(zone), CATALOGUE, rt.trace, clock=lambda: rt.twin.state.t))
    rt.proposals[ep.proposal.id] = ep.proposal
    for _ in range(20):  # the zone moves on before the steward decides
        rt.twin.tick(1.0)
    out = rt.decide(ep.proposal.id, approve=True)
    p = rt.proposals[ep.proposal.id]
    assert out["ok"] and out["applied"]
    assert p.predicted["at_approval"] and p.predicted_at_proposal["peak_density"] is not None
    assert p.verify["start_t"] == rt.twin.state.t


def test_guard_refuses_a_more_disruptive_action_when_a_gentler_one_is_as_good(tmp_path):
    ep = episode(tmp_path)
    hold = ep.call("forecast_action", {"zone": "Zone C", "kind": "hold_entry", "target": "gates"})
    meter = ep.call("forecast_action", {"zone": "Zone C", "kind": "meter_entry", "target": "gate A", "rate_per_min": 1})
    assert abs(hold["peak_density_10min"] - meter["peak_density_10min"]) <= 0.25
    out = ep.call("propose_action", {"zone": "Zone C", "forecast_id": hold["forecast_id"], "radio_message": "Zone C. Hold gates.",
                                     "rationale": "r"})
    assert "error" in out and meter["forecast_id"] in out["error"]


def test_no_action_is_refused_while_the_zone_is_filling(tmp_path):
    from breathingroom.agent import Episode as Ep
    from breathingroom.runtime import CATALOGUE, Runtime

    rt = Runtime(tmp_path / "t.jsonl", use_model=False, camera=None, seed=1, twin_warmup_s=60)
    zone = rt.twin.name
    tools = rt.tools_for(zone)
    tools["get_zone_state"] = lambda zone: {"zone": zone, "density": 0.5, "minutes_to_4_5": 9.1}
    ep = Ep(zone, Tier.WARN, ["9.1 min to 4.5/m2"], tools, CATALOGUE, Trace(tmp_path / "e.jsonl"), clock=lambda: 0.0)
    ep.call("get_zone_state", {"zone": zone})
    out = ep.call("no_action", {"reason": "density low"})
    assert "error" in out and not ep.done
