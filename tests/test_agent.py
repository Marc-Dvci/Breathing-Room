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
