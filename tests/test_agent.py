import numpy as np

from breathingroom import choice
from breathingroom.agent import Episode, VenueCatalogue, run_deterministic
from breathingroom.physics import forecast
from breathingroom.policy import Tier
from breathingroom.trace import Trace

CAT = VenueCatalogue({"Zone C": [
    {"kind": "meter_entry", "target": "gate A", "rate_per_min": 60, "camera": "cam-1", "holding_area_m2": 60},
    {"kind": "open_exit", "target": "side", "width_m": 2.4},
    {"kind": "hold_entry", "target": "gates", "holding_area_m2": 60},
]})


def tools(trusted=True, state=None):
    t = np.arange(0, 120.0)
    n = 200 + 1.2 * t
    out = np.full_like(t, 3.0)

    def _forecast(zone, action):
        return forecast(zone, t, n, out, 90, 2.4, threshold=4.5, action=action, window_s=45)

    return {
        "get_zone_state": lambda zone: state or {"density": 3.8, "zone": zone, "minutes_to_4_5": 2.6},
        "check_camera": lambda camera: {"trusted": trusted, "floor_shift_m": 0.0 if trusted else 0.6,
                                        "reason": "" if trusted else "camera moved"},
        "cross_check_count": lambda zone: {"agree": True, "count_primary": 342, "count_second": 330},
        "locate_pushing": lambda zone: {"hotspots": [{"side": "left", "metres_before_gate": 1.4, "density": 4.1}]},
        "measure_exit_flow": lambda zone: {"outflow_per_min": 180},
        "_forecast": _forecast,
    }


def episode(tmp_path, trusted=True, reasons=("test",), cat=CAT, **kw):
    return Episode("Zone C", Tier.WARN, list(reasons), tools(trusted, **kw), cat, Trace(tmp_path / "t.jsonl"), clock=lambda: 0.0)


def checked(ep):
    """The measurements every proposal needs."""
    ep.call("get_zone_state", {"zone": "Zone C"})
    ep.call("check_camera", {"camera": "cam-1"})
    ep.call("cross_check_count", {"zone": "Zone C"})
    return ep


def test_deterministic_planner_follows_the_venue_wide_rule(tmp_path):
    ep = run_deterministic(episode(tmp_path))
    p = ep.proposal
    assert p is not None and p.forecast_id in ep.forecasts
    kinds = {a["kind"] for a in p.alternatives}
    assert {"none", "meter_entry", "open_exit", "hold_entry"} <= kinds  # every action, and the no-action baseline
    # the zone is already at 3.8/m2, so no action is safe; the door leaves the densest place least dense,
    # where metering and holding would pack their 60 m2 waiting areas
    assert p.action["kind"] == "open_exit"
    assert not p.choice["safe"] and p.choice["held_outside_worst"] == 0 and "no action keeps" in p.choice["reason"]
    assert [c["tool"] for c in ep.calls][:2] == ["get_zone_state", "check_camera"]


def test_untrusted_camera_asks_for_recalibration_and_proposes_nothing(tmp_path):
    ep = run_deterministic(episode(tmp_path, trusted=False))
    assert ep.proposal is None and ep.recalibration["camera"] == "cam-1"


def test_a_proposal_needs_the_camera_checked_and_trusted_and_the_count_cross_checked(tmp_path):
    ep = episode(tmp_path)
    ok = ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})
    msg = {"zone": "Zone C", "forecast_id": ok["forecast_id"], "radio_message": "Zone C filling. Open the side door.",
           "rationale": "r"}
    out = ep.call("propose_action", msg)
    assert "check_camera" in out["error"] and "cross_check_count" in out["error"]
    ep.call("get_zone_state", {"zone": "Zone C"})
    ep.call("cross_check_count", {"zone": "Zone C"})
    assert "check_camera" in ep.call("propose_action", msg)["error"]
    # a knocked camera at its latest check blocks the proposal
    bad = checked(episode(tmp_path, trusted=False))
    fc = bad.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})
    assert "trusted camera" in bad.call("propose_action", {**msg, "forecast_id": fc["forecast_id"]})["error"]
    ep.call("check_camera", {"camera": "cam-1"})
    assert ep.call("propose_action", msg)["ok"]


def test_pushing_must_be_located_and_said(tmp_path):
    ep = checked(episode(tmp_path, reasons=["pushing index 0.2/s2 >= 0.11 where the crowd reaches 4.1/m2"]))
    fc = ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})
    msg = {"zone": "Zone C", "forecast_id": fc["forecast_id"], "radio_message": "Zone C at 3.8 per square metre. Open the side door.",
           "rationale": "r"}
    assert "locate_pushing" in ep.call("propose_action", msg)["error"]
    ep.call("locate_pushing", {"zone": "Zone C"})
    assert "pushing" in ep.call("propose_action", msg)["error"]
    msg["radio_message"] = "Zone C at 3.8 per square metre, pushing left side 1.4 m before the gate. Open the side door."
    assert ep.call("propose_action", msg)["ok"]


def test_guards_refuse_uncited_or_unlisted_actions_and_other_zones(tmp_path):
    ep = checked(episode(tmp_path))
    assert "error" in ep.call("propose_action", {"zone": "Zone C", "forecast_id": "fc_made_up", "radio_message": "x", "rationale": ""})
    assert "error" in ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit", "target": "fire door"})
    assert "error" in ep.call("forecast_action", {"zone": "Zone D", "kind": "open_exit"})
    ok = ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})
    assert "error" in ep.call("propose_action", {"zone": "Zone D", "forecast_id": ok["forecast_id"],
                                                 "radio_message": "Zone C filling, open the side door.", "rationale": ""})
    for words in (31, 35):  # the limit is the 30 words the prompt states
        long = "Zone C " + " ".join(["word"] * (words - 2))
        out = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"], "radio_message": long, "rationale": ""})
        assert "at most 30" in out["error"]
    out = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"], "radio_message": "Open the side door.",
                                     "rationale": ""})
    assert "name the zone" in out["error"]
    assert ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"],
                                      "radio_message": "Zone C filling, open the side door.", "rationale": "r"})["ok"]


def test_radio_numbers_must_come_from_tools_as_the_same_kind_of_number(tmp_path):
    ep = checked(episode(tmp_path))  # density 3.8, 2.6 minutes to 4.5, counts 342 and 330
    ok = ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})  # target filled in: one open_exit
    assert "forecast_id" in ok and ep.forecast_actions[ok["forecast_id"]]["target"] == "side"
    assert ep.ungrounded_numbers("Zone C at 3.8 per square metre, stable for 17 minutes.") == ["17 (minutes)"]
    # a number some tool returned, stated as another quantity, is refused
    assert ep.ungrounded_numbers("Zone C: 342 people per square metre") == ["342 (density)"]
    assert ep.ungrounded_numbers("Zone C: 3.8 minutes to 4.5") == ["3.8 (minutes to 4.5)"]
    assert ep.ungrounded_numbers("Zone C: 2.6 minutes to 4.5, 342 people, 3.8/m²") == []
    assert ep.ungrounded_numbers("Zone C: open the side door (2.4 m), 1.4 m before the gate") == ["1.4 (metres)"]
    bad = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"],
                                     "radio_message": "Zone C at 3.8 per square metre, stable for 17 minutes.", "rationale": "r"})
    assert "error" in bad and "17" in bad["error"]
    good = ep.call("propose_action", {"zone": "Zone C", "forecast_id": ok["forecast_id"],
                                      "radio_message": "Zone C at 3.8 per square metre, 2.6 minutes to 4.5. Open the side passage.",
                                      "rationale": "r"})
    assert good.get("ok")


def test_minutes_to_4_5_cannot_be_stated_when_the_zone_is_not_filling(tmp_path):
    ep = checked(episode(tmp_path, state={"zone": "Zone C", "density": 1.4, "minutes_to_4_5": None}))
    ep.call("forecast_action", {"zone": "Zone C", "kind": "open_exit"})  # its forecast has minutes of its own
    assert ep.ungrounded_numbers("Zone C: 0 minutes left") == ["0 (minutes to 4.5)"]


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


def test_guard_refuses_an_action_that_moves_the_crowd_into_the_waiting_area(tmp_path):
    ep = checked(episode(tmp_path))
    hold = ep.call("forecast_action", {"zone": "Zone C", "kind": "hold_entry", "target": "gates"})
    meter = ep.call("forecast_action", {"zone": "Zone C", "kind": "meter_entry", "target": "gate A", "rate_per_min": 1})
    assert hold["people_held_at_entrance_10min"] > 60 * 3.5 and not hold["safe"]
    out = ep.call("propose_action", {"zone": "Zone C", "forecast_id": meter["forecast_id"], "radio_message": "Zone C. Meter gate A.",
                                     "rationale": "r"})
    door = next(k for k, v in ep.forecast_actions.items() if v["kind"] == "open_exit")  # code forecast what the model skipped
    assert "error" in out and door in out["error"] and "held outside" in out["error"]
    assert ep.call("propose_action", {"zone": "Zone C", "forecast_id": door, "radio_message": "Zone C. Open the side door.",
                                      "rationale": "r"})["ok"]


def test_rule_holds_nobody_outside_when_it_can_and_never_prices_an_undeclared_queue():
    Opt = choice.Option
    door = Opt("door", "open_exit", peak=0.6, error_mean=1.9, error_max=2.0)
    meter = Opt("meter", "meter_entry", peak=0.5, error_mean=0.1, error_max=0.2, held=340, queue_factor=2.9, holding_area_m2=120)
    assert choice.choose([door, meter]).chosen is door  # meter would put 8.2/m2 in the waiting area
    roomy = Opt("meter", "meter_entry", peak=0.5, error_mean=0.1, error_max=0.2, held=340, queue_factor=2.9, holding_area_m2=2000)
    steep = Opt("door", "open_exit", peak=2.0, error_mean=1.9, error_max=2.0)
    assert choice.choose([steep, roomy]).chosen is roomy  # the door is not safe; a large waiting area is
    unknown = Opt("meter", "meter_entry", peak=0.5, held=340)
    assert not unknown.safe and unknown.queue_rho_worst == float("inf")
    assert choice.choose([steep, unknown]).chosen is steep  # nothing safe: the densest place is lowest with the door


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
