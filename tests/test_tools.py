import json

import cv2
import numpy as np

from breathingroom import mcp, report
from breathingroom.drift import ShiftScreen
from breathingroom.runtime import CATALOGUE, Runtime, zone_crop
from breathingroom.trace import Trace


def textured(seed=0, size=(480, 640)):
    rs = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rs.integers(0, 255, size + (3,), dtype=np.uint8), (0, 0), 3)
    for _ in range(40):  # barriers and markings: strong straight edges
        x, y = rs.integers(0, size[1]), rs.integers(0, size[0])
        cv2.line(img, (int(x), int(y)), (int(x + rs.integers(-200, 200)), int(y + rs.integers(-200, 200))), (255, 255, 255), 3)
    return img


def test_ecc_screen_measures_a_knock_and_passes_a_steady_camera():
    ref = textured()
    to_floor = np.diag([0.01, 0.01, 1.0])  # 1 px = 1 cm
    probe = np.array([[100, 100], [540, 100], [540, 380], [100, 380]], float)
    sc = ShiftScreen([ref] * 5, to_floor, probe)
    steady = sc.check(ref)
    assert steady.verdict == "steady" and steady.floor_shift_m < 0.01
    M = cv2.getRotationMatrix2D((320, 240), 3.0, 1.0)
    M[:, 2] += (15, -10)
    knocked = sc.check(cv2.warpAffine(ref, M, (640, 480), borderMode=cv2.BORDER_REFLECT))
    true = np.linalg.norm(cv2.transform(probe.reshape(-1, 1, 2), M).reshape(-1, 2) - probe, axis=1).max() * 0.01
    assert knocked.verdict == "moved" and abs(knocked.floor_shift_m - true) < 0.02


def test_zone_crop_covers_the_zone_at_the_requested_scale():
    und = np.zeros((1000, 2000, 3), np.uint8)
    H = np.array([[100.0, 0, 1000], [0, 100.0, 500], [0, 0, 1]])  # 1 m = 100 px
    crop, mask = zone_crop(und, H, (-2, 2, -1, 1), scale=0.5)
    assert crop.shape[:2] == mask.shape[:2] == (96, 192)  # 200 x 400 px, halved, to multiples of 16
    assert 0.9 < mask.mean() <= 1.0


def test_mcp_lists_and_calls_read_only_tools(tmp_path):
    rt = Runtime(tmp_path / "t.jsonl", use_model=False, camera=None, seed=1, twin_warmup_s=60)
    init = mcp.handle(rt, CATALOGUE, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                      "params": {"protocolVersion": "2025-06-18", "capabilities": {}}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert mcp.handle(rt, CATALOGUE, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in mcp.handle(rt, CATALOGUE, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert {"get_zone_state", "locate_pushing", "forecast_action", "check_camera"} <= names
    assert not names & {"propose_action", "approve"}
    zone = rt.twin.name
    r = mcp.handle(rt, CATALOGUE, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "forecast_action", "arguments": {"zone": zone, "kind": "hold_entry"}}})["result"]
    assert not r["isError"] and "peak_density_10min" in json.loads(r["content"][0]["text"])
    bad = mcp.handle(rt, CATALOGUE, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                     "params": {"name": "forecast_action", "arguments": {"zone": zone, "kind": "teleport"}}})["result"]
    assert bad["isError"]


def test_evening_report_states_the_chain_and_each_decision(tmp_path):
    tr = Trace(tmp_path / "t.jsonl")
    tr.write("start", camera_run=None, use_model=False)
    tr.write("agent_start", episode="ep_1", zone="Zone C", tier="WARN", planner="deterministic")
    tr.write("proposal", id="pr_1", zone="Zone C", tier="WARN", radio="Zone C: meter both gates.", planner="deterministic",
             action={"kind": "meter_entry", "target": "both gates"}, predicted={"peak_density": 0.5}, alternatives=[], rationale="")
    tr.write("agent_end", episode="ep_1", proposal="pr_1")
    tr.write("decision", proposal="pr_1", zone="Zone C", approved=True, by="steward", applied="metered")
    tr.write("verify", proposal="pr_1", zone="Zone C", elapsed_s=90, predicted_rho=0.6, observed_rho=0.7, outcome="working as predicted")
    page = report.render(tmp_path / "t.jsonl")
    assert "Record intact" in page and "approved by steward" in page and "working as predicted" in page
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    (tmp_path / "t.jsonl").write_text("\n".join(lines[:2] + lines[3:]) + "\n")  # delete the proposal
    assert "Record BROKEN" in report.render(tmp_path / "t.jsonl")


def test_pushing_counts_where_the_crowd_is_packed():
    from breathingroom.policy import Tier, TierPolicy, ZoneReading

    pol = TierPolicy()
    sparse = pol.decide(ZoneReading("a", 0, 1.2, 0.3, None))  # pushing index high, nobody packed
    packed_spot = pol.decide(ZoneReading("b", 0, 1.2, 0.3, None, rho_peak=4.8))  # a dense spot at the gate
    assert sparse.tier == Tier.NORMAL
    assert packed_spot.tier == Tier.WARN and "4.8/m2" in packed_spot.reasons[0]
