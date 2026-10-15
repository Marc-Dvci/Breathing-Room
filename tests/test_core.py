import json

import cv2
import numpy as np
import pytest

from breathingroom.calib import HEAD_HEIGHT_M, FloorGrid, apply_h, calibrate, focal_from_floor
from breathingroom.datasets import fzj
from breathingroom.fields import FieldBuilder, sample
from breathingroom.physics import Action, forecast, level_of_service
from breathingroom.policy import Thresholds, Tier, TierPolicy, ZoneReading
from breathingroom.trace import Trace, verify


def synthetic_camera(f=1100.0, w=1920, h=1080, height=6.0, tilt_deg=50.0):
    """A camera `height` m above the floor, pitched down by `tilt_deg` from the horizon."""
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    a = np.radians(tilt_deg)
    # world: x right, y forward, z up. Camera looks along +y, pitched down.
    R = np.array([[1, 0, 0], [0, -np.sin(a), -np.cos(a)], [0, np.cos(a), -np.sin(a)]])
    C = np.array([0, -4.0, height])
    t = -R @ C
    return K, R, t.reshape(3, 1)


def project(K, R, t, X):
    P = (R @ X.T + t).T
    p = (K @ P.T).T
    return p[:, :2] / p[:, 2:]


def test_calibration_recovers_focal_height_and_head_plane():
    K, R, t = synthetic_camera()
    floor = np.array([[x, y] for x in (-3, -1, 1, 3) for y in (2, 5, 8)], float)
    img = project(K, R, t, np.c_[floor, np.zeros(len(floor))])
    cam = calibrate("synthetic", (1920, 1080), img, floor)
    assert cam.K[0, 0] == pytest.approx(1100, rel=0.02)
    assert cam.height_m == pytest.approx(6.0, rel=0.02)
    heads = np.array([[0.5, 4.0], [-2.0, 6.5], [2.5, 3.0]])
    himg = project(K, R, t, np.c_[heads, np.full(3, HEAD_HEIGHT_M)])
    back = apply_h(cam.H_head, himg)
    assert np.abs(back - heads).max() < 0.02
    # the floor homography misplaces the same heads by a lot on an oblique view
    assert cam.head_error_if_floor_used(himg).min() > 0.5


def test_focal_unobservable_overhead():
    K = np.array([[900, 0, 960], [0, 900, 540], [0, 0, 1.0]])
    R = np.diag([1.0, -1.0, -1.0])
    t = -R @ np.array([[0], [0], [8.0]])
    floor = np.array([[x, y] for x in (-2, 0, 2) for y in (-2, 0, 2)], float)
    img = project(K, R, t, np.c_[floor, np.zeros(len(floor))])
    H = cv2.findHomography(img, floor)[0]
    assert focal_from_floor(H, (960, 540)) is None


def crowd(spacing=0.5, extent=6.0, v=(1.0, 0.0), noise=0.0, seed=0):
    xs = np.arange(-extent / 2, extent / 2 + 1e-9, spacing)
    P = np.array([[x, y] for x in xs for y in xs])
    rs = np.random.default_rng(seed)
    V = np.tile(np.array(v, float), (len(P), 1)) + rs.normal(0, noise, P.shape)
    return P, V


def test_fields_density_and_pressure_on_a_known_crowd():
    grid = FloorGrid(-5, -5, 5, 5, cell=0.1)
    fb = FieldBuilder(grid, np.eye(3), (10, 10))
    P, V = crowd(spacing=0.5)  # 4 people per m2, all at 1 m/s
    fr = fb.build(0.0, P, np.ones(len(P)), V)
    rho_c = sample(grid, fr.rho, np.array([[0.0, 0.0]]))[0]
    assert rho_c == pytest.approx(4.0, rel=0.05)
    assert sample(grid, fr.pressure, np.array([[0.0, 0.0]]))[0] < 1e-4
    P, V = crowd(spacing=0.5, noise=0.3, seed=1)
    fr = fb.build(0.0, P, np.ones(len(P)), V)
    var_c = sample(grid, fr.var, np.array([[0.0, 0.0]]))[0]
    assert 0.08 < var_c < 0.25  # 2 * 0.3^2 = 0.18, minus finite-sample shrinkage


def test_forecast_time_to_threshold_and_actions():
    t = np.arange(0, 120.0)
    n = 100 + 1.0 * t  # +1 person/s
    out = np.full_like(t, 2.0)  # 2 persons/s leave
    fc = forecast("z", t, n, out, area_m2=100, exit_width_m=1.6, threshold=4.5, window_s=60)
    # 219 now, 450 at threshold: 231 s at +1/s, a little longer because the exits' outflow
    # rises with occupancy from 2.0 to their capacity of 1.3 x 1.6 = 2.08 persons/s
    assert 231 < fc.seconds_to_threshold < 260
    assert fc.inflow_pps == pytest.approx(3.0, rel=0.02)
    met = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60, action=Action("meter_entry", "g", rate_pps=1.5))
    assert met.net_rate_pps < 0 and met.seconds_to_threshold is None
    hold = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60, action=Action("hold_entry", "g"))
    assert hold.net_rate_pps == pytest.approx(-2.0, rel=0.05)
    assert level_of_service(0.5) == "A" and level_of_service(4.0) == "E" and level_of_service(6.0) == "F"


def test_policy_tiers_and_hysteresis():
    pol = TierPolicy(Thresholds(clear_seconds=30))
    d = pol.decide(ZoneReading("z", 0, 4.7, 0.0, None))
    assert d.tier == Tier.ACT
    d = pol.decide(ZoneReading("z", 10, 1.0, 0.0, None))
    assert d.tier == Tier.ACT  # held: not clear long enough
    d = pol.decide(ZoneReading("z", 45, 1.0, 0.0, None))
    assert d.tier == Tier.NORMAL
    d = pol.decide(ZoneReading("y", 0, 2.5, 0.0, 4.0))
    assert d.tier == Tier.ACT  # four minutes to the act density
    d = pol.decide(ZoneReading("x", 0, 3.6, 0.15, None))
    assert d.tier == Tier.ACT  # pushing in a dense zone


def test_trace_chain_detects_tampering(tmp_path):
    tr = Trace(tmp_path / "t.jsonl")
    for i in range(5):
        tr.write("event", i=i)
    assert verify(tmp_path / "t.jsonl")[0]
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    e = json.loads(lines[2])
    e["i"] = 99
    lines[2] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    (tmp_path / "t.jsonl").write_text("\n".join(lines) + "\n")
    ok, n, msg = verify(tmp_path / "t.jsonl")
    assert not ok and n == 3


def test_rating_parser(tmp_path):
    p = tmp_path / "r.txt"
    p.write_text("# personal information:\n# ID| Comment\n#  1|Frame 0: 2\n#   |Frame 50: 3\n#   |Frame 80: End\n"
                 "#  2|Frame 10: 4\n# id frame x/m y/m z/m\n1\t0\t0\t0\t1.7\n")
    r = fzj.load_ratings(p, 100)
    assert r[1][0] == 2 and r[1][49] == 2 and r[1][50] == 3 and r[1][80] == 0
    assert r[2][9] == 0 and r[2][10] == 4 and r[2][99] == 4
    assert fzj.parse_run_name("110_c_12_h0")["motivation"] == "high"
    assert fzj.bottleneck_width("4D030_w080_l200_h0.txt") == 0.8


def test_metering_one_gate_caps_only_its_share():
    t = np.arange(0, 120.0)
    n = 100 + 1.0 * t
    out = np.full_like(t, 2.0)  # inflow 3/s
    one = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60,
                   action=Action("meter_entry", "gate A", rate_pps=0.5, share=0.5))
    assert one.inflow_pps == pytest.approx(1.5 + 0.5, rel=0.02)  # gate B's half keeps coming
    both = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60,
                    action=Action("meter_entry", "both", rate_pps=0.5))
    assert both.inflow_pps == pytest.approx(0.5, rel=0.02)
    # the new exit scales the outflow by the added width (2.0 x 4.0 / 1.6), within capacity
    side = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60, action=Action("open_exit", "side", width_m=2.4))
    assert side.outflow_pps == pytest.approx(5.0, rel=0.02) and side.net_rate_pps < 0
    # an empty zone lets out what comes in
    empty = forecast("z", t, np.zeros_like(t), np.zeros_like(t), 100, 1.6, threshold=4.5, window_s=60)
    assert empty.seconds_to_threshold is None and empty.peak_rho == 0


def test_forecast_carries_a_rising_inflow_forward():
    t = np.arange(0, 120.0)
    out = np.full_like(t, 2.0)
    q_in = 2.0 + 0.01 * t  # arrivals rising by 0.01 person/s every second
    n = 100 + np.concatenate([[0], np.cumsum(q_in[1:] - out[1:])])
    fc = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60)
    assert fc.inflow_trend_pps2 == pytest.approx(0.01, rel=0.1)
    flat = forecast("z", t, n, out, 100, 1.6, threshold=4.5, window_s=60, trend_persist_s=0)
    assert fc.peak_rho > flat.peak_rho  # the trend fills the zone faster than a constant inflow
