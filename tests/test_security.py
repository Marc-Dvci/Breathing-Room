"""Who may decide, and what a rewrite of the whole trace runs into."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from breathingroom import api
from breathingroom.stewards import FAILS_PER_CLIENT, Stewards, hash_pin
from breathingroom.trace import Trace, _body, compare_cloudwatch, new_signing_key, verify, verify_signatures


def test_a_decision_is_recorded_under_the_steward_whose_pin_it_is():
    s = Stewards({"Ana": hash_pin("246810"), "Ben": hash_pin("135791")})
    assert s.identify("135791", "1.2.3.4") == "Ben"
    assert s.identify("000000", "1.2.3.4") is None
    with pytest.raises(ValueError):
        hash_pin("1234")  # too short to be a PIN


def test_guessing_pins_is_refused_after_a_few_tries():
    t = [0.0]
    s = Stewards({"Ana": hash_pin("246810", iterations=1000)}, clock=lambda: t[0])
    for _ in range(FAILS_PER_CLIENT):
        s.identify("111111", "9.9.9.9")
    assert s.locked("9.9.9.9") and not s.locked("8.8.8.8")
    t[0] += 301
    assert not s.locked("9.9.9.9")


def _request(peer="127.0.0.1", fwd="203.0.113.5"):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers={"x-forwarded-for": fwd})


def test_production_refuses_decisions_without_steward_accounts(monkeypatch):
    monkeypatch.setattr(api, "STEWARDS", Stewards({}))
    monkeypatch.setattr(api, "MODE", "production")
    with pytest.raises(HTTPException) as e:
        api._steward({"who": "anyone"}, _request())
    assert e.value.status_code == 403
    monkeypatch.setattr(api, "MODE", "demo")
    assert api._steward({"who": "steward"}, _request()) == ("steward", "none (public demo)")


def test_with_accounts_the_pin_names_the_steward_even_in_the_demo(monkeypatch):
    monkeypatch.setattr(api, "STEWARDS", Stewards({"Ana": hash_pin("246810", iterations=1000)}))
    monkeypatch.setattr(api, "MODE", "demo")
    assert api._steward({"pin": "246810", "who": "someone else"}, _request()) == ("Ana", "pin")
    with pytest.raises(HTTPException) as e:
        api._steward({"who": "Ana"}, _request())
    assert e.value.status_code == 403


def test_production_hides_frames_crops_state_record_and_report_until_a_steward_signs_in(monkeypatch):
    monkeypatch.setattr(api, "STEWARDS", Stewards({"Ana": hash_pin("246810", iterations=1000)}))
    monkeypatch.setattr(api, "MODE", "production")
    monkeypatch.setattr(api, "MCP_TOKEN", "")
    for path in ("/api/camera.jpg", "/api/evidence/ev_1.jpg", "/api/state", "/api/trace", "/report", "/api/trace/verify"):
        assert not api.may_read(path, None)
        assert not api.may_read(path, "forged.cookie")
    for path in ("/", "/static/app.js", "/health", "/api/session", "/api/mode"):
        assert api.may_read(path, None)  # the sign-in page and the health check
    good = api.session_cookie("Ana")
    assert api.session_name(good) == "Ana" and api.may_read("/api/camera.jpg", good)
    assert api.session_name(api.session_cookie("Ana", now=0)) is None  # expired
    assert api.session_name(api.session_cookie("Ben")) is None  # no such steward (or the account was removed)
    b64, sig = good.split(".")
    assert api.session_name(b64 + "." + "0" * len(sig)) is None
    # the MCP endpoint needs its own token, and stays closed without one
    assert not api.may_read("/mcp", good)
    monkeypatch.setattr(api, "MCP_TOKEN", "s3cret-token")
    assert api.may_read("/mcp", None, "Bearer s3cret-token") and not api.may_read("/mcp", None, "Bearer guess")
    monkeypatch.setattr(api, "MODE", "demo")
    assert api.may_read("/api/camera.jpg", None)  # the public demonstration, disclosed as such


def test_evidence_crops_expire_and_a_silent_zone_is_reported_and_not_investigated(tmp_path, monkeypatch):
    from breathingroom import runtime as rtm

    cam = SimpleNamespace(evidence={"ev_old": (0.0, b"jpg"), "ev_new": (rtm.time.time(), b"jpg")}, frame_lock=rtm.threading.Lock())
    assert rtm.CameraZone.evidence_image(cam, "ev_new") == b"jpg"
    assert rtm.CameraZone.evidence_image(cam, "ev_old") is None and "ev_old" not in cam.evidence
    rt = rtm.Runtime(tmp_path / "t.jsonl", use_model=False, camera=None, seed=1, twin_warmup_s=60)
    rt.running = True
    zone = rt.twin.name
    rt._feed(zone, True)
    assert rt.health()["ok"]
    monkeypatch.setattr(rtm, "FEED_STALE_S", 0.0)
    rt._feed(zone, False, "OSError: camera unreachable")
    h = rt.health()
    assert not h["ok"] and h["zones"][zone].startswith("lost")
    assert rt.tools_for(zone)["get_zone_state"](zone)["feed"].startswith("lost")
    assert not rt.twin.state.trusted or "feed lost" in " ".join(rt.twin.state.reasons)
    assert zone not in rt.active_episode
    monkeypatch.setattr(rtm, "FEED_STALE_S", 10.0)
    rt._feed(zone, True)
    assert rt.health()["ok"]


def _rewrite_without_key(path, change):
    """What someone who can rewrite the file but has no key can do: edit, then re-hash the chain."""
    prev, out = "0" * 64, []
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        e.pop("hash"), e.pop("sig", None)
        change(e)
        e["prev"] = prev
        e["hash"] = prev = hashlib.sha256(_body(e).encode()).hexdigest()
        out.append(_body(e))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


class FakeLogs:
    def __init__(self):
        self.events = []

    def put_log_events(self, logEvents, **_):
        self.events += logEvents

    def get_log_events(self, startTime=0, nextToken=None, **_):
        return {"events": [e for e in self.events if e["timestamp"] >= startTime], "nextForwardToken": "end"}


def test_a_rewritten_trace_re_hashes_but_fails_its_signatures_and_the_cloudwatch_copy(tmp_path):
    key = tmp_path / "k.pem"
    new_signing_key(key)
    tr = Trace(tmp_path / "t.jsonl", signing_key=key)
    tr._cw, tr._group, tr._stream = FakeLogs(), "g", "s"
    tr.write("start")
    tr.write("decision", proposal="pr_1", approved=False, by="Ana", auth="pin")
    pub = tr.public_key()["public_key"]
    assert verify(tr.path)[0]
    assert verify_signatures(tr.path, pub)["ok"]
    assert compare_cloudwatch(tr, lag_s=0)["ok"]

    _rewrite_without_key(tr.path, lambda e: e.update(approved=True) if e["kind"] == "decision" else None)
    tr._cw_cache = None
    assert verify(tr.path)[0]  # the hash chain alone cannot tell
    sig = verify_signatures(tr.path, pub)
    assert sig["signed"] == 0 and sig["unsigned"] == 2  # the forger cannot sign
    cw = compare_cloudwatch(tr, lag_s=0)
    assert not cw["ok"] and cw["matched"] == 0 and cw["unwitnessed"] == 2
    # the edited decision has a new hash; the untouched start entry lost its signature
    assert cw["missing_in_cloudwatch"] == 1 and cw["different_in_cloudwatch"] == 1


def test_an_entry_lost_before_its_copy_is_still_fixed_by_the_next_one(tmp_path):
    tr = Trace(tmp_path / "t.jsonl")
    logs = FakeLogs()
    tr._cw, tr._group, tr._stream = logs, "g", "s"
    tr.write("tool_call", tool="check_camera")
    tr.write("error", where="camera")
    logs.events.pop()  # the process stopped between the local write and the copy
    tr.write("start")
    cw = compare_cloudwatch(tr, lag_s=0)
    assert cw["ok"] and cw["matched"] == 2 and cw["covered_by_next"] == 1
    # the same gap at the end of the file is not covered: only lag excuses it, and only briefly
    logs.events.pop()
    tr._cw_cache = None
    assert not compare_cloudwatch(tr, lag_s=0)["ok"]
    tr._cw_cache = None
    late = compare_cloudwatch(tr, lag_s=120)
    assert late["ok"] and late["not_yet_in_cloudwatch"] == 2
