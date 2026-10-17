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
    assert compare_cloudwatch(tr)["ok"]

    _rewrite_without_key(tr.path, lambda e: e.update(approved=True) if e["kind"] == "decision" else None)
    tr._cw_cache = None
    assert verify(tr.path)[0]  # the hash chain alone cannot tell
    sig = verify_signatures(tr.path, pub)
    assert sig["signed"] == 0 and sig["unsigned"] == 2  # the forger cannot sign
    cw = compare_cloudwatch(tr)
    assert not cw["ok"] and cw["matched"] == 0
    # the edited decision has a new hash; the untouched start entry lost its signature
    assert cw["missing_in_cloudwatch"] == 1 and cw["different_in_cloudwatch"] == 1
