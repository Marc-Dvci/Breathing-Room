"""Append-only, hash-chained trace of everything the system measured, decided and did.

Every entry carries the SHA-256 of the previous entry, so a deleted or edited line breaks the
chain. A hash chain alone does not stop someone who can rewrite the whole file: they can
recompute every hash. Two things do:

- Each entry is signed with an Ed25519 key (BR_TRACE_KEY, a PEM file) that the console reads
  but cannot write. The public key is in the first entry of every run and at /api/trace/key, so
  a copy of the trace can be checked by anyone, without access to AWS.
- Each entry also goes to CloudWatch Logs (BR_CLOUDWATCH_GROUP). The instance's role may only
  append to that log group, not delete or change events, so the CloudWatch copy is a witness
  the instance cannot rewrite. `compare_cloudwatch` checks the local file against it.

The signature sits outside the hashed body, so the chain is the same with or without it."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path

CW_CACHE_S = 60.0  # the report may be reloaded often; CloudWatch is read at most once a minute
CW_LAG_S = 120.0  # entries this recent may not be readable from CloudWatch yet


def _body(entry: dict) -> str:
    return json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))


def load_signing_key(path: str | Path | None):
    if not path or not Path(path).exists():
        return None
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    return load_pem_private_key(Path(path).read_bytes(), password=None)


def new_signing_key(path: str | Path) -> None:
    """Write a new Ed25519 private key (PEM) to `path`."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    pem = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                     serialization.NoEncryption())
    Path(path).write_bytes(pem)


def _raw_public(key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class Trace:
    def __init__(self, path: str | Path, cloudwatch_group: str | None = None, stream: str | None = None,
                 signing_key: str | Path | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev = self._last_hash()
        self._key = load_signing_key(signing_key or os.environ.get("BR_TRACE_KEY"))
        self._cw = None
        self._cw_cache: tuple[float, tuple, dict] | None = None
        self.mirror_failures = 0
        group = cloudwatch_group or os.environ.get("BR_CLOUDWATCH_GROUP")
        if group:
            try:
                import boto3

                self._cw = boto3.client("logs")
                self._group = group
                self._stream = stream or os.environ.get("BR_CLOUDWATCH_STREAM", "console")
                try:
                    self._cw.create_log_stream(logGroupName=group, logStreamName=self._stream)
                except self._cw.exceptions.ResourceAlreadyExistsException:
                    pass
            except Exception:  # CloudWatch is optional; the local chain is the record
                self._cw = None

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        last = None
        with open(self.path, "rb") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else "0" * 64

    def public_key(self) -> dict | None:
        if self._key is None:
            return None
        raw = _raw_public(self._key)
        return {"alg": "Ed25519", "public_key": base64.b64encode(raw).decode(),
                "fingerprint": hashlib.sha256(raw).hexdigest()[:16]}

    def write(self, kind: str, **data) -> dict:
        with self._lock:
            entry = {"ts": round(time.time(), 3), "kind": kind, **data, "prev": self._prev}
            entry["hash"] = hashlib.sha256(_body(entry).encode()).hexdigest()
            if self._key is not None:
                entry["sig"] = base64.b64encode(self._key.sign(entry["hash"].encode())).decode()
            line = _body(entry)
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            self._prev = entry["hash"]
        if self._cw is not None:
            try:
                self._cw.put_log_events(logGroupName=self._group, logStreamName=self._stream,
                                        logEvents=[{"timestamp": int(entry["ts"] * 1000), "message": line}])
            except Exception:
                self.mirror_failures += 1
        return entry

    def tail(self, n: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]


def verify(path: str | Path) -> tuple[bool, int, str]:
    """Re-hash the chain. Returns (ok, entries checked, message)."""
    prev = "0" * 64
    n = 0
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        e = json.loads(line)
        h = e.pop("hash")
        e.pop("sig", None)
        if e.get("prev") != prev:
            return False, n, f"entry {n}: previous-hash mismatch"
        if hashlib.sha256(_body(e).encode()).hexdigest() != h:
            return False, n, f"entry {n}: content does not match its hash"
        prev = h
    return True, n, "chain intact"


def verify_signatures(path: str | Path, public_key_b64: str | None) -> dict:
    """Check every entry's signature against one public key. Entries written before the key was
    configured carry no signature and are counted, not failed."""
    if not public_key_b64:
        return {"checked": False, "message": "no signing key configured"}
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    raw = base64.b64decode(public_key_b64)
    pub = Ed25519PublicKey.from_public_bytes(raw)
    fp = hashlib.sha256(raw).hexdigest()[:16]
    signed = unsigned = 0
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        e = json.loads(line)
        if "sig" not in e:
            unsigned += 1
            continue
        try:
            pub.verify(base64.b64decode(e["sig"]), e["hash"].encode())
        except InvalidSignature:
            return {"checked": True, "ok": False, "signed": signed, "unsigned": unsigned, "fingerprint": fp,
                    "message": f"entry {n}: signature does not match key {fp}"}
        signed += 1
    return {"checked": True, "ok": True, "signed": signed, "unsigned": unsigned, "fingerprint": fp,
            "message": f"{signed} entries signed by key {fp}" + (f"; {unsigned} written before signing" if unsigned else "")}


def compare_cloudwatch(trace: Trace, since_ts: float | None = None, client=None, lag_s: float = CW_LAG_S) -> dict:
    """Compare the local entries written since `since_ts` with the CloudWatch copy. A local entry
    that CloudWatch does not hold, or holds with different content, is reported: it was either
    never mirrored (the writer counts those failures) or changed afterwards."""
    cw = client or trace._cw
    if cw is None:
        return {"checked": False, "message": "no CloudWatch copy configured"}
    entries = [json.loads(x) for x in trace.path.read_text(encoding="utf-8").splitlines() if x.strip()] \
        if trace.path.exists() else []
    if since_ts is not None:
        entries = [e for e in entries if e["ts"] >= since_ts]
    if not entries:
        return {"checked": True, "ok": True, "local": 0, "matched": 0, "message": "no entries to compare"}
    key = (trace.path.stat().st_size, entries[0]["ts"])
    if trace._cw_cache and trace._cw_cache[1] == key and time.time() - trace._cw_cache[0] < CW_CACHE_S:
        return trace._cw_cache[2]
    try:
        remote: dict[str, str] = {}
        kw = {"logGroupName": trace._group, "logStreamName": trace._stream, "startFromHead": True,
              "startTime": int(entries[0]["ts"] * 1000)}
        token = None
        while True:
            r = cw.get_log_events(**kw, **({"nextToken": token} if token else {}))
            for ev in r.get("events", []):
                try:
                    remote[json.loads(ev["message"])["hash"]] = ev["message"]
                except (ValueError, KeyError):
                    continue
            if r.get("nextForwardToken") in (None, token):
                break
            token = r["nextForwardToken"]
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "message": f"CloudWatch not readable: {type(exc).__name__}"}
    state = []
    for e in entries:
        got = remote.get(e["hash"])
        state.append("match" if got == _body(e) else ("missing" if got is None else "different"))
    # An entry CloudWatch never received (the process stopped between the local write and the copy)
    # is still fixed by the witness when CloudWatch holds the entry after it: that entry carries its
    # hash. A rewrite changes every hash from the edit to the end, so it cannot hide this way.
    witnessed = [False] * len(entries)
    for i in range(len(entries) - 1, -1, -1):
        if state[i] == "match":
            witnessed[i] = True
        elif state[i] == "missing" and i + 1 < len(entries) and witnessed[i + 1]:
            e = {k: v for k, v in entries[i].items() if k not in ("hash", "sig")}
            witnessed[i] = (entries[i + 1].get("prev") == entries[i]["hash"]
                            and hashlib.sha256(_body(e).encode()).hexdigest() == entries[i]["hash"])
    now = time.time()
    matched = state.count("match")
    covered = sum(w and s != "match" for w, s in zip(witnessed, state))
    pending = sum(not w and s == "missing" and now - e["ts"] < lag_s for w, s, e in zip(witnessed, state, entries))
    bad = [e["hash"][:12] for w, s, e in zip(witnessed, state, entries)
           if not w and not (s == "missing" and now - e["ts"] < lag_s)]
    msg = f"{matched} of {len(entries)} entries match the CloudWatch copy"
    if covered:
        msg += f"; {covered} never copied (written just before a restart) are fixed by the entry after them"
    if pending:
        msg += f"; the newest {pending} are not in CloudWatch yet"
    if bad:
        msg += f"; {len(bad)} do not match, first at {bad[0]}"
    out = {"checked": True, "ok": not bad, "local": len(entries), "matched": matched, "covered_by_next": covered,
           "not_yet_in_cloudwatch": pending, "unwitnessed": len(bad),
           "missing_in_cloudwatch": state.count("missing"), "different_in_cloudwatch": state.count("different"),
           "mirror_failures_since_start": trace.mirror_failures, "message": msg}
    trace._cw_cache = (time.time(), key, out)
    return out
