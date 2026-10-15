"""Append-only, hash-chained trace of everything the system measured, decided and did.

Every entry carries the SHA-256 of the previous entry, so a deleted or edited line breaks the
chain. Entries also go to CloudWatch Logs when BR_CLOUDWATCH_GROUP is set."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path


class Trace:
    def __init__(self, path: str | Path, cloudwatch_group: str | None = None, stream: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev = self._last_hash()
        self._cw = None
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

    def write(self, kind: str, **data) -> dict:
        with self._lock:
            entry = {"ts": round(time.time(), 3), "kind": kind, **data, "prev": self._prev}
            body = json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))
            entry["hash"] = hashlib.sha256(body.encode()).hexdigest()
            line = json.dumps(entry, sort_keys=True, default=str, separators=(",", ":"))
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
            self._prev = entry["hash"]
        if self._cw is not None:
            try:
                self._cw.put_log_events(logGroupName=self._group, logStreamName=self._stream,
                                        logEvents=[{"timestamp": int(entry["ts"] * 1000), "message": line}])
            except Exception:
                pass
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
        if e.get("prev") != prev:
            return False, n, f"entry {n}: previous-hash mismatch"
        body = json.dumps(e, sort_keys=True, default=str, separators=(",", ":"))
        if hashlib.sha256(body.encode()).hexdigest() != h:
            return False, n, f"entry {n}: content does not match its hash"
        prev = h
    return True, n, "chain intact"
