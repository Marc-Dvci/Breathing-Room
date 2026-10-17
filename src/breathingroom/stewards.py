"""Steward accounts: who may approve or reject a proposal.

Each steward has a name and a PIN of at least 6 digits. The accounts file stores a salted
PBKDF2 hash of each PIN, never the PIN. A decision is recorded under the name whose PIN was
entered, so every approval in the trace has an accountable person. Wrong PINs are counted per
client and in total, and guessing is refused for a while once either count is reached.

    python -m breathingroom.stewards "Ana Ruiz" >> stewards.json-lines   # prompts for the PIN

then collect the lines into one JSON object, {"Ana Ruiz": "pbkdf2_sha256$...", ...}, and point
BR_STEWARDS_FILE at it.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

ITERATIONS = 200_000
MIN_PIN_DIGITS = 6
FAILS_PER_CLIENT = 5  # wrong PINs from one client within FAIL_WINDOW_S ...
FAILS_TOTAL = 30  # ... or from everyone together, before guessing is refused
FAIL_WINDOW_S = 300.0


def hash_pin(pin: str, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    if not (pin.isdigit() and len(pin) >= MIN_PIN_DIGITS):
        raise ValueError(f"a steward PIN is at least {MIN_PIN_DIGITS} digits")
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def _matches(pin: str, stored: str) -> bool:
    try:
        algo, it, salt, want = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    dk = hashlib.pbkdf2_hmac("sha256", pin.encode(), base64.b64decode(salt), int(it))
    return hmac.compare_digest(dk, base64.b64decode(want))


class Stewards:
    def __init__(self, accounts: dict[str, str] | None = None, clock=time.monotonic):
        self.accounts = dict(accounts or {})
        self.clock = clock
        self._fails: dict[str, deque] = defaultdict(deque)
        self._all_fails: deque = deque()

    @classmethod
    def from_env(cls) -> "Stewards":
        """BR_STEWARDS_FILE (JSON name -> PIN hash), plus BR_STEWARD_PIN as a single account
        named "steward" for an installation that has not set up named accounts yet."""
        accounts: dict[str, str] = {}
        f = os.environ.get("BR_STEWARDS_FILE")
        if f:
            accounts.update(json.loads(Path(f).read_text(encoding="utf-8")))
        pin = os.environ.get("BR_STEWARD_PIN")
        if pin:
            accounts.setdefault("steward", hash_pin(pin))
        return cls(accounts)

    def __bool__(self) -> bool:
        return bool(self.accounts)

    def _prune(self, q: deque, now: float) -> None:
        while q and now - q[0] > FAIL_WINDOW_S:
            q.popleft()

    def locked(self, client: str) -> bool:
        now = self.clock()
        self._prune(self._fails[client], now)
        self._prune(self._all_fails, now)
        return len(self._fails[client]) >= FAILS_PER_CLIENT or len(self._all_fails) >= FAILS_TOTAL

    def identify(self, pin: str, client: str) -> str | None:
        """The name of the steward whose PIN this is, or None (and the failure is counted).
        Every account is checked, so the time taken does not depend on which one matched."""
        found = None
        for name, stored in self.accounts.items():
            if _matches(str(pin), stored) and found is None:
                found = name
        if found is None:
            now = self.clock()
            self._fails[client].append(now)
            self._all_fails.append(now)
        return found


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    pin = getpass.getpass(f"PIN for {argv[0]} (at least {MIN_PIN_DIGITS} digits): ")
    if pin != getpass.getpass("again: "):
        print("the two PINs differ", file=sys.stderr)
        return 1
    print(json.dumps({argv[0]: hash_pin(pin)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
