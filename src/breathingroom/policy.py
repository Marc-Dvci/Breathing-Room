"""Deterministic alert tiers. The tier is computed here, from measured numbers only; the
agent investigates and proposes, and it cannot lower a tier.

Thresholds (people/m^2) follow the crowd-safety literature: about 2/m^2 is where free
movement ends, 4/m^2 is where contact begins and the Green Guide's holding density is
exceeded, and above 5/m^2 crowds can no longer manage their own movement (Fruin level F,
Still 2014). The pressure threshold is set on the pressure index's own scale, from the
Jülich entrance runs (eval/results/test_a_threshold.json)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    NORMAL = 0
    WATCH = 1
    WARN = 2
    ACT = 3
    CRITICAL = 4


@dataclass
class Thresholds:
    watch_rho: float = 2.0
    warn_rho: float = 3.5
    act_rho: float = 4.5
    critical_rho: float = 5.5
    # 1/s^2, on this system's scale (R = 0.7 m kernel, 2 s window). Set on Jülich runs 030/040/050
    # (Youden point against the pushing ratings); on runs 060/150/160 it flags 90% of pushing
    # person-seconds and 31% of walking ones. eval/results/test_a_threshold.json
    warn_pressure: float = 0.11
    warn_minutes: float = 10.0  # minutes to act_rho at the current trend
    act_minutes: float = 5.0
    clear_seconds: float = 30.0  # a tier only drops after its condition is clear this long


@dataclass
class ZoneReading:
    zone: str
    t: float
    rho: float
    pressure: float
    minutes_to_act: float | None
    trusted: bool = True
    trust_reason: str = ""


@dataclass
class Decision:
    zone: str
    t: float
    tier: Tier
    reasons: list[str] = field(default_factory=list)
    trusted: bool = True

    def to_json(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.name
        return d


class TierPolicy:
    def __init__(self, th: Thresholds | None = None):
        self.th = th or Thresholds()
        self._held: dict[str, tuple[Tier, float]] = {}

    def raw(self, r: ZoneReading) -> tuple[Tier, list[str]]:
        th = self.th
        tier, why = Tier.NORMAL, []

        def up(t: Tier, msg: str):
            nonlocal tier
            why.append(msg)
            tier = max(tier, t)

        if r.rho >= th.critical_rho:
            up(Tier.CRITICAL, f"density {r.rho:.1f}/m2 >= {th.critical_rho}")
        elif r.rho >= th.act_rho:
            up(Tier.ACT, f"density {r.rho:.1f}/m2 >= {th.act_rho}")
        elif r.rho >= th.warn_rho:
            up(Tier.WARN, f"density {r.rho:.1f}/m2 >= {th.warn_rho}")
        elif r.rho >= th.watch_rho:
            up(Tier.WATCH, f"density {r.rho:.1f}/m2 >= {th.watch_rho}")
        if r.pressure >= th.warn_pressure and r.rho >= th.watch_rho:
            up(Tier.WARN, f"pushing index {r.pressure:.3f}/s2 >= {th.warn_pressure}")
            if r.rho >= th.warn_rho:
                up(Tier.ACT, "pushing in a dense zone")
        if r.minutes_to_act is not None:
            if r.minutes_to_act <= th.act_minutes:
                up(Tier.ACT, f"{r.minutes_to_act:.1f} min to {th.act_rho}/m2 at this inflow")
            elif r.minutes_to_act <= th.warn_minutes:
                up(Tier.WARN, f"{r.minutes_to_act:.1f} min to {th.act_rho}/m2 at this inflow")
        return tier, why

    def decide(self, r: ZoneReading) -> Decision:
        """Raw tier with hysteresis: a tier drops only after its condition has been clear
        for `clear_seconds`."""
        tier, why = self.raw(r)
        held = self._held.get(r.zone)  # (tier, last time the raw tier reached it)
        if held is None or tier >= held[0]:
            self._held[r.zone] = (tier, r.t)
        elif r.t - held[1] < self.th.clear_seconds:
            why.append(f"holding {held[0].name}: clear for {r.t - held[1]:.0f} of {self.th.clear_seconds:.0f} s")
            tier = held[0]
        else:
            self._held[r.zone] = (tier, r.t)
        if not r.trusted:
            why.append(f"camera not trusted: {r.trust_reason}")
        return Decision(r.zone, r.t, tier, why, r.trusted)
