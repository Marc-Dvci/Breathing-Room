"""Which action to propose: the venue-wide rule.

Ranking actions on the monitored zone's peak alone moves danger out of sight: metering or
holding an entrance keeps the zone clear by keeping people waiting before the gates, where no
camera measures them. In the twin, metering both gates held a median of 912 people there
(eval/results/test_b_policy.json). So every action is judged on two places:

- the zone: its predicted peak plus the largest error that action's forecast made in testing;
- the waiting area before the entrance the action restricts: the people it holds back, times
  the largest factor by which the queue forecast fell short in testing, over the waiting area
  the venue declared for that entrance (people/m2).

An action is safe when both stay below SAFE_RHO (the WARN density). Among safe actions the
rule takes the one that holds the fewest people outside, then the least disruptive, then the
lowest zone peak. When no action is safe, it takes the one whose worst place (zone or waiting
area) is least dense, preferring the gentlest within 0.25/m2. An action that holds people
back where the venue has declared no waiting area is never safe: its queue cannot be priced.

The rule is shared by the deterministic planner, the guard on the model's proposals and the
evaluation (eval/test_b_policy.py), so all three judge actions the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from .policy import Thresholds

SAFE_RHO = Thresholds().warn_rho  # 3.5/m2: below it a zone or a waiting area is not even at WARN
NEAR_RHO = 0.25  # people/m2: densities this close count as equal
NEAR_HELD = 25.0  # people: queues this close count as equal
# how disruptive each kind of action is for the crowd
DISRUPTION = {"none": 0, "send_steward": 0, "announce": 0, "open_exit": 0, "meter_entry": 1, "hold_entry": 2}


@dataclass
class Option:
    key: str  # forecast id, or an action name in the evaluation
    kind: str
    peak: float  # predicted zone peak, people/m2
    error_mean: float = 0.0  # mean amount the zone peak exceeded this action's forecast in testing
    error_max: float = 0.0  # the largest amount
    held: float = 0.0  # predicted people held before the entrance
    queue_factor: float = 1.0  # largest ratio of simulated to predicted queue in testing
    holding_area_m2: float | None = None  # the waiting area the venue declared for that entrance

    @property
    def disruption(self) -> int:
        return DISRUPTION.get(self.kind, 3)

    @property
    def peak_ranked(self) -> float:
        return self.peak + self.error_mean

    @property
    def peak_worst(self) -> float:
        return self.peak + self.error_max

    @property
    def held_worst(self) -> float:
        return self.held * self.queue_factor

    @property
    def queue_rho_worst(self) -> float:
        if self.held_worst < 1.0:
            return 0.0
        if not self.holding_area_m2:
            return float("inf")
        return self.held_worst / self.holding_area_m2

    @property
    def venue_worst(self) -> float:
        return max(self.peak_worst, self.queue_rho_worst)

    @property
    def safe(self) -> bool:
        return self.peak_worst < SAFE_RHO and self.queue_rho_worst < SAFE_RHO

    def summary(self) -> dict:
        q = self.queue_rho_worst
        return {"zone_peak_worst": round(self.peak_worst, 2), "held_outside_worst": round(self.held_worst),
                "waiting_area_m2": self.holding_area_m2,
                "waiting_area_density_worst": None if q == float("inf") else round(q, 2), "safe": self.safe}


@dataclass
class Choice:
    chosen: Option
    acceptable: list[Option]
    reason: str


def choose(options: list[Option]) -> Choice:
    """The venue-wide rule (module docstring). `acceptable` holds every option the rule cannot
    tell apart from the chosen one; a proposal of any of them is accepted."""
    if not options:
        raise ValueError("no options to choose from")
    safe = [o for o in options if o.safe]
    if safe:
        fewest = min(o.held_worst for o in safe)
        near = [o for o in safe if o.held_worst <= fewest + NEAR_HELD]
        gentlest = min(o.disruption for o in near)
        acceptable = [o for o in near if o.disruption == gentlest]
        chosen = min(acceptable, key=lambda o: o.peak_ranked)
        held = f"holds about {chosen.held_worst:.0f} people outside" if chosen.held_worst >= 1 else "holds nobody outside"
        reason = (f"keeps the zone below {SAFE_RHO}/m2 even with the largest error measured in testing "
                  f"({chosen.peak_worst:.1f}) and {held}, the fewest of {len(safe)} safe actions")
    else:
        lowest = min(o.venue_worst for o in options)
        near = [o for o in options if o.venue_worst <= lowest + NEAR_RHO]
        gentlest = min(o.disruption for o in near)
        acceptable = [o for o in near if o.disruption == gentlest]
        chosen = min(acceptable, key=lambda o: (o.venue_worst, o.held_worst))
        reason = (f"no action keeps both the zone and the waiting areas below {SAFE_RHO}/m2; this one keeps the "
                  f"densest place lowest ({chosen.venue_worst:.1f}/m2 with the largest error measured in testing)")
    return Choice(chosen, acceptable, reason)
