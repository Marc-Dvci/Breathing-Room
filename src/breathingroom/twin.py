"""Venue twin: a JuPedSim crowd in a street that feeds a narrow exit.

The twin is where steward actions can be rehearsed and where the forecaster is scored
against a ground truth that runs for minutes. Arrivals come through two gates at a rate
that rises like an event egress; the street ends in a 2.4 m exit, and a second exit door
(1.5 m) beside it can be opened.

Geometry (metres): street x 0..30, y 0..6. Gate A spans y 0..3 at x = 0, gate B y 3..6.
Main exit x 30..33, y 1.8..4.2. Second door x 30..33, y 4.5..6.0, behind a 0.3 m wall
(closed until opened).
Zone C, the stretch in front of the exit, is x 15..30.

Other venues are a `Layout` with another street, exit, door and demand; the module constants
describe the default one, which the console runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jupedsim as jps
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

def egress_profile(t: float) -> float:
    """Total arrivals (persons/s) at time t: a surge like the end of a concert."""
    return float(np.interp(t, [0, 60, 240, 480, 600], [1.5, 2.5, 6.5, 6.5, 2.0]))


@dataclass(frozen=True)
class Layout:
    """A street of length x width metres with two entry gates at x = 0, a main exit centred on
    the far end and a second door in its upper corner. Zone C is the far half of the street.
    `demand` scales the arrival surge."""
    length: float = 30.0
    width: float = 6.0
    exit_width: float = 2.4
    door_width: float = 1.5
    demand: float = 1.0

    def __post_init__(self):
        if self.width - self.door_width < (self.width + self.exit_width) / 2 + 0.3:
            raise ValueError("the second door must be at least 0.3 m from the main exit")

    @property
    def street(self) -> Polygon:
        return box(0, 0, self.length, self.width)

    @property
    def exit_y(self) -> tuple[float, float]:
        y0 = (self.width - self.exit_width) / 2
        return y0, y0 + self.exit_width

    @property
    def door_y0(self) -> float:
        return self.width - self.door_width

    @property
    def main_exit(self) -> Polygon:
        return box(self.length, self.exit_y[0], self.length + 3, self.exit_y[1])

    @property
    def side(self) -> Polygon:
        return box(self.length, self.door_y0, self.length + 3, self.width)

    @property
    def zone_c(self) -> tuple[float, float, float, float]:
        return (self.length / 2, self.length, 0.0, self.width)

    @property
    def zone_c_area(self) -> float:
        return self.length / 2 * self.width

    @property
    def gates(self) -> dict[str, tuple[float, float, float, float]]:
        h = self.width / 2
        return {"gate A": (0.3, 1.6, 0.3, h - 0.3), "gate B": (0.3, 1.6, h + 0.3, self.width - 0.3)}

    def profile(self, t: float) -> float:
        return self.demand * egress_profile(t)


DEFAULT = Layout()
STREET = DEFAULT.street
MAIN_EXIT = DEFAULT.main_exit
SIDE = DEFAULT.side  # the second exit door
ZONE_C = DEFAULT.zone_c
ZONE_C_AREA = DEFAULT.zone_c_area
GATES = DEFAULT.gates
MAIN_EXIT_WIDTH = DEFAULT.exit_width
SIDE_WIDTH = DEFAULT.door_width


@dataclass
class Observation:
    t: float
    positions: np.ndarray  # (n, 2)
    velocities: np.ndarray  # (n, 2)
    n_zone: int
    outflow_pps: float
    backlog: dict[str, int]
    arrivals_pps: float
    side_open: bool


@dataclass
class Venue:
    seed: int = 0
    dt: float = 0.05
    radius: float = 0.18
    layout: Layout = DEFAULT
    profile: callable = None  # arrivals (persons/s) at time t; default: the layout's surge
    gate_limit_pps: dict[str, float | None] = field(default_factory=lambda: {"gate A": None, "gate B": None})
    side_open: bool = False
    model: str = "csm"  # csm: collision-free speed model; avm: anticipation velocity model (slower)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        L = self.layout
        if self.profile is None:
            self.profile = L.profile
        self.gates = L.gates
        geom = unary_union([L.street, L.main_exit, L.side])
        if self.model == "avm":
            # anticipates neighbours' motion, so streams that cross at the exits keep moving;
            # the collision-free speed model gridlocks there
            pedestrian_model, self._params = jps.AnticipationVelocityModel(rng_seed=self.seed), jps.AnticipationVelocityModelAgentParameters
        else:
            pedestrian_model, self._params = jps.CollisionFreeSpeedModel(), jps.CollisionFreeSpeedModelAgentParameters
        self.sim = jps.Simulation(model=pedestrian_model, geometry=geom, dt=self.dt)
        self.exit_main = self.sim.add_exit_stage(box(L.length + 2, L.exit_y[0], L.length + 3, L.exit_y[1]))
        self.exit_side = self.sim.add_exit_stage(box(L.length + 2, L.door_y0, L.length + 3, L.width))
        self.j_main = self.sim.add_journey(jps.JourneyDescription([self.exit_main]))
        self.j_side = self.sim.add_journey(jps.JourneyDescription([self.exit_side]))
        self.backlog = {g: 0.0 for g in self.gates}  # people arrived but not yet through the gate
        self.allowed = {g: 0.0 for g in self.gates}  # fractional passage credit under metering
        self._last_pos: dict[int, np.ndarray] = {}
        self._removed_total = 0
        self._out_hist: list[tuple[float, int]] = []

    @property
    def t(self) -> float:
        return float(self.sim.elapsed_time())

    # -- actions ---------------------------------------------------------------------
    def meter(self, gate: str, rate_pps: float | None) -> None:
        self.gate_limit_pps[gate] = rate_pps

    def open_side(self) -> None:
        """Open the second exit door: people in Zone C on the door's side of the street
        re-route to it; new arrivals through gate B mostly head for it."""
        if self.side_open:
            return
        self.side_open = True
        L = self.layout
        for a in self.sim.agents():
            x, y = a.position
            if x >= L.length / 2 and y >= L.door_y0 - 0.9 and a.journey_id == self.j_main:
                self.sim.switch_agent_journey(a.id, self.j_side, self.exit_side)

    # -- stepping --------------------------------------------------------------------
    def _spawn(self, gate: str, n_wanted: int) -> int:
        x0, x1, y0, y1 = self.gates[gate]
        done = 0
        for _ in range(min(n_wanted * 4, 12)):  # a jammed gate area stops trying quickly
            if done >= n_wanted:
                break
            p = (float(self.rng.uniform(x0, x1)), float(self.rng.uniform(y0, y1)))
            if list(self.sim.agents_in_range(p, 2.2 * self.radius)):
                continue
            journey, stage = (self.j_main, self.exit_main)
            # gate B arrivals take the door 75% of the time: the door then carries about 38% of
            # the crowd, its share of the total exit width (1.5 of 3.9 m)
            if self.side_open and gate == "gate B" and self.rng.random() < 0.75:
                journey, stage = (self.j_side, self.exit_side)
            self.sim.add_agent(self._params(
                position=p, journey_id=journey, stage_id=stage, radius=self.radius,
                desired_speed=float(self.rng.normal(1.3, 0.15)), time_gap=1.0))
            done += 1
        return done

    def step(self, seconds: float = 1.0) -> None:
        steps = int(round(seconds / self.dt))
        for _ in range(steps):
            rate = self.profile(self.t) / len(self.gates)
            for g in self.gates:
                self.backlog[g] += rate * self.dt
                lim = self.gate_limit_pps.get(g)
                self.allowed[g] += (lim if lim is not None else 1e9) * self.dt
                n = int(min(self.backlog[g], self.allowed[g]))
                if n > 0:
                    k = self._spawn(g, n)
                    self.backlog[g] -= k
                    self.allowed[g] -= k
                if lim is None:
                    self.allowed[g] = 0.0
                else:
                    self.allowed[g] = min(self.allowed[g], 1.0)
            before = self.sim.agent_count()
            self.sim.iterate()
            self._removed_total += max(0, before - self.sim.agent_count())
            if self.side_open and self.sim.iteration_count() % 10 == 0:
                self._take_the_door_in_front()
        self._out_hist.append((self.t, self._removed_total))

    def _take_the_door_in_front(self) -> None:
        """Close to the exits, people use the open door they are lined up with. Without this,
        a person bound for the second door who is pushed into the main exit keeps pressing
        against the wall between the two and blocks the main exit."""
        L = self.layout
        for a in self.sim.agents():
            x, y = a.position
            if x < L.length - 3:
                continue
            want_side = y >= L.door_y0 - 0.15 if x >= L.length - 0.5 else y >= L.door_y0 - 0.5
            if want_side and a.journey_id == self.j_main:
                self.sim.switch_agent_journey(a.id, self.j_side, self.exit_side)
            elif not want_side and a.journey_id == self.j_side:
                self.sim.switch_agent_journey(a.id, self.j_main, self.exit_main)

    def observe(self) -> Observation:
        ids, pos = [], []
        for a in self.sim.agents():
            ids.append(a.id)
            pos.append(a.position)
        pos = np.array(pos, float).reshape(-1, 2)
        vel = np.zeros_like(pos)
        for i, (aid, p) in enumerate(zip(ids, pos)):
            if aid in self._last_pos:
                vel[i] = (p - self._last_pos[aid]) / 1.0
        self._last_pos = {aid: p for aid, p in zip(ids, pos)}
        x0, x1, y0, y1 = self.layout.zone_c
        inz = (pos[:, 0] >= x0) & (pos[:, 0] < x1) & (pos[:, 1] >= y0) & (pos[:, 1] < y1) if len(pos) else np.zeros(0, bool)
        out = 0.0
        if len(self._out_hist) >= 2:
            (t0, r0), (t1, r1) = self._out_hist[-2], self._out_hist[-1]
            out = (r1 - r0) / max(t1 - t0, 1e-9)
        return Observation(self.t, pos, vel, int(inz.sum()), float(out),
                           {g: int(b) for g, b in self.backlog.items()}, self.profile(self.t), self.side_open)


def polygon_outline() -> dict:
    """Floor plan outline for the console (street, exits, second door, zone, gates)."""
    def coords(p: Polygon):
        return [list(map(float, c)) for c in p.exterior.coords]
    return {
        "street": coords(STREET), "main_exit": coords(MAIN_EXIT), "side": coords(SIDE),
        "zone_c": coords(box(ZONE_C[0], ZONE_C[2], ZONE_C[1], ZONE_C[3])),
        "gates": {g: [x0, y0, x1, y1] for g, (x0, x1, y0, y1) in GATES.items()},
    }
