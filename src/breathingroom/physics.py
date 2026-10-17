"""Crowd physics on zones: level of service, flux through lines, and the zone mass balance
that turns the current measurements into minutes-to-danger and into the predicted effect
of an action.

The forecast is the continuity equation integrated over a zone (a control volume):

    dN/dt = Q_in - Q_out

N is the number of people in the zone (measured), Q_out the flow through its exits
(measured), so Q_in = dN/dt + Q_out follows without any camera on the entrances, and so
does its trend. The zone is integrated forward until it reaches a density rho*. An action
changes one term: metering an entrance caps its share of Q_in, holding it removes that
share, opening an exit raises the outflow capacity. Every number in a proposal comes from
these lines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.stats import theilslopes

# Specific flow through a bottleneck, persons per metre of width per second. Seyfried et al.
# (2009) and Kretz et al. (2006) measure 1.6 to 2.0; 1.3 is the conservative planning value
# (Green Guide to Safety at Sports Grounds, 6th ed., 2018: 82 persons/m/min on level ways).
SPECIFIC_FLOW = 1.3

# Fruin (1971) level of service for queuing areas, in people per square metre.
FRUIN_QUEUE = [("A", 0.0), ("B", 0.83), ("C", 1.11), ("D", 1.54), ("E", 3.57), ("F", 5.26)]  # 1 / (m^2 per person)


def level_of_service(rho: float) -> str:
    los = "A"
    for name, lo in FRUIN_QUEUE:
        if rho >= lo:
            los = name
    return los


@dataclass
class Line:
    """A gate or exit as a segment on the floor; `normal` points out of the zone."""

    name: str
    a: tuple[float, float]
    b: tuple[float, float]
    kind: str = "exit"  # exit | entry

    @property
    def width(self) -> float:
        return float(np.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1]))

    def samples(self, n: int = 20) -> np.ndarray:
        t = (np.arange(n) + 0.5) / n
        a, b = np.array(self.a), np.array(self.b)
        return a[None] + t[:, None] * (b - a)[None]

    def normal(self, inside: tuple[float, float]) -> np.ndarray:
        a, b = np.array(self.a), np.array(self.b)
        d = (b - a) / max(self.width, 1e-9)
        n = np.array([d[1], -d[0]])
        mid = (a + b) / 2
        return -n if (np.array(inside) - mid) @ n > 0 else n


def flux(line: Line, rho_at: np.ndarray, v_at: np.ndarray, inside: tuple[float, float]) -> float:
    """People per second crossing `line` outwards: integral of rho (v . n) along it."""
    n = line.normal(inside)
    per = rho_at * (v_at @ n)
    return float(per.mean() * line.width)


@dataclass
class Action:
    """A steward action, in terms of the one term of the mass balance it changes."""

    kind: str  # meter_entry | hold_entry | open_exit | announce | send_steward | none
    # announce and send_steward act on behaviour (pushing), which the mass balance does not
    # model: their forecast is the no-action forecast.
    target: str = ""
    rate_pps: float = 0.0  # meter_entry: allowed inflow, persons/s
    width_m: float = 0.0  # open_exit: extra exit width
    share: float = 1.0  # meter_entry / hold_entry: fraction of the zone's inflow that comes through `target`

    def describe(self) -> str:
        if self.kind == "meter_entry":
            return f"meter {self.target} to {self.rate_pps * 60:.0f} people/min"
        if self.kind == "hold_entry":
            return f"hold {self.target}"
        if self.kind == "open_exit":
            return f"open {self.target} ({self.width_m:.1f} m)"
        if self.kind == "announce":
            return f"announce: {self.target}"
        if self.kind == "send_steward":
            return f"send a steward to {self.target}"
        return "no action"


@dataclass
class Forecast:
    zone: str
    action: str
    now_n: float
    area_m2: float
    rho_now: float
    net_rate_pps: float
    net_rate_ci: tuple[float, float]
    inflow_pps: float
    outflow_pps: float
    exit_capacity_pps: float
    threshold: float
    seconds_to_threshold: float | None
    seconds_to_threshold_range: tuple[float | None, float | None]
    peak_rho: float
    horizon_s: float
    curve_t: list[float] = field(default_factory=list)
    curve_rho: list[float] = field(default_factory=list)
    inflow_trend_pps2: float = 0.0
    # metering or holding an entrance moves people out of the zone into a queue before it: the
    # most people waiting there within the horizon, and how long the last of them has waited
    held_back_peak: float = 0.0
    held_back_wait_s: float = 0.0
    id: str = ""

    def to_json(self) -> dict:
        return asdict(self)


TREND_PERSIST_S = 120.0  # a rising or falling inflow is assumed to keep its trend this long, then hold
TREND_CAP = 1.5  # ...and to rise to at most 1.5 times its current value
TREND_LAG_S = 5.0  # inflow is estimated over 5 s differences


def _inflow_series(t: np.ndarray, n: np.ndarray, q_out: np.ndarray, lag_s: float = TREND_LAG_S):
    """Inflow at each sample: dN/dt over `lag_s` plus the mean outflow over the same span."""
    k = max(1, int(round(lag_s / max(np.median(np.diff(t)), 1e-6)))) if len(t) > 1 else 1
    if len(t) <= k:
        return t[:0], t[:0]
    dn = (n[k:] - n[:-k]) / np.maximum(t[k:] - t[:-k], 1e-9)
    qo = np.convolve(q_out, np.ones(k) / k, mode="valid")[1:] if k > 1 else q_out[1:]
    return t[k:], dn + qo[-len(dn):]


def forecast(zone: str, t: np.ndarray, n: np.ndarray, q_out: np.ndarray, area_m2: float,
             exit_width_m: float, threshold: float = 5.0, horizon_s: float = 600.0,
             action: Action | None = None, window_s: float = 30.0, rho_max: float = 8.0,
             specific_flow: float = SPECIFIC_FLOW, min_points: int = 10,
             trend_persist_s: float = TREND_PERSIST_S) -> Forecast:
    """Minutes-to-threshold for a zone, under the current trend or under an action.

    t, n, q_out: recent history (seconds, people in zone, outflow persons/s). The inflow
    follows from the mass balance (Q_in = dN/dt + Q_out); its trend over the window is
    carried forward for `trend_persist_s` (to at most TREND_CAP times its current value),
    then held. Outflow scales with the people in the zone up to the exits' capacity. The zone
    is integrated in 5 s steps. With fewer than `min_points` samples
    in the window the trend is taken as flat."""
    t = np.asarray(t, float)
    n = np.asarray(n, float)
    q_out = np.asarray(q_out, float)
    w = t >= t[-1] - window_s
    if w.sum() >= max(3, min_points):
        slope, _, lo, hi = theilslopes(n[w], t[w])
    else:
        slope = lo = hi = 0.0
    out_now = float(np.mean(q_out[w])) if w.any() else 0.0
    inflow = max(0.0, slope + out_now)
    growth = 0.0
    ti, qi = _inflow_series(t[w], n[w], q_out[w])
    if len(ti) >= max(3, min_points - int(TREND_LAG_S)):
        growth = float(theilslopes(qi, ti)[0])
    cap = specific_flow * exit_width_m
    action = action or Action("none")
    extra = specific_flow * action.width_m if action.kind == "open_exit" else 0.0
    cap_total = cap + extra
    n_now = float(n[-1])
    nmax = rho_max * area_m2

    def entering(q):
        if action.kind == "meter_entry":  # only the metered entrance's share is capped
            return q * (1 - action.share) + min(q * action.share, action.rate_pps)
        if action.kind == "hold_entry":
            return q * (1 - action.share)
        return q

    def leaving(q_in_eff, n_cur):
        # outflow scales with the people in the zone (and with the open exit width) until the
        # exits reach their capacity; an empty zone lets out what comes in
        if n_now < 1.0:
            return min(cap_total, q_in_eff)
        base = out_now * n_cur / n_now
        if action.kind == "open_exit" and cap > 0:
            base *= cap_total / cap
        return min(cap_total, base) if cap_total > 0 else base

    def integrate(q0: float, until: float, dt: float = 5.0):
        ts = np.arange(0.0, until + dt / 2, dt)
        ns = np.empty_like(ts)
        cur = n_now
        for i, tau in enumerate(ts):
            ns[i] = cur
            q = q0 + growth * min(tau, trend_persist_s)
            q = float(np.clip(q, 0.0, TREND_CAP * max(q0, 0.5)))
            qe = entering(q)
            cur = float(np.clip(cur + (qe - leaving(qe, cur)) * dt, 0.0, nmax))
        return ts, ns

    long_s = max(horizon_s, 1800.0)
    target = threshold * area_m2

    def crossing(ts, ns):
        k = np.flatnonzero(ns >= target)
        if not len(k):
            return None
        if k[0] == 0:
            return 0.0
        a, b = k[0] - 1, k[0]
        return float(ts[a] + (target - ns[a]) / max(ns[b] - ns[a], 1e-9) * (ts[b] - ts[a]))

    ts, ns = integrate(inflow, long_s)
    tt = crossing(ts, ns)
    rng = (crossing(*integrate(inflow + (hi - slope), long_s)), crossing(*integrate(inflow + (lo - slope), long_s)))
    grid = np.linspace(0, horizon_s, 61)
    curve = np.interp(grid, ts, ns) / area_m2
    infl0 = entering(inflow)
    outf0 = leaving(infl0, n_now)
    net = infl0 - outf0
    held, held_peak, wait = 0.0, 0.0, 0.0
    for tau in np.arange(0.0, horizon_s, 5.0):  # arrivals the action turns away from the zone queue up
        q = float(np.clip(inflow + growth * min(tau, trend_persist_s), 0.0, TREND_CAP * max(inflow, 0.5)))
        held += (q - entering(q)) * 5.0
        held_peak = max(held_peak, held)
    if held_peak > 0:
        let_through = action.rate_pps if action.kind == "meter_entry" else 0.0
        wait = horizon_s if let_through <= 0 else min(horizon_s, held / let_through)
    fc = Forecast(zone, action.describe(), n_now, area_m2, n_now / area_m2, float(net),
                  (float(net + lo - slope), float(net + hi - slope)), float(infl0), float(outf0), float(cap_total),
                  threshold, tt, rng, float(curve.max()), horizon_s, grid.tolist(), curve.tolist(),
                  inflow_trend_pps2=growth, held_back_peak=float(held_peak), held_back_wait_s=float(wait))
    fc.id = "fc_" + hashlib.sha256(json.dumps(fc.to_json(), sort_keys=True, default=str).encode()).hexdigest()[:12]
    return fc
