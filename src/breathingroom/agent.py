"""The steward's assistant: an agent that investigates a zone with vision tools, compares
actions with the physics, and puts one proposal in front of a person.

Division of labour:
- the tier comes from policy.TierPolicy, from measured numbers;
- every number in a proposal comes from a tool result (OpenCV measurements, physics.forecast);
- the model chooses which tools to call, in what order, which action to propose, and writes
  the radio message;
- a person approves or rejects. Nothing reaches a gate or a PA system without that.

The safeguards are code, not instructions. `Episode._propose` refuses a proposal, and the
refusal goes back to the model as a tool result, unless:
- the zone's state was read, its camera checked and trusted at the latest check, and its count
  cross-checked in this episode; where the zone is flagged for pushing, the pushing located;
- it cites a forecast of this episode, for this episode's zone, of an action in the venue's
  catalogue, at the catalogue's metering rate;
- the action is one the venue-wide rule (choice.py) accepts, judged against every action in
  the catalogue: code forecasts any the model did not;
- the radio message has at most 30 words, names the zone, says pushing is detected when the
  zone is flagged for it, and every number in it matches a tool result of the same kind
  (a density with a density, minutes to 4.5 with the zone's own minutes_to_4_5, and so on);
  it says the action proposed and asks for nothing else: no other action's words, gates or
  doors, and no instruction outside the catalogue.

Model: openai.gpt-oss-120b through the Bedrock API endpoint (bedrock-mantle, OpenAI-compatible).
When the model is unavailable, a deterministic planner calls the same tools in a fixed order.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import choice
from .choice import DISRUPTION  # noqa: F401  (re-exported: how disruptive each kind of action is)
from .physics import Action, Forecast
from .policy import Thresholds, Tier
from .trace import Trace

MODEL_ID = os.environ.get("BR_MODEL_ID", "openai.gpt-oss-120b")
MANTLE_URL = os.environ.get("BR_MANTLE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1")
MAX_STEPS = 12
RADIO_MAX_WORDS = 30
FORECAST_HORIZON_MIN = 10
# What a number in the radio message is, from the words after it, and the tool-result fields a
# number of that kind may come from. A density can only be a density some tool returned, and so on.
_UNIT_RE = [
    ("density", re.compile(r"\s*(?:people|persons)?\s*(?:/|per)\s*(?:m2|m²|m\^2|sq\.?\s*m|square\s+met(?:re|er)s?)", re.I)),
    ("rate", re.compile(r"\s*(?:people|persons)?\s*(?:/|per|a)\s*min(?:ute)?s?\b", re.I)),
    ("minutes", re.compile(r"\s*min(?:ute)?s?\b", re.I)),
    ("metres", re.compile(r"\s*(?:m|metres?|meters?)\b", re.I)),
    ("people", re.compile(r"\s*(?:people|persons)\b", re.I)),
]
_FIELDS = {
    "density": {"density", "density_now", "peak_density_10min", "peak_with_error", "peak", "threshold",
                "zone_peak_worst", "waiting_area_density_worst"},
    "rate": {"rate_per_min", "inflow_per_min", "outflow_per_min", "net_rate_per_min", "exit_capacity_per_min",
             "trend_per_min"},
    "minutes": {"minutes_to_4_5", "minutes_to_threshold", "longest_wait_at_entrance_min"},
    "metres": {"metres_before_gate", "floor_shift_m", "exit_width_m", "width_m"},
    "people": {"people", "count_primary", "count_second", "people_held_at_entrance_10min", "held_outside_worst"},
}
_TIME_LEFT_RE = re.compile(r"\s*min(?:ute)?s?\s+(?:to|left|until|before|from)\b", re.I)
# Words that ask someone at the zone to do something, by the kind of action they belong to. The
# steward reads the radio message out, so it must ask for the action cited and for nothing else:
# no other action's words, no instruction outside the catalogue. Words of the action's own target
# are allowed ("please stop pushing, the gate is open").
_ACTION_WORDS = {
    "open_exit": {"open", "opens", "opening"},
    "meter_entry": {"meter", "meters", "metering", "slow", "slowing"},  # not "limit": "4.5 limit" is a density
    "hold_entry": {"hold", "holding", "close", "closing", "shut", "stop", "stopping"},
    "send_steward": {"send", "sending", "steward", "stewards"},
    "announce": {"announce", "announcing", "announcement"},
}
_OTHER_INSTRUCTIONS = {"evacuate", "evacuating", "evacuation", "run", "clear", "clearing", "wave", "divert", "diverting",
                       "redirect", "move", "crush", "stampede", "panic"}
_INSTRUCTIONS = set().union(*_ACTION_WORDS.values(), _OTHER_INSTRUCTIONS)


@dataclass
class Proposal:
    id: str
    zone: str
    tier: str
    action: dict
    forecast_id: str
    predicted: dict
    alternatives: list[dict]
    radio: str
    rationale: str
    evidence: list[str]
    created_t: float
    planner: str  # "model" or "deterministic"
    status: str = "pending"  # pending | approved | rejected | expired
    decided_by: str = ""
    verify: dict = field(default_factory=dict)
    replay: bool = False  # the zone is a recording: approving records the decision only
    predicted_at_proposal: dict = field(default_factory=dict)  # set when approval re-forecasts
    evidence_images: list[dict] = field(default_factory=list)  # camera crops from locate_pushing
    choice: dict = field(default_factory=dict)  # what the venue-wide rule chose, and why

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class VenueCatalogue:
    """What stewards can do, per zone. Actions outside this list cannot be proposed."""

    actions: dict[str, list[dict]]

    def allowed(self, zone: str, kind: str, target: str) -> bool:
        return any(a["kind"] == kind and a.get("target", "") == target for a in self.actions.get(zone, []))


SYSTEM_PROMPT = """You assist crowd-safety stewards at a venue. A zone has been flagged by the
alert policy. Investigate it with the tools, then propose ONE action for a person to approve,
or say that no action is needed.

Rules (code checks each one and refuses a proposal that breaks it):
1. Every number you state must come from a tool result in this conversation, as the same kind
   of quantity: a density from a density, a time from a time.
2. Before proposing, confirm the measurement: call get_zone_state, check_camera for the zone's
   camera and cross_check_count for the zone. If the camera is not trusted, call
   request_recalibration and do not propose a crowd action from that camera.
3. Compare the actions with forecast_action (code forecasts any you leave out), then propose
   the one the venue-wide rule accepts. An action protects the zone only if it does not move
   the danger elsewhere: metering or holding an entrance keeps people waiting before it, and
   people_held_at_entrance_10min and waiting_area_density_worst say how many and how dense.
   An action is safe when zone_peak_worst and waiting_area_density_worst both stay below 3.5.
   Among safe actions, propose the one that holds the fewest people outside, then the least
   disruptive; if none is safe, the one whose densest place is lowest. Holding an entrance is
   the last resort.
   The forecast models crowd numbers only. When the zone is flagged for pushing and the
   density is not rising, a steward at the spot or an announcement is the proportionate
   action; their forecast equals the no-action forecast, and you say so.
4. propose_action must cite the forecast_id of the forecast that supports it.
5. The radio message is at most 30 words, plain English, and names the zone, the density,
   the minutes left, and the action, and asks for nothing but that action. No jargon. The minutes left are minutes_to_4_5 from
   get_zone_state, the zone as it is now; never take them from an action's forecast. When
   minutes_to_4_5 is null the zone is not filling: say so, and never write "0 minutes". When
   the zone is flagged for pushing, say that pushing is detected.
6. When the zone is flagged for pushing, call locate_pushing before choosing an action. If it
   finds a hotspot, the action goes to that spot: a steward or an announcement, and the radio
   message says where (the side and the metres before the gate). If it finds no hotspot, the
   pushing is not concentrated anywhere: say so, and decide from the density and the trend.
7. You never approve anything yourself. A steward decides."""


def tool_specs(catalogue_actions: list[str]) -> list[dict]:
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                                                 "parameters": {"type": "object", "properties": props, "required": req}}}
    return [
        fn("get_zone_state", "Current measured state of a zone: density, pushing index, trend, inflow, outflow.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("check_camera", "Check that the camera has not moved since calibration: a fast ECC alignment against the "
           "calibration frame, confirmed with ALIKED + LightGlue feature matching when the fast check is not steady.",
           {"camera": {"type": "string"}}, ["camera"]),
        fn("cross_check_count", "Count the zone again with a second, independent counting model and report agreement.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("locate_pushing", "Find where in the zone people are pushing: regions above the pushing alert threshold, "
           "with their size, peak index, density, side and distance before the gate, and a camera crop for the steward.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("measure_exit_flow", "Measured flow through the zone's exits over the last 30 s, with capacity.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("forecast_action", "Predict the zone's density over the next 10 minutes under an action, with the mass balance. "
           f"kind is one of {catalogue_actions}. A metering rate and a door's width come from the venue.",
           {"zone": {"type": "string"}, "kind": {"type": "string"}, "target": {"type": "string"}}, ["zone", "kind"]),
        fn("propose_action", "Put one action in front of the steward for approval.",
           {"zone": {"type": "string"}, "forecast_id": {"type": "string"}, "radio_message": {"type": "string"},
            "rationale": {"type": "string"}}, ["zone", "forecast_id", "radio_message", "rationale"]),
        fn("request_recalibration", "Ask a person to recalibrate a camera that can no longer be trusted.",
           {"camera": {"type": "string"}, "reason": {"type": "string"}}, ["camera", "reason"]),
        fn("no_action", "Close the episode without a proposal, with the reason.",
           {"reason": {"type": "string"}}, ["reason"]),
    ]


class Episode:
    """One investigation of one zone. Tools are bound to the live runtime."""

    def __init__(self, zone: str, tier: Tier, reasons: list[str], tools: dict[str, Callable[..., dict]],
                 catalogue: VenueCatalogue, trace: Trace, clock: Callable[[], float]):
        self.id = "ep_" + uuid.uuid4().hex[:10]
        self.zone = zone
        self.tier = tier
        self.reasons = reasons
        self.tools = tools
        self.catalogue = catalogue
        self.trace = trace
        self.clock = clock
        self.forecasts: dict[str, Forecast] = {}
        self.forecast_actions: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.proposal: Proposal | None = None
        self.closed_reason = ""
        self.recalibration: dict | None = None
        self.planner = "model"
        self._by_code = False

    # -- tool dispatch -----------------------------------------------------------------
    @property
    def camera(self) -> str | None:
        return next((a.get("camera") for a in self.catalogue.actions.get(self.zone, []) if a.get("camera")), None)

    def _scope_error(self, name: str, args: dict) -> str:
        """Every tool works on this episode's zone and its camera; a call about another one is refused."""
        if "zone" in args and args["zone"] != self.zone:
            return f"this investigation is about zone {self.zone!r}; call {name} with that zone"
        if name in ("check_camera", "request_recalibration") and self.camera and args.get("camera") != self.camera:
            return f"the camera of {self.zone!r} is {self.camera!r}"
        return ""

    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        args = dict(args)
        if name in ("get_zone_state", "cross_check_count", "locate_pushing", "measure_exit_flow", "forecast_action",
                    "propose_action"):
            args.setdefault("zone", self.zone)
        try:
            scope = self._scope_error(name, args)
            if scope:
                out = {"error": scope}
            elif name == "forecast_action":
                out = self._forecast(args)
            elif name == "propose_action":
                out = self._propose(args)
            elif name == "request_recalibration":
                self.recalibration = {"camera": args.get("camera", ""), "reason": args.get("reason", "")}
                out = {"ok": True, "message": "recalibration requested from the control room"}
            elif name == "no_action":
                refused = self._no_action_refusal()
                if refused:
                    out = {"error": refused}
                else:
                    self.closed_reason = args.get("reason", "")
                    out = {"ok": True}
            elif name in self.tools:
                out = self.tools[name](**args)
            else:
                out = {"error": f"unknown tool {name}"}
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        except Exception as e:  # a failing tool is reported to the model, not raised
            out = {"error": f"{type(e).__name__}: {e}"}
        ms = (time.perf_counter() - t0) * 1000
        rec = {"episode": self.id, "tool": name, "args": args, "result": out, "ms": round(ms, 1)}
        if self._by_code:
            rec["by"] = "code"  # a call the guards made, not the planner
        self.calls.append(rec)
        self.trace.write("tool_call", **rec)
        return out

    def _no_action_refusal(self) -> str:
        """A zone that its own state says will reach 4.5/m2 within 10 minutes cannot be closed
        without action, unless a no-action forecast in this episode says it stays below."""
        states = [c["result"] for c in self.calls if c["tool"] == "get_zone_state" and isinstance(c["result"], dict)]
        mins = states[-1].get("minutes_to_4_5") if states else None
        if mins is None or mins > 10:
            return ""
        if any(v["kind"] == "none" and self.forecasts[k].peak_rho < 4.5 for k, v in self.forecast_actions.items()):
            return ""
        return (f"get_zone_state reports {mins} minutes to 4.5/m2 at the current inflow; an action's forecast is not the "
                "zone's future. Propose the action that keeps the peak lowest, or forecast kind none to show the zone stays below 4.5")

    def _forecast(self, a: dict) -> dict:
        zone = self.zone
        kind = a.get("kind", "none")
        target = a.get("target", "") or ""
        same_kind = [x for x in self.catalogue.actions.get(zone, []) if x["kind"] == kind]
        if kind != "none" and not target and len(same_kind) == 1:
            target = same_kind[0].get("target", "")  # one action of this kind: the target is not ambiguous
        if kind != "none" and not self.catalogue.allowed(zone, kind, target):
            opts = [f"{x['kind']}:{x.get('target', '')}" for x in self.catalogue.actions.get(zone, [])]
            return {"error": f"action {kind}:{target} is not available in {zone}; available: {opts}"}
        spec = next((x for x in self.catalogue.actions.get(zone, []) if x["kind"] == kind and x.get("target", "") == target), {})
        listed = spec.get("rate_per_min")  # a metering rate is the venue's, like a door's width
        if kind == "meter_entry" and listed is not None and a.get("rate_per_min") not in (None, "") \
                and round(float(a["rate_per_min"])) != round(float(listed)):
            return {"error": f"{kind}:{target} meters at the venue's {listed} people/min, the rate its forecast error was "
                             "measured at; forecast it without rate_per_min"}
        act = Action(kind, target, rate_pps=float(listed if listed is not None else a.get("rate_per_min", 60)) / 60.0,
                     width_m=float(spec.get("width_m", 0)), share=float(spec.get("share", 1.0)))
        fc = self.tools["_forecast"](zone=zone, action=act)
        self.forecasts[fc.id] = fc
        self.forecast_actions[fc.id] = {"kind": kind, "target": target, "rate_per_min": round(act.rate_pps * 60),
                                       "width_m": act.width_m}
        mins = None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1)
        err = spec.get("forecast_error")
        rec = self.forecast_actions[fc.id]
        rec.update({"forecast_error": err, "forecast_error_max": spec.get("forecast_error_max", err),
                    "queue_factor": spec.get("queue_factor", 1.0), "holding_area_m2": spec.get("holding_area_m2"),
                    "held_at_entrance": round(fc.held_back_peak), "longest_wait_min": round(fc.held_back_wait_s / 60, 1)})
        priced = self._option(fc.id).summary()
        rec.update(priced)
        return {"forecast_id": fc.id, "action": fc.action, "density_now": round(fc.rho_now, 2),
                "peak_density_10min": round(fc.peak_rho, 2), "minutes_to_threshold": mins,
                "measured_forecast_error": err,
                "peak_with_error": None if err is None else round(fc.peak_rho + err, 2),
                "threshold": fc.threshold, "net_rate_per_min": round(fc.net_rate_pps * 60, 1),
                "inflow_per_min": round(fc.inflow_pps * 60, 1), "outflow_per_min": round(fc.outflow_pps * 60, 1),
                "people_held_at_entrance_10min": round(fc.held_back_peak),
                "longest_wait_at_entrance_min": round(fc.held_back_wait_s / 60, 1), **priced}

    def _option(self, fid: str) -> choice.Option:
        a, f = self.forecast_actions[fid], self.forecasts[fid]
        err = a.get("forecast_error") or 0.0
        return choice.Option(fid, a["kind"], f.peak_rho, err, a.get("forecast_error_max") or err, f.held_back_peak,
                             a.get("queue_factor") or 1.0, a.get("holding_area_m2"))

    def _latest(self, tool: str) -> dict | None:
        res = [c["result"] for c in self.calls if c["tool"] == tool and isinstance(c["result"], dict)
               and "error" not in c["result"]]
        return res[-1] if res else None

    @property
    def flagged_for_pushing(self) -> bool:
        return any("pushing" in r for r in self.reasons)

    def _missing_checks(self) -> list[str]:
        """Measurements a proposal rests on that this episode has not made, or that failed."""
        missing = []
        state = self._latest("get_zone_state")
        if state is None:
            missing.append("get_zone_state")
        elif state.get("feed", "live") != "live":
            missing.append(f"a live feed: the zone's source is {state['feed']}")
        if self.camera:
            cams = [c["result"] for c in self.calls if c["tool"] == "check_camera"]
            if not cams or not isinstance(cams[-1], dict) or "error" in cams[-1]:
                missing.append(f"check_camera ({self.camera})")
            elif not cams[-1].get("trusted", False):
                missing.append(f"a trusted camera: {self.camera} failed its last check, call request_recalibration")
        if self._latest("cross_check_count") is None:
            missing.append("cross_check_count")
        if self.flagged_for_pushing and self._latest("locate_pushing") is None:
            missing.append("locate_pushing (the zone is flagged for pushing)")
        return missing

    def _complete_forecasts(self) -> None:
        """Forecast every catalogue action the model did not, so the rule compares all of them."""
        done = {(v["kind"], v["target"]) for v in self.forecast_actions.values()}
        self._by_code = True
        try:
            for x in self.catalogue.actions.get(self.zone, []):
                if x["kind"] != "none" and (x["kind"], x.get("target", "")) not in done:
                    self.call("forecast_action", {"zone": self.zone, "kind": x["kind"], "target": x.get("target", ""),
                                                  "rate_per_min": x.get("rate_per_min", 60)})
            if not any(v["kind"] == "none" for v in self.forecast_actions.values()):
                self.call("forecast_action", {"zone": self.zone, "kind": "none"})  # the baseline the steward compares with
        finally:
            self._by_code = False

    def _propose(self, a: dict) -> dict:
        fid = a.get("forecast_id", "")
        if fid not in self.forecasts:
            return {"error": "forecast_id not found: call forecast_action first and cite its forecast_id"}
        act = self.forecast_actions[fid]
        if act["kind"] == "none":
            return {"error": "the cited forecast is for no action; cite the forecast of the action you propose"}
        missing = self._missing_checks()
        if missing:
            return {"error": "before proposing, this episode needs: " + "; ".join(missing)}
        radio = (a.get("radio_message") or "").strip()
        problems = self.radio_problems(radio, act)
        if problems:
            return {"error": "radio_message: " + "; ".join(problems)}
        self._complete_forecasts()
        options = [self._option(k) for k, v in self.forecast_actions.items() if v["kind"] != "none"]
        ch = choice.choose(options)
        if fid not in {o.key for o in ch.acceptable}:
            c, mine = ch.chosen, self._option(fid)
            ca = self.forecast_actions[c.key]
            return {"error": f"the venue-wide rule proposes {ca['kind']}:{ca['target']} (forecast_id {c.key}): it "
                             f"{ch.reason}. This action: zone peak with the largest tested error {mine.peak_worst:.2f}/m2, "
                             f"about {mine.held_worst:.0f} people held outside"
                             + ("" if mine.queue_rho_worst == 0 else
                                f" ({'no declared waiting area' if mine.queue_rho_worst == float('inf') else f'{mine.queue_rho_worst:.1f}/m2 in the waiting area'})")
                             + f". Propose forecast_id {c.key}"}
        fc = self.forecasts[fid]
        alts = [{"forecast_id": k, **self.forecast_actions[k], "peak": round(f.peak_rho, 2),
                 "peak_with_error": None if self.forecast_actions[k].get("forecast_error") is None
                 else round(f.peak_rho + self.forecast_actions[k]["forecast_error"], 2),
                 "minutes_to_threshold": None if f.seconds_to_threshold is None else round(f.seconds_to_threshold / 60, 1)}
                for k, f in self.forecasts.items()]
        evidence = [c["tool"] for c in self.calls]
        images = [{"id": h["evidence_image"], "side": h.get("side"), "metres_before_gate": h.get("metres_before_gate"),
                   "peak_pushing_index": h.get("peak_pushing_index")}
                  for c in self.calls if c["tool"] == "locate_pushing" and isinstance(c["result"], dict)
                  for h in (c["result"].get("hotspots") or [])[:2] if h.get("evidence_image")]
        rule = self._option(fid)
        self.proposal = Proposal(
            id="pr_" + uuid.uuid4().hex[:10], zone=self.zone, tier=self.tier.name, action=act,
            forecast_id=fid, predicted={"peak_density": round(fc.peak_rho, 2), "curve_t": fc.curve_t,
                                        "curve_rho": [round(x, 3) for x in fc.curve_rho],
                                        "minutes_to_threshold": None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1)},
            alternatives=alts, radio=radio, rationale=(a.get("rationale") or "").strip(), evidence=evidence,
            created_t=self.clock(), planner=self.planner, evidence_images=images[-2:],
            choice={"rule": "venue-wide", "reason": ch.reason if fid == ch.chosen.key else
                    f"as safe as the rule's choice: {ch.reason}", **rule.summary(),
                    "safe_density": choice.SAFE_RHO, "compared": len(options)})
        return {"ok": True, "proposal_id": self.proposal.id, "message": "sent to the steward for approval"}

    def _typed_values(self) -> tuple[dict[str, list[float]], list[float]]:
        """Numbers returned by tools in this episode: by kind (from the field they came in),
        and all of them, including numbers inside text results, the catalogue and the zone name."""
        typed: dict[str, list[float]] = {k: [] for k in _FIELDS}
        known: list[float] = []

        def walk(x, key=""):
            if isinstance(x, bool) or x is None:
                return
            if isinstance(x, (int, float)):
                known.append(float(x))
                for kind, keys in _FIELDS.items():
                    if key in keys:
                        typed[kind].append(float(x))
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, k)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v, key)
            elif isinstance(x, str):
                known.extend(float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", x))

        for c in self.calls:
            walk(c["result"])
        walk(self.catalogue.actions.get(self.zone, []))
        walk(self.zone)
        th = Thresholds()
        typed["density"] += [th.watch_rho, th.warn_rho, th.act_rho, th.critical_rho]
        typed["minutes"].append(float(FORECAST_HORIZON_MIN))
        return typed, known

    def ungrounded_numbers(self, text: str) -> list[str]:
        """Numbers in `text` that no tool returned in this episode as that kind of quantity, at the
        precision written. The kind is read from the words after the number (per m2: a density;
        minutes to, left or until: the zone's own minutes_to_4_5; per minute: a rate; m: metres;
        people: a count). A number with no unit only has to match some tool result."""
        typed, known = self._typed_values()
        state = self._latest("get_zone_state") or {}
        time_left = state.get("minutes_to_4_5")
        out = []
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)", text):
            tok, rest = m.group(1), text[m.end():]
            dec = len(tok.split(".")[1]) if "." in tok else 0
            v, tol = float(tok), 0.5 * 10 ** -dec + 1e-9
            if _TIME_LEFT_RE.match(rest):
                pool, kind = ([] if time_left is None else [float(time_left)]), "minutes to 4.5"
            else:
                kind = next((k for k, rx in _UNIT_RE if rx.match(rest)), None)
                pool = typed[kind] if kind else known
            if not any(abs(abs(k) - v) <= tol for k in pool):
                out.append(f"{tok} ({kind})" if kind else tok)
        return out

    def off_action_words(self, radio: str, act: dict) -> tuple[bool, list[str]]:
        """Whether `radio` says the action `act`, and what it asks for besides: instruction words
        of other kinds of action or outside the catalogue, and gates or doors that other actions
        of the zone restrict or open."""
        text = radio.lower()
        words = set(re.findall(r"[a-z]+", text))
        own = _ACTION_WORDS.get(act["kind"], set())
        target = act.get("target", "").lower()
        target_words = set(re.findall(r"[a-z]+", target))
        says = bool(words & own) or (act["kind"] == "announce" and target_words - {"the", "is", "a"} <= words)
        other = sorted(w for w in words & _INSTRUCTIONS if w not in own and w not in target_words)
        for x in self.catalogue.actions.get(self.zone, []):
            t = x.get("target", "").lower()
            if x["kind"] in ("meter_entry", "hold_entry", "open_exit") and t and t not in target \
                    and re.search(r"\b" + re.escape(t) + r"\b", text):
                other.append(t)
        return says, other

    def radio_problems(self, radio: str, act: dict | None = None) -> list[str]:
        out = []
        words = len(radio.split())
        if not radio:
            return ["the message is empty"]
        if words > RADIO_MAX_WORDS:
            out.append(f"{words} words; at most {RADIO_MAX_WORDS}")
        short = self.zone.split(" - ")[0].lower()
        if short not in radio.lower():
            out.append(f"it must name the zone ({self.zone.split(' - ')[0]})")
        if self.flagged_for_pushing and "push" not in radio.lower():
            out.append("the zone is flagged for pushing: say that pushing is detected")
        bad = self.ungrounded_numbers(radio)
        if bad:
            out.append(f"it states {', '.join(bad)}, which no tool returned in this episode as that kind of number")
        if act is not None:
            says, other = self.off_action_words(radio, act)
            name = f"{act['kind'].replace('_', ' ')} {act.get('target', '')}".strip()
            if not says:
                out.append(f"it must say the action proposed ({name})")
            if other:
                out.append(f"it asks for {', '.join(repr(w) for w in other)}, which is not the action proposed ({name}): "
                           "the steward reads this message out, so it asks for that action only")
        return out

    @property
    def done(self) -> bool:
        return self.proposal is not None or bool(self.closed_reason) or self.recalibration is not None


def brief(ep: Episode) -> str:
    acts = ep.catalogue.actions.get(ep.zone, [])
    return json.dumps({
        "zone": ep.zone, "tier": ep.tier.name, "why": ep.reasons,
        "camera": next((a.get("camera") for a in acts if a.get("camera")), None),
        "actions_available": [{k: v for k, v in a.items() if k != "camera"} for a in acts if a["kind"] != "none"],
    })


def run_model(ep: Episode, client=None, model: str = MODEL_ID) -> Episode:
    """Drive the episode with the model. Raises if the model endpoint is unreachable."""
    if client is None:
        client = mantle_client()
    kinds = sorted({a["kind"] for v in ep.catalogue.actions.values() for a in v})
    tools = tool_specs(kinds)
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": "Flagged zone:\n" + brief(ep)}]
    ep.trace.write("agent_start", episode=ep.id, zone=ep.zone, tier=ep.tier.name, planner="model", model=model)
    for step in range(MAX_STEPS):
        r = client.chat.completions.create(model=model, messages=msgs, tools=tools, max_completion_tokens=1500)
        m = r.choices[0].message
        calls = m.tool_calls or []
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        if not calls:
            if not ep.done:
                msgs.append({"role": "user", "content": "Use the tools: finish with propose_action, request_recalibration or no_action."})
                continue
            break
        for c in calls:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            out = ep.call(c.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out)})
        if ep.done:
            break
    ep.trace.write("agent_end", episode=ep.id, steps=step + 1, proposal=ep.proposal.id if ep.proposal else None,
                   closed=ep.closed_reason or None, recalibration=ep.recalibration)
    return ep


def run_deterministic(ep: Episode) -> Episode:
    """The same investigation without a model: fixed tool order, the venue-wide rule chooses."""
    ep.planner = "deterministic"
    ep.trace.write("agent_start", episode=ep.id, zone=ep.zone, tier=ep.tier.name, planner="deterministic")
    acts = [a for a in ep.catalogue.actions.get(ep.zone, []) if a["kind"] != "none"]
    cam = ep.camera
    state = ep.call("get_zone_state", {"zone": ep.zone})
    if cam:
        c = ep.call("check_camera", {"camera": cam})
        if not c.get("trusted", False):
            ep.call("request_recalibration", {"camera": cam, "reason": c.get("reason") or c.get("error") or "camera check failed"})
            ep.trace.write("agent_end", episode=ep.id, planner="deterministic", recalibration=ep.recalibration)
            return ep
    ep.call("cross_check_count", {"zone": ep.zone})
    spot, pushing = None, ep.flagged_for_pushing
    if pushing:
        hs = ep.call("locate_pushing", {"zone": ep.zone}).get("hotspots") or []
        spot = hs[0] if hs else None
    ep.call("measure_exit_flow", {"zone": ep.zone})
    for a in acts:
        ep.call("forecast_action", {"zone": ep.zone, "kind": a["kind"], "target": a.get("target", ""),
                                    "rate_per_min": a.get("rate_per_min", 60)})
    options = [ep._option(k) for k, v in ep.forecast_actions.items() if v["kind"] != "none"]
    if not options:
        ep.call("no_action", {"reason": "no action available"})
        return ep
    ch = choice.choose(options)
    best = ep.forecasts[ch.chosen.key]
    mins = state.get("minutes_to_4_5")  # the zone as it is now, not under the action
    rho = state.get("density", best.rho_now)
    where = (f", pushing {spot['side']} side {spot['metres_before_gate']:.1f} m before the gate" if spot
             else ", pushing detected, not in one spot" if pushing else "")
    radio = (f"{ep.zone.split(' - ')[0]}: {rho:.1f} per square metre"
             + (f", about {mins:.0f} minutes to 4.5" if mins is not None else ", not filling")
             + where + f". Proposed: {best.action}.")
    ep.call("propose_action", {"zone": ep.zone, "forecast_id": ch.chosen.key, "radio_message": radio,
                               "rationale": f"venue-wide rule over {len(options)} actions: {ch.reason}"})
    ep.trace.write("agent_end", episode=ep.id, planner="deterministic", proposal=ep.proposal.id if ep.proposal else None)
    return ep


def mantle_client():
    """OpenAI-compatible client for the Bedrock API endpoint, with a short-term bearer token
    minted from the caller's AWS credentials."""
    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI

    region = os.environ.get("AWS_REGION", "us-east-1")
    return OpenAI(base_url=MANTLE_URL, api_key=provide_token(region=region), timeout=30, max_retries=1)


def investigate(ep: Episode, use_model: bool = True) -> Episode:
    if use_model:
        try:
            run_model(ep)
            if ep.done:
                return ep
            ep.trace.write("agent_fallback", episode=ep.id, reason="model ended without a decision")
        except Exception as e:
            ep.trace.write("agent_fallback", episode=ep.id, reason=f"{type(e).__name__}: {e}")
        # a fresh deterministic pass over the same tools
        ep.forecasts.clear()
        ep.forecast_actions.clear()
        ep.proposal = None
    return run_deterministic(ep)
