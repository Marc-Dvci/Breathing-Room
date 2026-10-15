"""The steward's assistant: an agent that investigates a zone with vision tools, compares
actions with the physics, and puts one proposal in front of a person.

Division of labour:
- the tier comes from policy.TierPolicy, from measured numbers;
- every number in a proposal comes from a tool result (OpenCV measurements, physics.forecast);
- the model chooses which tools to call, in what order, which action to propose, and writes
  the radio message;
- a person approves or rejects. Nothing reaches a gate or a PA system without that.

A proposal must cite a forecast the agent computed in this episode, and its action must be in
the venue's catalogue. Anything else is refused by `Episode.propose` and the refusal goes back
to the model as a tool result.

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

from .physics import Action, Forecast
from .policy import Tier
from .trace import Trace

MODEL_ID = os.environ.get("BR_MODEL_ID", "openai.gpt-oss-120b")
MANTLE_URL = os.environ.get("BR_MANTLE_URL", "https://bedrock-mantle.us-east-1.api.aws/v1")
MAX_STEPS = 10


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

Rules:
1. Every number you state must come from a tool result in this conversation. The radio
   message is checked: a number no tool returned is refused.
2. Before proposing, confirm the measurement: call check_camera for the zone's camera and
   cross_check_count for the zone. If the camera is not trusted, call request_recalibration
   and do not propose a crowd action from that camera.
3. Compare at least two actions with forecast_action, then propose the one whose
   peak_with_error stays lowest with the least disruption. peak_with_error adds the error this
   forecast made for that action when tested against the crowd simulation; an action with a
   large measured error is less certain to work. Holding an entrance is the last resort.
   The forecast models crowd numbers only. When the zone is flagged for pushing and the
   density is not rising, a steward at the spot or an announcement is the proportionate
   action; their forecast equals the no-action forecast, and you say so.
4. propose_action must cite the forecast_id of the forecast that supports it.
5. The radio message is at most 30 words, plain English, and names the zone, the density,
   the minutes left, and the action. No jargon. The minutes left are minutes_to_4_5 from
   get_zone_state, the zone as it is now; never take them from an action's forecast. When
   minutes_to_4_5 is null the zone is not filling: say so, and never write "0 minutes". When
   the pushing index is at or above the alert threshold, say that pushing is detected.
6. You never approve anything yourself. A steward decides."""


def tool_specs(catalogue_actions: list[str]) -> list[dict]:
    def fn(name, desc, props, req):
        return {"type": "function", "function": {"name": name, "description": desc,
                                                 "parameters": {"type": "object", "properties": props, "required": req}}}
    return [
        fn("get_zone_state", "Current measured state of a zone: density, pushing index, trend, inflow, outflow.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("check_camera", "Check that the camera has not moved since calibration (ALIKED + LightGlue against the reference frame).",
           {"camera": {"type": "string"}}, ["camera"]),
        fn("cross_check_count", "Count the zone again with a second, independent counting model and report agreement.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("measure_exit_flow", "Measured flow through the zone's exits over the last 30 s, with capacity.",
           {"zone": {"type": "string"}}, ["zone"]),
        fn("forecast_action", "Predict the zone's density over the next 10 minutes under an action, with the mass balance. "
           f"kind is one of {catalogue_actions}. For meter_entry give rate_per_min; for open_exit the width comes from the venue.",
           {"zone": {"type": "string"}, "kind": {"type": "string"}, "target": {"type": "string"},
            "rate_per_min": {"type": "number"}}, ["zone", "kind"]),
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

    # -- tool dispatch -----------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        try:
            if name == "forecast_action":
                out = self._forecast(args)
            elif name == "propose_action":
                out = self._propose(args)
            elif name == "request_recalibration":
                self.recalibration = {"camera": args.get("camera", ""), "reason": args.get("reason", "")}
                out = {"ok": True, "message": "recalibration requested from the control room"}
            elif name == "no_action":
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
        self.calls.append(rec)
        self.trace.write("tool_call", **rec)
        return out

    def _forecast(self, a: dict) -> dict:
        zone = a.get("zone", self.zone)
        kind = a.get("kind", "none")
        target = a.get("target", "") or ""
        same_kind = [x for x in self.catalogue.actions.get(zone, []) if x["kind"] == kind]
        if kind != "none" and not target and len(same_kind) == 1:
            target = same_kind[0].get("target", "")  # one action of this kind: the target is not ambiguous
        if kind != "none" and not self.catalogue.allowed(zone, kind, target):
            opts = [f"{x['kind']}:{x.get('target', '')}" for x in self.catalogue.actions.get(zone, [])]
            return {"error": f"action {kind}:{target} is not available in {zone}; available: {opts}"}
        spec = next((x for x in self.catalogue.actions.get(zone, []) if x["kind"] == kind and x.get("target", "") == target), {})
        act = Action(kind, target, rate_pps=float(a.get("rate_per_min", spec.get("rate_per_min", 60))) / 60.0,
                     width_m=float(spec.get("width_m", 0)), share=float(spec.get("share", 1.0)))
        fc = self.tools["_forecast"](zone=zone, action=act)
        self.forecasts[fc.id] = fc
        self.forecast_actions[fc.id] = {"kind": kind, "target": target, "rate_per_min": round(act.rate_pps * 60),
                                       "width_m": act.width_m}
        mins = None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1)
        err = spec.get("forecast_error")
        self.forecast_actions[fc.id]["forecast_error"] = err
        return {"forecast_id": fc.id, "action": fc.action, "density_now": round(fc.rho_now, 2),
                "peak_density_10min": round(fc.peak_rho, 2), "minutes_to_threshold": mins,
                "measured_forecast_error": err,
                "peak_with_error": None if err is None else round(fc.peak_rho + err, 2),
                "threshold": fc.threshold, "net_rate_per_min": round(fc.net_rate_pps * 60, 1),
                "inflow_per_min": round(fc.inflow_pps * 60, 1), "outflow_per_min": round(fc.outflow_pps * 60, 1)}

    def _propose(self, a: dict) -> dict:
        fid = a.get("forecast_id", "")
        if fid not in self.forecasts:
            return {"error": "forecast_id not found: call forecast_action first and cite its forecast_id"}
        act = self.forecast_actions[fid]
        if act["kind"] == "none":
            return {"error": "the cited forecast is for no action; cite the forecast of the action you propose"}
        radio = (a.get("radio_message") or "").strip()
        if not radio or len(radio.split()) > 40:
            return {"error": "radio_message must be present and at most 30 words"}
        ungrounded = self.ungrounded_numbers(radio)
        if ungrounded:
            return {"error": f"radio_message states {', '.join(ungrounded)}, which no tool returned in this episode; "
                             "use only numbers from tool results"}
        fc = self.forecasts[fid]
        if not any(v["kind"] == "none" for v in self.forecast_actions.values()):
            self._forecast({"zone": a.get("zone", self.zone), "kind": "none"})  # the baseline the steward compares with
        alts = [{"forecast_id": k, **self.forecast_actions[k], "peak": round(f.peak_rho, 2),
                 "peak_with_error": None if self.forecast_actions[k].get("forecast_error") is None
                 else round(f.peak_rho + self.forecast_actions[k]["forecast_error"], 2),
                 "minutes_to_threshold": None if f.seconds_to_threshold is None else round(f.seconds_to_threshold / 60, 1)}
                for k, f in self.forecasts.items()]
        evidence = [c["tool"] for c in self.calls]
        self.proposal = Proposal(
            id="pr_" + uuid.uuid4().hex[:10], zone=a.get("zone", self.zone), tier=self.tier.name, action=act,
            forecast_id=fid, predicted={"peak_density": round(fc.peak_rho, 2), "curve_t": fc.curve_t,
                                        "curve_rho": [round(x, 3) for x in fc.curve_rho],
                                        "minutes_to_threshold": None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1)},
            alternatives=alts, radio=radio, rationale=(a.get("rationale") or "").strip(), evidence=evidence,
            created_t=self.clock(), planner=self.planner)
        return {"ok": True, "proposal_id": self.proposal.id, "message": "sent to the steward for approval"}

    def ungrounded_numbers(self, text: str) -> list[str]:
        """Numbers in `text` that match no number returned by a tool in this episode (or the
        zone's catalogue), at the precision written. Zone names and gate letters are skipped."""
        known: list[float] = []

        def walk(x):
            if isinstance(x, bool):
                return
            if isinstance(x, (int, float)):
                known.append(float(x))
            elif isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    walk(v)
            elif isinstance(x, str):
                known.extend(float(m) for m in re.findall(r"-?\d+(?:\.\d+)?", x))

        for c in self.calls:
            walk(c["result"])
        walk(self.catalogue.actions.get(self.zone, []))
        walk(self.zone)
        out = []
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)", text):
            tok = m.group(1)
            dec = len(tok.split(".")[1]) if "." in tok else 0
            v = float(tok)
            tol = 0.5 * 10 ** -dec + 1e-9
            if not any(abs(abs(k) - v) <= tol for k in known):
                out.append(tok)
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
    """The same investigation without a model: fixed tool order, lowest predicted peak wins."""
    ep.planner = "deterministic"
    ep.trace.write("agent_start", episode=ep.id, zone=ep.zone, tier=ep.tier.name, planner="deterministic")
    acts = [a for a in ep.catalogue.actions.get(ep.zone, []) if a["kind"] != "none"]
    cam = next((a.get("camera") for a in ep.catalogue.actions.get(ep.zone, []) if a.get("camera")), None)
    state = ep.call("get_zone_state", {"zone": ep.zone})
    if cam:
        c = ep.call("check_camera", {"camera": cam})
        if not c.get("trusted", True):
            ep.call("request_recalibration", {"camera": cam, "reason": c.get("reason", "camera moved")})
            ep.trace.write("agent_end", episode=ep.id, planner="deterministic", recalibration=ep.recalibration)
            return ep
    ep.call("cross_check_count", {"zone": ep.zone})
    ep.call("measure_exit_flow", {"zone": ep.zone})
    results = []
    for a in acts:
        out = ep.call("forecast_action", {"zone": ep.zone, "kind": a["kind"], "target": a.get("target", ""),
                                          "rate_per_min": a.get("rate_per_min", 60)})
        if "forecast_id" in out:
            disruption = {"send_steward": 0, "announce": 0, "open_exit": 0, "meter_entry": 1, "hold_entry": 2}.get(a["kind"], 3)
            ranked = out["peak_with_error"] if out.get("peak_with_error") is not None else out["peak_density_10min"]
            results.append((ranked, disruption, out))
    if not results:
        ep.call("no_action", {"reason": "no action available"})
        return ep
    # the least disruptive action whose peak (with its measured error) is within 0.25/m2 of the lowest
    lowest = min(r[0] for r in results)
    results.sort(key=lambda r: (r[1], r[0]) if r[0] <= lowest + 0.25 else (99, r[0]))
    best = results[0][2]
    mins = state.get("minutes_to_4_5")  # the zone as it is now, not under the action
    rho = state.get("density", best["density_now"])
    radio = (f"{ep.zone}: {rho:.1f} per square metre"
             + (f", about {mins:.0f} minutes to 4.5" if mins is not None else ", not filling")
             + f". Proposed: {best['action']}.")
    ep.call("propose_action", {"zone": ep.zone, "forecast_id": best["forecast_id"], "radio_message": radio,
                               "rationale": f"lowest predicted peak with its measured forecast error ({results[0][0]}/m2) "
                                            f"of {len(results)} actions compared"})
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
