"""The venue's vision tools as a Model Context Protocol server (Streamable HTTP, JSON responses).

Any MCP client can read the live zones through the same functions the steward's assistant
calls: zone state, the ECC camera screen, pushing hotspots, exit flow, and forecasts of
catalogue actions. The server is read-only. Proposals and approvals stay in the console,
where a steward decides.

    POST /mcp   JSON-RPC 2.0: initialize, tools/list, tools/call, ping
"""

from __future__ import annotations

import json
from typing import Any

from . import choice
from .physics import Action

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "breathing-room", "title": "Breathing Room venue vision tools", "version": "1.1.0"}
INSTRUCTIONS = ("Read-only crowd-safety measurements from the demonstration venue: Entrance E1 (a recorded overhead camera "
                "processed live with OpenCV 5) and Zone C - Station Road (a JuPedSim venue twin). Densities are people per "
                "square metre; the pushing index is Helbing's crowd pressure in 1/s^2, alert threshold 0.11. "
                "Nothing here changes the venue: a steward approves actions in the console.")


def _zone_prop(zones: list[str]) -> dict:
    return {"type": "string", "enum": zones, "description": "zone name"}


def tool_list(zones: list[str], kinds: list[str]) -> list[dict]:
    z = _zone_prop(zones)
    ro = {"readOnlyHint": True, "openWorldHint": False}
    return [
        {"name": "list_zones", "description": "The venue's zones, their sources and the actions stewards can take in each.",
         "inputSchema": {"type": "object", "properties": {}}, "annotations": ro},
        {"name": "get_zone_state", "description": "Measured state of a zone now: density, pushing index, tier, trend, "
         "outflow and minutes until 4.5 people/m2 at the current inflow.",
         "inputSchema": {"type": "object", "properties": {"zone": z}, "required": ["zone"]}, "annotations": ro},
        {"name": "check_camera", "description": "Fast check that a zone's camera has not moved since calibration "
         "(cv.findTransformECC on the still background, converted to metres on the floor).",
         "inputSchema": {"type": "object", "properties": {"zone": z}, "required": ["zone"]}, "annotations": ro},
        {"name": "locate_pushing", "description": "Regions of a zone where the pushing index is above the alert threshold, "
         "with size, density, side and distance before the gate. Each region has a camera crop at /api/evidence/<id>.jpg.",
         "inputSchema": {"type": "object", "properties": {"zone": z}, "required": ["zone"]}, "annotations": ro},
        {"name": "measure_exit_flow", "description": "Measured flow through a zone's exits over the last 30 s, with capacity.",
         "inputSchema": {"type": "object", "properties": {"zone": z}, "required": ["zone"]}, "annotations": ro},
        {"name": "forecast_action", "description": "Predicted density over the next 10 minutes if a catalogue action were "
         "taken, from the zone mass balance, with the error that action's forecast made in testing.",
         "inputSchema": {"type": "object", "properties": {
             "zone": z, "kind": {"type": "string", "enum": ["none"] + kinds},
             "target": {"type": "string", "description": "the action's target, as list_zones gives it"},
             "rate_per_min": {"type": "number", "description": "for meter_entry"}}, "required": ["zone", "kind"]},
         "annotations": ro},
        {"name": "get_proposals", "description": "The assistant's latest proposals with their status, radio message, "
         "predicted effect, steward decision and verification.",
         "inputSchema": {"type": "object", "properties": {}}, "annotations": ro},
    ]


def call(rt, catalogue, name: str, args: dict) -> Any:
    zones = rt.zones()
    if name == "list_zones":
        return {"zones": [{"zone": n, "source": z.state.source,
                           "actions": [{k: v for k, v in a.items() if k != "camera"} for a in catalogue.actions.get(n, [])]}
                          for n, z in zones.items()]}
    if name == "get_proposals":
        with rt.lock:
            ps = sorted(rt.proposals.values(), key=lambda p: -p.created_t)[:5]
            return {"proposals": [{"id": p.id, "zone": p.zone, "status": p.status, "radio": p.radio, "action": p.action,
                                   "predicted_peak": p.predicted.get("peak_density"), "decided_by": p.decided_by,
                                   "verify": {k: v for k, v in p.verify.items() if k != "due_t"},
                                   "evidence_images": [f"/api/evidence/{e['id']}.jpg" for e in p.evidence_images]}
                                  for p in ps]}
    zone = args.get("zone")
    if zone not in zones:
        raise ValueError(f"unknown zone {zone!r}; zones: {list(zones)}")
    tools = rt.tools_for(zone)
    if name == "get_zone_state":
        return tools["get_zone_state"](zone)
    if name == "measure_exit_flow":
        return tools["measure_exit_flow"](zone)
    if name == "locate_pushing":
        return tools["locate_pushing"](zone)
    if name == "check_camera":
        z = zones[zone]
        if not hasattr(z, "screen_camera"):
            return {"note": "simulated zone: positions come from the twin, no camera to drift"}
        r = z.screen_camera()
        return {**r, "trusted": z.state.trusted, "trust_reason": z.state.trust_reason}
    if name == "forecast_action":
        kind, target = args.get("kind", "none"), args.get("target", "") or ""
        same = [a for a in catalogue.actions.get(zone, []) if a["kind"] == kind]
        if kind != "none" and not target and len(same) == 1:
            target = same[0].get("target", "")
        spec = next((a for a in same if a.get("target", "") == target), None)
        if kind != "none" and spec is None:
            raise ValueError(f"{kind}:{target} is not in {zone}'s catalogue; see list_zones")
        spec = spec or {}
        act = Action(kind, target, rate_pps=float(args.get("rate_per_min", spec.get("rate_per_min", 60))) / 60.0,
                     width_m=float(spec.get("width_m", 0)), share=float(spec.get("share", 1.0)))
        fc = tools["_forecast"](zone=zone, action=act)
        err = spec.get("forecast_error")
        priced = choice.Option(fc.id, kind, fc.peak_rho, err or 0.0, spec.get("forecast_error_max", err) or 0.0,
                               fc.held_back_peak, spec.get("queue_factor", 1.0), spec.get("holding_area_m2"))
        return {"action": fc.action, "density_now": round(fc.rho_now, 2), "peak_density_10min": round(fc.peak_rho, 2),
                "measured_forecast_error": err, "peak_with_error": None if err is None else round(fc.peak_rho + err, 2),
                "minutes_to_4_5": None if fc.seconds_to_threshold is None else round(fc.seconds_to_threshold / 60, 1),
                "inflow_per_min": round(fc.inflow_pps * 60, 1), "outflow_per_min": round(fc.outflow_pps * 60, 1),
                "people_held_at_entrance_10min": round(fc.held_back_peak), **priced.summary()}
    raise ValueError(f"unknown tool {name}")


def handle(rt, catalogue, msg: dict, trace=None) -> dict | None:
    """One JSON-RPC message. Returns the response, or None for a notification."""
    mid, method, params = msg.get("id"), msg.get("method", ""), msg.get("params") or {}
    if mid is None:
        return None  # notifications (notifications/initialized, cancelled) need no answer

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    if method == "initialize":
        want = params.get("protocolVersion", PROTOCOL_VERSIONS[0])
        return ok({"protocolVersion": want if want in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                   "capabilities": {"tools": {"listChanged": False}}, "serverInfo": SERVER_INFO,
                   "instructions": INSTRUCTIONS})
    if method == "ping":
        return ok({})
    if method == "tools/list":
        kinds = sorted({a["kind"] for v in catalogue.actions.values() for a in v})
        return ok({"tools": tool_list(list(rt.zones()), kinds)})
    if method == "tools/call":
        name, args = params.get("name", ""), params.get("arguments") or {}
        try:
            out = call(rt, catalogue, name, args)
            err = False
        except Exception as e:  # tool errors go back to the client as results, per the protocol
            out, err = {"error": f"{type(e).__name__}: {e}"}, True
        if trace is not None:
            trace.write("mcp_call", tool=name, args=args, error=err)
        return ok({"content": [{"type": "text", "text": json.dumps(out)}], "structuredContent": out, "isError": err})
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
