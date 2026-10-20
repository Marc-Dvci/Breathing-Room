"""The evening report: what the system measured, proposed and did, and who decided, rendered
from the hash-chained trace for the safety officer who reviews the evening afterwards.

The report reads only the trace. Its first line states whether the chain re-hashes intact and
the hash of the last entry, so a printed copy can be matched to the record later.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

from .trace import compare_cloudwatch, verify, verify_signatures

SHOWN = {"tier", "drift", "drift_screen", "agent_start", "agent_end", "agent_fallback", "proposal", "decision",
         "verify", "expired", "demo", "scenario", "error", "start"}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _evening(entries: list[dict]) -> list[dict]:
    """Entries since the latest start of an evening (a restart or a new evening in the twin)."""
    starts = [i for i, e in enumerate(entries)
              if e["kind"] == "start" or (e["kind"] == "scenario" and "new evening" in str(e.get("event", "")))]
    return entries[starts[-1]:] if starts else entries


def _clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def _line(e: dict) -> tuple[str, str]:
    k = e["kind"]
    if k == "tier":
        return "Alert tier", f"{e['tier']}: " + "; ".join(e.get("reasons") or []) + f" (density {e.get('rho')}/m², pushing index {e.get('pressure')}/s²)"
    if k == "drift":
        return "Camera", ("trusted again" if e.get("trusted") else "not trusted") + (f": {e['reason']}" if e.get("reason") else "") + \
            f" (floor shift {e.get('floor_shift_m')} m, {e.get('method', 'features')})"
    if k == "drift_screen":
        return "Camera", f"ECC screen {e['screen'].get('verdict')}; {e.get('confirmed')} confirmed the camera {'steady' if e.get('trusted') else 'moved'}"
    if k == "agent_start":
        return "Assistant", f"investigation {e['episode']} starts ({e.get('planner')}{', ' + e['model'] if e.get('model') else ''}, tier {e.get('tier')})"
    if k == "agent_end":
        out = f"proposal {e['proposal']}" if e.get("proposal") else (
            f"recalibration requested: {e['recalibration'].get('reason', '')}" if e.get("recalibration") else
            f"closed without action: {e.get('closed') or ''}")
        return "Assistant", f"investigation {e['episode']} ends: {out}"
    if k == "agent_fallback":
        return "Assistant", f"deterministic planner takes over: {e.get('reason')}"
    if k == "proposal":
        a = e.get("action") or {}
        return "Proposal", f"{e['id']}: “{e.get('radio')}” ({a.get('kind')} {a.get('target', '')}, predicted peak {e.get('predicted', {}).get('peak_density')}/m²)"
    if k == "decision":
        return "Decision", f"{'approved' if e.get('approved') else 'rejected'} {e['proposal']} by {e.get('by')}" + (f"; {e['applied']}" if e.get("applied") else "")
    if k == "verify":
        return "Verification", f"{e['proposal']}: {e['outcome']} (measured {e['observed_rho']}/m² against {e['predicted_rho']}/m² predicted, after {e['elapsed_s']} s)"
    if k == "expired":
        return "Proposal", f"{e['proposal']} expired without a decision"
    if k == "demo":
        return "Demonstration", str(e.get("action"))
    if k == "scenario":
        return "Venue", str(e.get("event"))
    if k == "error":
        return "Error", f"{e.get('where')}: {e.get('error')}"
    return k, ""


def _witnesses(trace, ev: list[dict]) -> str:
    """The checks a rewrite of the whole file cannot pass: signatures, and the CloudWatch copy."""
    if trace is None:
        return ""
    out = []
    key = trace.public_key()
    if key:
        s = verify_signatures(trace.path, key["public_key"])
        out.append(("ok" if s.get("ok") else "bad", f"Signatures {'valid' if s.get('ok') else 'INVALID'}: {s['message']}."))
    cw = compare_cloudwatch(trace, since_ts=ev[0]["ts"] if ev else None)
    if cw.get("checked"):
        out.append(("ok" if cw.get("ok") else "bad", f"Independent copy: {cw['message']} (the console can only append to it)."))
    return "".join(f'<p class="chain {c}">{html.escape(t)}</p>' for c, t in out)


def render(path: Path, everything: bool = False, trace=None) -> str:
    entries = _load(path)
    ok, n, msg = verify(path) if entries else (True, 0, "empty trace")
    ev = entries if everything else _evening(entries)
    witnesses = _witnesses(trace, ev) if entries else ""
    props = {e["id"]: e for e in ev if e["kind"] == "proposal"}
    decisions = {e["proposal"]: e for e in ev if e["kind"] == "decision"}
    verifies = {e["proposal"]: e for e in ev if e["kind"] == "verify"}
    expired = {e["proposal"] for e in ev if e["kind"] == "expired"}
    calls: dict[str, list[dict]] = {}
    for e in ev:
        if e["kind"] == "tool_call":
            calls.setdefault(e["episode"], []).append(e)
    episode_of = {e["proposal"]: e["episode"] for e in ev if e["kind"] == "agent_end" and e.get("proposal")}
    episodes = [e for e in ev if e["kind"] == "agent_start"]
    fallbacks = sum(e["kind"] == "agent_fallback" for e in ev)
    lost = sum(e["kind"] == "drift" and not e.get("trusted") for e in ev)
    recal = sum(e["kind"] == "agent_end" and bool(e.get("recalibration")) for e in ev)
    approved = sum(d.get("approved") for d in decisions.values())
    rejected = len(decisions) - approved
    worked = sum(v["outcome"].startswith("working") for v in verifies.values())
    esc = html.escape

    def card(pid: str, p: dict) -> str:
        d, v = decisions.get(pid), verifies.get(pid)
        by = esc(str(d.get("by"))) + (" (no PIN: public demonstration)" if str(d.get("auth", "")).startswith("none") else "") if d else ""
        state = ("approved by " + by) if d and d.get("approved") else (
            "rejected by " + by if d else ("expired without a decision" if pid in expired else "pending"))
        steps = "".join(f"<li><b>{esc(c['tool'].replace('_', ' '))}</b> <span>{c['ms']:.0f} ms</span></li>"
                        for c in calls.get(episode_of.get(pid, ""), []))
        alts = "".join(f"<tr><td>{esc(a.get('kind', '').replace('_', ' '))} {esc(str(a.get('target', '')))}</td><td>{a.get('peak')}</td>"
                       f"<td>{'' if a.get('forecast_error') is None else '+' + str(a['forecast_error'])}</td>"
                       f"<td>{a.get('held_outside_worst', a.get('held_at_entrance')) or ''}</td>"
                       f"<td>{'' if not a.get('held_outside_worst') else (str(a['waiting_area_density_worst']) if a.get('waiting_area_density_worst') is not None else 'not declared')}</td></tr>"
                       for a in sorted(p.get("alternatives") or [], key=lambda a: a.get("peak", 0)))
        rule = p.get("choice") or {}
        return f"""<section class="prop">
  <h3>{esc(pid)} · {esc(p.get('zone', ''))} · {esc(p.get('tier', ''))} · {_clock(p['ts'])} UTC</h3>
  <p class="radio">“{esc(p.get('radio', ''))}”</p>
  <p>Planner: {esc(p.get('planner', ''))}. Rationale: {esc(p.get('rationale', ''))}</p>
  <table><thead><tr><th>Action compared</th><th>Predicted peak, /m²</th><th>Tested error</th><th>Held outside</th><th>Waiting area, /m²</th></tr></thead><tbody>{alts}</tbody></table>
  {f"<p><b>Venue-wide rule:</b> {esc(rule['reason'])}.</p>" if rule.get("reason") else ""}
  <p><b>Decision:</b> {state}.{f" <b>Verification:</b> {esc(v['outcome'])}, measured {v['observed_rho']}/m² against {v['predicted_rho']}/m² predicted after {v['elapsed_s']} s." if v else ""}</p>
  {f'<details><summary>Tool calls behind this proposal ({len(calls.get(episode_of.get(pid, ""), []))})</summary><ol class="calls">{steps}</ol></details>' if steps else ""}
</section>"""

    ep_zone = {e["episode"]: e.get("zone", "") for e in episodes}
    rows = "".join(f"<tr><td>{_clock(e['ts'])}</td><td>{esc(str(e.get('zone') or ep_zone.get(e.get('episode'), '')))}</td><td>{esc(_line(e)[0])}</td>"
                   f"<td>{esc(_line(e)[1])}</td></tr>" for e in ev if e["kind"] in SHOWN)
    head = entries[-1]["hash"] if entries else ""
    period = f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(ev[0]['ts']))} to {_clock(ev[-1]['ts'])} UTC" if ev else "no entries"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breathing Room · evening report</title><link rel="icon" href="/static/icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/static/report.css"></head><body><main>
<header><h1>Evening report</h1><p>Demonstration venue · Station Road and Entrance E1 · {esc(period)}</p>
<p class="chain {'ok' if ok else 'bad'}">Record {'intact' if ok else 'BROKEN'}: {esc(msg)}, {n} entries re-hashed. Last entry hash <code>{esc(head)}</code></p>
{witnesses}
<p class="noprint"><a href="/">Back to the console</a> · <a href="/report?all=1">whole record</a> · <button onclick="window.print()">Print or save as PDF</button></p></header>
<section class="kpis">
<div><b>{len(episodes)}</b><span>investigations</span></div>
<div><b>{len(props)}</b><span>proposals</span></div>
<div><b>{approved}</b><span>approved</span></div>
<div><b>{rejected}</b><span>rejected</span></div>
<div><b>{len(expired)}</b><span>expired</span></div>
<div><b>{worked} of {len(verifies)}</b><span>approved actions verified working</span></div>
<div><b>{lost}</b><span>times a camera lost trust</span></div>
<div><b>{recal}</b><span>recalibration requests</span></div>
<div><b>{fallbacks}</b><span>model fallbacks</span></div>
</section>
<h2>Proposals</h2>
{''.join(card(pid, p) for pid, p in props.items()) or '<p>No proposals in this period.</p>'}
<h2>Timeline</h2>
<table class="timeline"><thead><tr><th>UTC</th><th>Zone</th><th>Event</th><th>Detail</th></tr></thead><tbody>{rows}</tbody></table>
<p class="foot">Generated from the hash-chained trace at {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC. Each entry carries the SHA-256 of the previous one; any edited or deleted line breaks the chain above.{" A rewrite of the whole record is caught by the checks under it." if witnesses else ""}</p>
</main></body></html>"""
