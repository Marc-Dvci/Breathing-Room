"use strict";
const $ = (s) => document.querySelector(s);
const TIER_SHAPE = { NORMAL: "●", WATCH: "◆", WARN: "▲", ACT: "■", CRITICAL: "✖" };
let last = null;
let pendingIds = new Set();

function fmtClock(t) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function esc(x) { return String(x ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }

function pushingWord(p) { if (p >= 0.18) return "strong pushing"; if (p >= 0.11) return "pushing"; if (p >= 0.06) return "restless"; return "calm"; }

function renderTiles(zones) {
  const el = $("#tiles");
  el.innerHTML = Object.values(zones).map((z) => {
    const filling = z.minutes_to_act != null;
    const mins = !filling ? "–" : (z.minutes_to_act > 30 ? "30+" : z.minutes_to_act.toFixed(1));
    const why = (z.reasons || []).slice(0, 2).join(" · ");
    return `<article class="tile tier-${z.tier} ${z.trusted ? "" : "untrusted"}">
      <h2>${esc(z.zone)}</h2>
      <span class="tier" aria-label="alert tier ${z.tier}">${TIER_SHAPE[z.tier] || ""} ${z.tier}</span>
      <div class="big">${z.rho.toFixed(1)}<small>people / m²</small></div>
      <div class="count"><div class="mins">${mins}</div><div class="lbl">${filling ? "min to 4.5 / m² at this inflow" : "not filling"}</div></div>
      <div class="facts"><span>Level <b>${z.los}</b></span><span><b>${pushingWord(z.pressure)}</b> (pushing index ${z.pressure.toFixed(3)} /s²)</span>
        <span><b>${Math.round(z.n)}</b> people on ${z.area_m2.toFixed(0)} m²</span>${z.trusted ? "" : `<span><b>camera not trusted</b></span>`}</div>
      <div class="why">${esc(z.trusted ? why : z.trust_reason)}</div>
      <div class="src">${esc(z.source || "")}</div>
    </article>`;
  }).join("");
}

function densColor(r) {
  // sequential: calm teal to crowded red
  const stops = [[0, [156, 201, 191]], [2, [143, 190, 160]], [3.5, [231, 163, 62]], [4.5, [214, 96, 46]], [5.5, [180, 35, 44]]];
  let a = stops[0], b = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) if (r >= stops[i][0] && r <= stops[i + 1][0]) { a = stops[i]; b = stops[i + 1]; }
  const t = Math.max(0, Math.min(1, (r - a[0]) / ((b[0] - a[0]) || 1)));
  const c = a[1].map((v, i) => Math.round(v + t * (b[1][i] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function fitCanvas(cv, aspect) {
  // draw at the displayed size times the pixel ratio, so text stays at its real size
  const dpr = window.devicePixelRatio || 1, w = cv.clientWidth || 900, h = Math.round(w * aspect);
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [g, w, h];
}

function renderPlan(fp) {
  const cv = $("#plan");
  const [g, W, H] = fitCanvas(cv, 0.3);
  const pad = 14;
  const sx = (W - 2 * pad) / 34, sy = (H - 2 * pad) / 7.4, s = Math.min(sx, sy);
  const X = (x) => pad + x * s, Y = (y) => H - pad - y * s;
  g.clearRect(0, 0, W, H);
  const poly = (pts, fill, stroke, dash) => {
    g.beginPath(); pts.forEach((p, i) => (i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])))); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.setLineDash(dash || []); g.strokeStyle = stroke; g.lineWidth = 1.5; g.stroke(); g.setLineDash([]); }
  };
  const o = fp.outline;
  poly(o.street, "#f1f3f5", "#9aa3ae");
  poly(o.main_exit, "#e7f3f1", "#0f766e");
  poly(o.side, fp.side_open ? "#e7f3f1" : "#eceff2", fp.side_open ? "#0f766e" : "#b6bec8", fp.side_open ? [] : [5, 4]);
  poly(o.zone_c, "rgba(15,118,110,0.06)", "#0f766e", [6, 4]);
  g.fillStyle = "#5b6472"; g.font = "12px Inter, Segoe UI, sans-serif";
  g.fillText("Zone C", X(15.3), Y(5.55));
  g.fillText("exit 2.4 m", X(30.2), Y(1.8) + 14);
  g.fillText(fp.side_open ? "door 2 OPEN" : "door 2 (closed)", X(30.2), Y(6.0) - 4);
  for (const [name, r] of Object.entries(o.gates)) {
    g.fillStyle = "#14181f"; g.fillRect(X(0) - 4, Y(r[3]), 4, (r[3] - r[1]) * s);
    const meter = fp.meters[name];
    g.fillStyle = "#5b6472";
    g.fillText(`${name}${meter != null ? ` · metered ${meter}/min` : ""}`, X(0.4), Y(r[1]) - 4);
  }
  const pts = fp.agents, rho = fp.agent_rho;
  for (let i = 0; i < pts.length; i++) {
    g.beginPath(); g.arc(X(pts[i][0]), Y(pts[i][1]), Math.max(2, 0.18 * s), 0, 6.283);
    g.fillStyle = densColor(rho[i] || 0); g.fill();
  }
  const bl = fp.backlog || {};
  $("#planinfo").innerHTML = `<span>Arrivals <b>${fp.arrivals_per_min}/min</b></span>` +
    Object.entries(bl).map(([k, v]) => `<span>Waiting outside ${esc(k)} <b>${v}</b></span>`).join("") +
    `<span>People in the street <b>${pts.length}</b></span>`;
}

function renderChart(zone, proposals) {
  const cv = $("#chart");
  const [g, W, H] = fitCanvas(cv, 0.3);
  const L = 30, R = 10, T = 12, B = 22;
  g.clearRect(0, 0, W, H);
  const ht = zone.history.t, hr = zone.history.rho;
  if (!ht.length) return;
  const tNow = ht[ht.length - 1];
  const t0 = Math.max(0, tNow - 480), t1 = tNow + 420;
  const ymax = 6;
  const X = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R), Y = (r) => H - B - (r / ymax) * (H - T - B);
  g.strokeStyle = "#e3e6eb"; g.lineWidth = 1; g.fillStyle = "#5b6472"; g.font = "12px Inter, Segoe UI, sans-serif";
  for (let r = 0; r <= ymax; r += 1) { g.beginPath(); g.moveTo(L, Y(r)); g.lineTo(W - R, Y(r)); g.stroke(); g.fillText(String(r), L - 14, Y(r) + 4); }
  [[3.5, "#b7791f", "WARN"], [4.5, "#c2410c", "ACT"]].forEach(([r, c, l]) => {
    g.strokeStyle = c; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(L, Y(r)); g.lineTo(W - R, Y(r)); g.stroke(); g.setLineDash([]);
    g.fillStyle = c; g.fillText(l, W - R - 34, Y(r) - 4);
  });
  g.strokeStyle = "#9aa3ae"; g.beginPath(); g.moveTo(X(tNow), T); g.lineTo(X(tNow), H - B); g.stroke();
  g.fillStyle = "#5b6472"; g.fillText("now", X(tNow) + 4, T + 10);
  for (let m = Math.ceil(t0 / 120) * 120; m <= t1; m += 120) g.fillText(fmtClock(m), X(m) - 12, H - 8);
  g.strokeStyle = "#14181f"; g.lineWidth = 2; g.beginPath();
  let started = false;
  for (let i = 0; i < ht.length; i++) { if (ht[i] < t0) continue; const x = X(ht[i]), y = Y(hr[i]); started ? g.lineTo(x, y) : (g.moveTo(x, y), (started = true)); }
  g.stroke();
  const p = proposals.find((q) => q.zone === zone.zone && (q.status === "pending" || q.status === "approved"));
  if (p) {
    const start = p.status === "approved" && p.verify && p.verify.start_t != null ? p.verify.start_t : p.created_t;
    g.strokeStyle = "#0f766e"; g.lineWidth = 2; g.setLineDash([7, 5]); g.beginPath();
    p.predicted.curve_t.forEach((dt, i) => { const x = X(start + dt), y = Y(p.predicted.curve_rho[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke(); g.setLineDash([]);
  }
}

function actionLabel(a) {
  if (a.kind === "none") return "no action";
  const k = a.kind.replace("_", " ");
  return `${k} ${a.target}${a.kind === "meter_entry" ? ` (${a.rate_per_min}/min)` : ""}`;
}

function proposalCard(p) {
  const alts = (p.alternatives || []).slice().sort((a, b) => a.peak - b.peak);
  const none = alts.find((a) => a.kind === "none");
  const v = p.verify || {};
  let status = "";
  if (p.status !== "pending") {
    const bad = /not working|rejected|no decision/.test(v.outcome || "");
    const head = { approved: "Approved", rejected: "Rejected", expired: "Expired" }[p.status] || p.status;
    const detail = v.outcome ? ` · ${esc(v.outcome)}${v.observed_rho != null ? ` (measured ${v.observed_rho}/m², predicted ${v.predicted_rho}/m² after ${v.elapsed_s} s)` : ""}`
      : p.status === "approved" ? " · checking the effect 90 s after the action" : "";
    status = `<div class="status ${bad ? "bad" : ""}">${head}${detail}</div>`;
  }
  return `<div class="prop" data-zone="${esc(p.zone)}">
    <div class="kicker"><span>${esc(p.zone)} · ${esc(p.tier)}</span><span>${p.planner === "model" ? "gpt-oss-120b" : "deterministic planner"}</span></div>
    <div class="radio">“${esc(p.radio)}”</div>
    <div class="effect">
      <div><div class="v">${p.predicted.peak_density.toFixed(1)}</div><div class="l">predicted peak in 10 min, people/m², with this action</div></div>
      <div><div class="v">${none ? none.peak.toFixed(1) : "–"}</div><div class="l">without action</div></div>
    </div>
    <table><thead><tr><th>Action compared</th><th>Peak</th><th>Min to 4.5</th></tr></thead><tbody>
      ${alts.map((a) => `<tr class="${a.forecast_id === p.forecast_id ? "chosen" : ""}"><td>${esc(actionLabel(a))}</td><td>${a.peak.toFixed(1)}</td><td>${a.minutes_to_threshold ?? "–"}</td></tr>`).join("")}
    </tbody></table>
    <div class="why">${esc(p.rationale)}</div>
    ${p.replay ? `<div class="note">This zone is a recording. Approving records the decision; the footage cannot change.</div>` : ""}
    ${p.status === "pending" ? `<div class="buttons"><button class="approve" data-id="${p.id}" data-v="approve">Approve</button><button class="reject" data-id="${p.id}" data-v="reject">Reject</button></div>` : status}
  </div>`;
}

function renderProposal(snap) {
  const el = $("#proposal");
  const recal = snap.recalibrations.filter((r) => !r.resolved).slice(-1)[0];
  let html = "";
  if (recal) html += `<div class="recal"><b>Recalibration requested · ${esc(recal.camera)}</b><br>${esc(recal.reason)}<br><span class="small-muted">Numbers from this camera are not used until a person recalibrates it.</span></div>`;
  const byZone = {};
  const rank = (p) => (p.status === "pending" ? 0 : p.status === "approved" && !(p.verify || {}).done ? 1 : 2);
  for (const p of snap.proposals) {
    const cur = byZone[p.zone];
    if (!cur || rank(p) < rank(cur) || (rank(p) === rank(cur) && p.created_t > cur.created_t)) byZone[p.zone] = p;
  }
  const cards = Object.values(byZone).sort((a, b) => rank(a) - rank(b));
  const active = snap.episodes.filter((e) => e.active);
  if (active.length) html += active.map((e) => `<p class="working">Investigating ${esc(e.zone)}…</p>`).join("");
  if (!cards.length && !active.length) html += `<p class="empty">No action needed. The assistant investigates any zone that reaches WARN.</p>`;
  html += cards.map(proposalCard).join("");
  // keep the DOM (and focus) stable while nothing changed
  if (el.dataset.html !== html) {
    el.innerHTML = html;
    el.dataset.html = html;
    el.querySelectorAll("button[data-id]").forEach((b) => (b.onclick = () => decide(b.dataset.id, b.dataset.v, b)));
  }
}

function summarise(c) {
  const r = c.result || {};
  if (r.error) return `refused: ${r.error}`;
  switch (c.tool) {
    case "get_zone_state": return `${r.density}/m², level ${r.level_of_service}, trend ${r.trend_per_min}/min`;
    case "check_camera": return r.trusted ? `camera steady (${r.floor_shift_m ?? 0} m${r.matches ? `, ${r.inliers}/${r.matches} LightGlue matches` : ""})` : `camera moved ${r.floor_shift_m} m`;
    case "cross_check_count": return `${r.count_primary} vs ${r.count_second} · ${r.agree ? "agree" : "disagree"}`;
    case "measure_exit_flow": return `outflow ${r.outflow_per_min}/min of ${r.exit_capacity_per_min}/min capacity`;
    case "forecast_action": return `${r.action}: peak ${r.peak_density_10min}/m²${r.minutes_to_threshold != null ? `, ${r.minutes_to_threshold} min to ${r.threshold}` : ""}`;
    case "propose_action": return "sent to the steward";
    case "request_recalibration": return "recalibration requested";
    case "no_action": return "closed without action";
    default: return JSON.stringify(r).slice(0, 80);
  }
}

function renderSteps(snap) {
  const ep = snap.episodes.slice(-1)[0];
  $("#stepzone").textContent = ep ? `· ${ep.zone}` : "";
  $("#steps").innerHTML = !ep ? `<li><span></span><span class="res">Nothing to investigate yet.</span></li>` :
    ep.calls.map((c, i) => `<li><span class="n">${i + 1}</span><span><span class="tool">${esc(c.tool.replaceAll("_", " "))}</span><br><span class="res">${esc(summarise(c))}</span></span><span class="ms">${Math.round(c.ms)} ms</span></li>`).join("");
}

async function renderLog() {
  try {
    const r = await fetch("/api/trace?n=40"); const t = await r.json();
    $("#log").innerHTML = t.reverse().filter((e) => e.kind !== "tool_call").map((e) => {
      const time = new Date(e.ts * 1000).toLocaleTimeString();
      let msg = "";
      if (e.kind === "tier") msg = `${e.zone} → ${e.tier} (${(e.reasons || []).join("; ")})`;
      else if (e.kind === "proposal") msg = `proposal ${e.id}: ${e.radio}`;
      else if (e.kind === "decision") msg = `${e.approved ? "approved" : "rejected"} ${e.proposal} ${e.applied ? "· " + e.applied : ""}`;
      else if (e.kind === "verify") msg = `${e.zone}: ${e.outcome} (measured ${e.observed_rho}, predicted ${e.predicted_rho})`;
      else if (e.kind === "agent_start") msg = `assistant investigates ${e.zone} (${e.planner}${e.model ? ", " + e.model : ""})`;
      else if (e.kind === "agent_fallback") msg = `model unavailable, deterministic planner: ${e.reason}`;
      else if (e.kind === "expired") msg = `${e.zone}: proposal ${e.proposal} expired without a decision`;
      else if (e.kind === "scenario") msg = `${e.zone}: ${e.event}`;
      else if (e.kind === "demo") msg = e.action;
      else msg = JSON.stringify(Object.fromEntries(Object.entries(e).filter(([k]) => !["ts", "prev", "hash", "kind"].includes(k)))).slice(0, 140);
      return `<li>${time} <b>${esc(e.kind)}</b> ${esc(msg)} <span title="${e.hash}">#${e.hash.slice(0, 8)}</span></li>`;
    }).join("");
  } catch (_) { /* the log is secondary */ }
}

async function decide(id, verb, btn) {
  if (btn) btn.disabled = true;
  const r = await fetch(`/api/proposals/${id}/${verb}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ who: "steward" }) });
  if (!r.ok) { const m = (await r.json()).detail || "could not record the decision"; $("#proposal").insertAdjacentHTML("afterbegin", `<p class="err">${esc(m)}</p>`); }
  tick();
}

async function tick() {
  try {
    const r = await fetch("/api/state", { cache: "no-store" });
    const s = await r.json(); last = s;
    $("#conn").classList.add("on");
    const zc = s.zones["Zone C - Station Road"];
    $("#clock").textContent = `t = ${fmtClock(zc ? zc.t : 0)}`;
    const ep = s.episodes.slice(-1)[0];
    $("#planner").textContent = ep ? (ep.planner === "model" ? "assistant: gpt-oss-120b on Amazon Bedrock" : "assistant: deterministic planner") : "assistant ready";
    renderTiles(s.zones);
    renderPlan(s.floorplan);
    if (zc) renderChart(zc, s.proposals);
    renderProposal(s);
    renderSteps(s);
    if (s.camera_timing_ms != null) $("#camms").textContent = `${s.camera_timing_ms} ms of processing per second of video.`;
    $("#speed").textContent = s.speed > 1 ? "Normal speed" : "Fast forward ×4";
    $("#speed").dataset.next = s.speed > 1 ? "speed1" : "speed4";
  } catch (e) { $("#conn").classList.remove("on"); }
}

const demo = (a) => fetch(`/api/demo/${a}`, { method: "POST" }).then(tick);
$("#bump").onclick = () => demo("bump");
$("#unbump").onclick = () => demo("unbump");
$("#speed").onclick = () => demo($("#speed").dataset.next || "speed4");
$("#restart").onclick = () => demo("restart");
setInterval(tick, 1000);
setInterval(renderLog, 3000);
setInterval(() => { $("#cam").src = `/api/camera.jpg?t=${Date.now()}`; }, 1000);
tick(); renderLog();
