"""Result figures for the report, from the JSON files in eval/results/.

    python eval/figures.py      # writes docs/figures/fig_*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
FIG = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(FIG / name)


def fig_test_a():
    d = json.loads((RES / "test_a_scores.json").read_text())["sources"]["video (heads)"]
    bands = d["auc_pressure_by_density_band"]
    labels = [b.replace("5-9", "5+") for b in bands]
    auc = [bands[b]["auc"] for b in bands]
    n = [bands[b]["n"] for b in bands]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(labels))
    ax.bar(x, auc, width=0.56, color=BLUE, zorder=2)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(len(labels) - 0.5, 0.505, "chance", color=MUTED, fontsize=10, ha="right", va="bottom")
    for i, (a, k) in enumerate(zip(auc, n)):
        ax.text(i, a + 0.012, f"{a:.2f}", ha="center", color=INK, fontsize=11)
        ax.text(i, 0.415, f"n={k:,}", ha="center", color="white", fontsize=9)
    ax.set_xticks(x, [f"{lab} /m²" for lab in labels])
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("AUROC, pushing vs walking")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_test_a_bands.png")


def fig_twin():
    d = json.loads((RES / "test_b_twin.json").read_text())
    seed = d["seeds"]["0"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [("no action", MUTED), ("open second exit door", BLUE), ("meter both gates to 150/min", AQUA)]
    for name, col in names:
        r = seed["actions"].get(name)
        if not r:
            continue
        if name == "no action":
            base = np.array(seed["baseline_rho"])
            k0 = int(seed["warning_s"] - (len(seed["baseline_rho"]) - len(base)))
            sim = base[k0 - 1:k0 + 300]
        else:
            sim = np.array(r["rho_curve"])
        ax.plot(np.arange(len(sim)) / 60, sim, color=col, lw=2, label=f"{name}: simulated")
        if "pred_curve" in r:
            ax.plot(np.array(r["pred_t"]) / 60, r["pred_curve"], color=col, lw=1.6, ls=(0, (5, 4)))
    ax.axhline(4.0, color=ORANGE, lw=1, ls=(0, (2, 3)))
    ax.text(5.0, 4.08, "4.0 /m²", color=ORANGE, fontsize=10, ha="right")
    ax.set_xlim(0, 5)
    ax.set_xlabel("minutes after the warning")
    ax.set_ylabel("Zone C density, people/m²")
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_twin_actions.png")


def fig_flux():
    d = json.loads((RES / "test_e1_flux.json").read_text())
    t = np.arange(len(d["video"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(t, d["reference"], color=MUTED, lw=2, label="trajectories (reference)")
    ax.plot(t, d["video"], color=BLUE, lw=2, label="video: heads across the gate line")
    ax.set_xlabel("seconds of run 150")
    ax.set_ylabel("people per second")
    ax.set_ylim(0, 1.6)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, "fig_e1_outflow.png")


def fig_venues():
    d = json.loads((RES / "test_b_venues.json").read_text())
    names = list(d["layouts"])
    choice = d["summary"]["choice"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        lead = [r["lead_time_s"] / 60 for r in d["venues"][name].values() if r.get("lead_time_s") is not None]
        a.scatter(lead, [y] * len(lead), s=46, color=BLUE, alpha=0.85, lw=0, zorder=2)
        runs = [c for c in choice if c["venue"] == name]
        b.scatter([c["regret_plain"] for c in runs], [y] * len(runs), s=46, color="#b9b9b4", lw=0, zorder=2)
        b.scatter([c["regret_with_error"] for c in runs], [y] * len(runs), s=46, color=BLUE, alpha=0.85, lw=0, zorder=3)
    a.set_yticks(range(len(names)), list(reversed(names)))
    a.set_xlim(0, 6)
    a.set_xlabel("minutes of warning before 4 people/m²")
    b.set_xlim(-0.1, 2.6)
    b.set_xlabel("regret of the chosen action, people/m²")
    b.text(0.06, len(names) - 0.45, "with measured error", color=BLUE, fontsize=9.5)
    b.text(2.55, len(names) - 0.45, "forecast alone", color=MUTED, fontsize=9.5, ha="right")
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(names) - 0.1)
    b.spines["left"].set_visible(False)
    save(fig, "fig_venues.png")


if __name__ == "__main__":
    fig_test_a()
    fig_flux()
    if (RES / "test_b_twin.json").exists():
        fig_twin()
    if (RES / "test_b_venues.json").exists():
        fig_venues()
