"""Charts v2 (campagne systèmes) — 5 nouveaux graphiques EN pour le README.

05_hero_gains        dumbbell avant→après par domaine (le tableau des gains, visuel)
06_imitation_trap    legacy codebase : RAG s'effondre, les poids tiennent
07_deliverables      livrables exécutés : 7B+LLML ≥ 14B-in-context
08_learning_loop     la boucle d'apprentissage : 0 → RAG immédiat → poids → système
09_swap_economics    multi-tenant : swap 2 ms vs reload (log)
Lance : python scripts/make_charts_v2.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ, "assets", "en")

EMER, EMER_LT = "#059669", "#34d399"
SLATE, SLATE_LT = "#94a3b8", "#cbd5e1"
INK, MUTE, RED, GRID = "#0f172a", "#64748b", "#ef4444", "#eef2f6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 13,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": INK,
})


def _clean(ax, xmax=110):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT)
    ax.tick_params(length=0)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, linewidth=1.4)
    ax.set_axisbelow(True)


def _foot(fig):
    fig.text(0.995, 0.012, "github.com/Romaindcrl/llml — all numbers measured locally",
             ha="right", va="bottom", fontsize=9.5, color=SLATE)


def hero():
    rows = [  # (label, before, after, note)
        ("Facts of a real project", 14, 86, "weights, 0 ctx"),
        ("Conventions amid legacy code", 36, 100, "vs best alt. (RAG)"),
        ("Unseen SDK — learned alone", 0, 62, "autonomous"),
        ("Open QA (routed to RAG)", 59, 94, ""),
        ("Working code to a 20k spec", 62, 81, "executed tests"),
        ("HumanEval — public benchmark", 92, 98, "verification loop"),
        ("Mixed workload (router)", 82, 96, ""),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ys = range(len(rows))[::-1]
    for y, (label, b, a, note) in zip(ys, rows):
        ax.plot([b, a], [y, y], color=SLATE_LT, lw=3, zorder=1)
        ax.scatter([b], [y], s=110, color=SLATE, zorder=2)
        ax.scatter([a], [y], s=150, color=EMER, zorder=3)
        ax.annotate("", xy=(a - 1.5, y), xytext=(b + 1.5, y),
                    arrowprops=dict(arrowstyle="-|>", color=EMER, lw=1.8))
        ax.text(a + 2.5, y, f"{a}%", va="center", fontsize=13, fontweight="bold", color=EMER)
        if b >= 8:
            ax.text(b - 2.5, y, f"{b}%", va="center", ha="right", fontsize=11.5, color=MUTE)
        else:
            ax.text(b + 1, y + 0.3, f"{b}%", va="bottom", ha="left", fontsize=11.5, color=MUTE)
        if note:
            ax.text(103, y - 0.32, note, va="center", ha="right", fontsize=9, color=SLATE)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=12.5, color=INK)
    _clean(ax, 112)
    ax.set_title("What LLML adds to a 7B — before → after, measured",
                 fontsize=16, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.04, "gray = 7B alone (or best alternative where noted) · green = 7B + LLML",
            transform=ax.transAxes, fontsize=11, color=MUTE)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.85, bottom=0.08)
    _foot(fig)
    fig.savefig(f"{OUT}/05_hero_gains.png", dpi=160)
    plt.close(fig)


def trap():
    labels = ["14B\neverything-in-ctx", "14B + RAG-spec", "7B + LLML", "14B + LLML"]
    vals = [100, 36, 100, 100]
    cols = [SLATE, RED, EMER, EMER]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
                fontsize=14, fontweight="bold", color=INK)
    ax.text(0, 88, "⚠ drops the\nfoundation module", ha="center", fontsize=9.5, color=RED)
    ax.text(1, 44, "retrieval can't fetch\npervasive rules", ha="center", fontsize=9.5, color=RED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT); ax.tick_params(length=0)
    ax.set_ylim(0, 112); ax.yaxis.grid(True, color=GRID, lw=1.4); ax.set_axisbelow(True)
    ax.set_ylabel("conventions respected (%)")
    ax.set_title("The imitation trap — window full of legacy code violating the standard",
                 fontsize=15, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.045, "20k-token spec · hard 32k window · overflow (spec+code = 44k) · both LLML arms keep the whole codebase",
            transform=ax.transAxes, fontsize=10.5, color=MUTE)
    plt.xticks(fontsize=11.5, color=INK)
    fig.subplots_adjust(bottom=0.13, top=0.82, left=0.09, right=0.97)
    _foot(fig)
    fig.savefig(f"{OUT}/06_imitation_trap.png", dpi=160)
    plt.close(fig)


def deliverables():
    labels = ["7B\nspec in context", "14B\nspec in context", "7B + LLML\ndecomposed", "14B + LLML\ndecomposed"]
    vals = [62, 78, 81, 81]
    ctx = ["20,333 tok/call", "20,333 tok/call", "0 tok", "0 tok"]
    cols = [SLATE, SLATE, EMER, EMER]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v, c in zip(bars, vals, ctx):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
                fontsize=14, fontweight="bold", color=INK)
        ax.text(b.get_x() + b.get_width() / 2, 6, c, ha="center", fontsize=9.5,
                color="white", fontweight="bold")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT); ax.tick_params(length=0)
    ax.set_ylim(0, 100); ax.yaxis.grid(True, color=GRID, lw=1.4); ax.set_axisbelow(True)
    ax.set_ylabel("behavioral asserts passed (executed)")
    ax.set_title("Delivering working code to a 20k-token spec",
                 fontsize=15, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.045, "modules generated then EXECUTED against a hidden behavioral test harness (16 asserts/entity)",
            transform=ax.transAxes, fontsize=10.5, color=MUTE)
    plt.xticks(fontsize=11.5, color=INK)
    fig.subplots_adjust(bottom=0.13, top=0.82, left=0.09, right=0.97)
    _foot(fig)
    fig.savefig(f"{OUT}/07_deliverables.png", dpi=160)
    plt.close(fig)


def learning():
    labels = ["before\n(unseen SDK)", "instant\n(RAG, t+50s)", "consolidated\n(weights, 0 ctx)", "expert system\n(weights+verify)"]
    vals = [0, 75, 62, 92]
    cols = [SLATE, EMER_LT, EMER, EMER]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center",
                fontsize=14, fontweight="bold", color=INK)
    ax.annotate("fails → reads the doc →\nwrites its own flashcards →\nretrains itself",
                xy=(2, 62), xytext=(0.55, 78), fontsize=10, color=MUTE,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT); ax.tick_params(length=0)
    ax.set_ylim(0, 105); ax.yaxis.grid(True, color=GRID, lw=1.4); ax.set_axisbelow(True)
    ax.set_ylabel("accuracy on the new domain")
    ax.set_title("The learning loop — zero human labels at every step",
                 fontsize=15, fontweight="bold", loc="left", pad=30)
    ax.text(0, 1.045, "verify-caught errors keep retraining the experts in production (drafts 8/12 → 10/12, autonomous)",
            transform=ax.transAxes, fontsize=10.5, color=MUTE)
    plt.xticks(fontsize=11.5, color=INK)
    fig.subplots_adjust(bottom=0.13, top=0.82, left=0.09, right=0.97)
    _foot(fig)
    fig.savefig(f"{OUT}/08_learning_loop.png", dpi=160)
    plt.close(fig)


def swap():
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    labels = ["reload the model", "hot-swap the expert (46 MB)"]
    vals = [1400, 2.4]
    bars = ax.barh(labels, vals, color=[SLATE, EMER], height=0.55)
    ax.set_xscale("log")
    ax.text(1400, 0, "  ~1.4 s", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(2.4, 1, "  ~2 ms  (~590× cheaper)", va="center", fontsize=13, fontweight="bold", color=EMER)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT); ax.tick_params(length=0)
    ax.set_xlim(1, 20000)
    ax.set_xlabel("switch time (ms, log scale)")
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=12.5, color=INK)
    ax.set_title("One frozen base, hundreds of tenants — switching experts is free",
                 fontsize=15, fontweight="bold", loc="left", pad=28)
    ax.text(0, 1.06, "measured: 46 MB/tenant · ~300 tenants on a 24 GB laptop · per-tenant isolation verified",
            transform=ax.transAxes, fontsize=10.5, color=MUTE)
    fig.subplots_adjust(left=0.30, right=0.96, top=0.78, bottom=0.18)
    _foot(fig)
    fig.savefig(f"{OUT}/09_swap_economics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    hero(); trap(); deliverables(); learning(); swap()
    print("OK ->", OUT, ": 05..09")
