"""Graphiques des benchmarks (PNG) — bilingue FR (LinkedIn) + EN (README), style soigné.

LLML = RAG + poids combinés (faits par RAG, style par poids). Les méthodes seules sont des
ABLATIONS, pas « notre solution ». Sortie : assets/fr/*.png et assets/en/*.png.
Lance : python scripts/make_charts.py   (génère les deux langues)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMER, EMER_LT = "#059669", "#34d399"      # LLML
SLATE, SLATE_LT = "#94a3b8", "#cbd5e1"    # ablations / concurrents
INK, MUTE, RED, GRID = "#0f172a", "#64748b", "#ef4444", "#eef2f6"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 13.5,
    "text.color": INK, "axes.labelcolor": MUTE, "xtick.color": INK, "ytick.color": MUTE,
})

T = {
  "fr": {
    "c1_title": "Seul LLML respecte conventions ET faits",
    "c1_conv": "Conventions (pervasives)", "c1_facts": "Faits spécifiques",
    "c1_ylab": "Exactitude (%)", "c1_m": ["RAG", "Compaction", "LLML"],
    "c1_n": ["seul · rate les\nconventions", "seule · perd\nles faits",
             "RAG + poids ·\nles deux ✓ (0 ctx)"],
    "c2_title": "Le cahier coûte 150 tokens — pas 20 000",
    "c2_sub": "la fenêtre de contexte reste libre pour ton code",
    "c2_ylab": "Tokens de contexte pour le cahier, par appel",
    "c2_m": ["Compaction", "RAG", "LLML\n(RAG + poids)"], "c2_note": "≈ 130× moins",
    "c3_title": "Le code remplit la fenêtre → la compaction lâche les faits",
    "c3_sub": "LLML tient à 100 % : le cahier vit dans les poids, pas dans le contexte",
    "c3_xlab": "Code projet déjà dans la fenêtre 32k (tokens)",
    "c3_ylab": "Exactitude des faits (%)", "c3_comp": "Compaction", "c3_ours": "LLML (RAG + poids)",
    "c4_title": "Les poids SEULS ne rappellent pas les faits",
    "c4_sub": "…donc LLML garde les faits dans le RAG (et le style dans les poids)",
    "c4_ylab": "Exactitude SQuAD (vraies questions)",
    "c4_m": ["Poids seuls\n(ablation)", "Modèle\nnu", "Compaction", "RAG\n= voie « faits »\nde LLML"],
    "c4_n1": "pas ce qu'on\nutilise pour les faits", "c4_n2": "ce que LLML\nfait vraiment",
  },
  "en": {
    "c1_title": "Only LLML gets conventions AND facts",
    "c1_conv": "Conventions (pervasive)", "c1_facts": "Specific facts",
    "c1_ylab": "Accuracy (%)", "c1_m": ["RAG", "Compaction", "LLML"],
    "c1_n": ["alone · misses\nthe conventions", "alone · loses\nthe facts",
             "RAG + weights ·\nboth ✓ (0 ctx)"],
    "c2_title": "The spec costs 150 tokens — not 20,000",
    "c2_sub": "the context window stays free for your code",
    "c2_ylab": "Context tokens for the spec, per call",
    "c2_m": ["Compaction", "RAG", "LLML\n(RAG + weights)"], "c2_note": "≈ 130× less",
    "c3_title": "As code fills the window, compaction loses the facts",
    "c3_sub": "LLML holds at 100%: the spec lives in the weights, not the context",
    "c3_xlab": "Project code already in the 32k window (tokens)",
    "c3_ylab": "Fact accuracy (%)", "c3_comp": "Compaction", "c3_ours": "LLML (RAG + weights)",
    "c4_title": "Weights ALONE can't recall facts",
    "c4_sub": "…so LLML keeps facts in RAG (and style in the weights)",
    "c4_ylab": "SQuAD accuracy (real questions)",
    "c4_m": ["Weights only\n(ablation)", "Base\nmodel", "Compaction", "RAG\n= LLML's\nfact route"],
    "c4_n1": "not what we\nuse for facts", "c4_n2": "what LLML\nactually does",
  },
}


def _clean(ax, ymax=110, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(SLATE_LT)
    ax.tick_params(length=0)
    ax.set_ylim(0, ymax)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=1.4); ax.set_axisbelow(True)


def _title(ax, main, sub=None):
    ax.set_title(main, fontsize=16.5, fontweight="bold", color=INK, pad=44 if sub else 30, loc="left")
    if sub:
        ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=12.5, color=MUTE, ha="left")


def _foot(fig):
    fig.text(0.995, 0.012, "github.com/Romaindcrl/llml", ha="right", va="bottom",
             fontsize=10, color=SLATE)


def _vlabels(ax, bars, dy=2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, f"{h:.0f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK)


def build(lang, outdir):
    L = T[lang]
    os.makedirs(outdir, exist_ok=True)

    # 1) conventions ET faits
    x = range(3); w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    b1 = ax.bar([i - w / 2 for i in x], [29, 91, 100], w, label=L["c1_conv"],
                color=[SLATE, SLATE, EMER])
    b2 = ax.bar([i + w / 2 for i in x], [100, 0, 100], w, label=L["c1_facts"],
                color=[SLATE_LT, SLATE_LT, EMER_LT])
    _vlabels(ax, b1); _vlabels(ax, b2)
    ax.set_xticks(list(x)); ax.set_xticklabels(L["c1_m"], fontsize=15, color=INK, fontweight="bold")
    _clean(ax); ax.set_ylabel(L["c1_ylab"]); _title(ax, L["c1_title"])
    for i, (note, col) in enumerate(zip(L["c1_n"], [RED, RED, EMER])):
        ax.text(i, -21, note, ha="center", fontsize=10.5, color=col,
                fontweight="bold" if i == 2 else "normal")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=12)
    fig.subplots_adjust(bottom=0.30, top=0.83, left=0.09, right=0.97); _foot(fig)
    fig.savefig(f"{outdir}/01_conventions_et_faits.png", dpi=160); plt.close(fig)

    # 2) coût de contexte
    ctx = [20382, 20484, 150]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    bars = ax.bar(L["c2_m"], ctx, color=[SLATE, SLATE, EMER], width=0.6)
    for b, v in zip(bars, ctx):
        ax.text(b.get_x() + b.get_width() / 2, v + 350, f"{v:,}".replace(",", " "),
                ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
    _clean(ax, ymax=23000); ax.set_ylabel(L["c2_ylab"]); _title(ax, L["c2_title"], L["c2_sub"])
    ax.annotate(L["c2_note"], xy=(2, 1400), xytext=(1.45, 12000), fontsize=15, fontweight="bold",
                color=EMER, arrowprops=dict(arrowstyle="-|>", color=EMER, lw=2.2))
    fig.subplots_adjust(bottom=0.12, top=0.82, left=0.11, right=0.97); _foot(fig)
    fig.savefig(f"{outdir}/02_cout_contexte.png", dpi=160); plt.close(fig)

    # 3) sous charge
    loads = [0, 12, 22]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(loads, [50, 0, 0], "-o", color=SLATE, lw=3, ms=11, label=L["c3_comp"],
            markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=SLATE)
    ax.plot(loads, [100, 100, 100], "-o", color=EMER, lw=3.4, ms=12, label=L["c3_ours"])
    for xx, yy in zip(loads, [50, 0, 0]):
        ax.text(xx, yy - 9, f"{yy}%", ha="center", fontsize=12, color=MUTE, fontweight="bold")
    for xx in loads:
        ax.text(xx, 103.5, "100%", ha="center", fontsize=12, color=EMER, fontweight="bold")
    _clean(ax, ygrid=True); ax.set_ylim(-16, 112); ax.set_xlim(-2, 24)
    ax.set_xticks(loads); ax.set_xticklabels([f"{l}k" for l in loads], color=INK)
    ax.set_xlabel(L["c3_xlab"]); ax.set_ylabel(L["c3_ylab"]); _title(ax, L["c3_title"], L["c3_sub"])
    ax.legend(frameon=False, fontsize=12.5, loc="center right")
    fig.subplots_adjust(bottom=0.13, top=0.82, left=0.09, right=0.97); _foot(fig)
    fig.savefig(f"{outdir}/03_sous_charge.png", dpi=160); plt.close(fig)

    # 4) pourquoi hybride
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    bars = ax.bar(L["c4_m"], [34, 59, 72, 88], color=[SLATE_LT, SLATE, SLATE, EMER], width=0.62)
    _vlabels(ax, bars)
    _clean(ax); ax.set_ylabel(L["c4_ylab"]); _title(ax, L["c4_title"], L["c4_sub"])
    ax.text(0, 52, L["c4_n1"], ha="center", fontsize=10, color=RED)
    ax.text(3, 78, L["c4_n2"], ha="center", fontsize=10, color=EMER, fontweight="bold")
    fig.subplots_adjust(bottom=0.18, top=0.80, left=0.09, right=0.97); _foot(fig)
    fig.savefig(f"{outdir}/04_pourquoi_hybride.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    import shutil
    flat = os.path.join(_PROJ, "assets")
    for f in os.listdir(flat) if os.path.isdir(flat) else []:
        if f.endswith(".png"):
            os.remove(os.path.join(flat, f))   # vire les anciens PNG à plat
    for lg in ("fr", "en"):
        build(lg, os.path.join(_PROJ, "assets", lg))
    print("OK ->", os.path.join("assets", "fr"), "+", os.path.join("assets", "en"))
