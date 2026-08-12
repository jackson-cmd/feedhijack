#!/usr/bin/env python3
"""Make a figure from the E4 lexical-ablation results."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper_chain"
FIG_DIR = PAPER_ROOT / "figures_pdf"


VARIANTS_ORDER = [
    "base", "shorter", "longer", "no_breaking", "no_dollar_amount",
    "no_revenue_claim", "no_named_short", "weak_qualifier",
    "hedged", "with_skepticism",
    "no_source_noun", "no_urgency", "no_historical_claim",
    "single_buyer", "single_metric", "weak_signal",
    "fewer_allegations", "weaker_allegation", "no_dist_signal",
    "no_pct", "no_segment", "no_form4_jargon",
]

LEX_ORDER = [
    "LEX-FAKE-CEO-ARREST", "LEX-INSIDER-CLUSTER", "LEX-FDA-APPROVE",
    "LEX-EARNINGS-MISS", "LEX-SHORT-REPORT",
]
LEX_LABEL = {
    "LEX-FAKE-CEO-ARREST": "CEO arrest (AAPL)",
    "LEX-INSIDER-CLUSTER": "Cluster insider buy (TSLA)",
    "LEX-FDA-APPROVE":     "Regulatory approval (NVDA)",
    "LEX-EARNINGS-MISS":   "Earnings miss (META)",
    "LEX-SHORT-REPORT":    "Short report (AMZN)",
}

VARIANT_LABEL = {
    "base": "Base",
    "shorter": "Shorter",
    "longer": "Longer",
    "no_breaking": "No 'BREAKING'",
    "no_urgency": "No urgency cue",
    "no_source_noun": "No source noun",
    "hedged": "Hedged",
    "weak_qualifier": "Weak qualifier",
    "with_skepticism": "With skepticism",
    "no_dollar_amount": "No \\$ amount",
    "no_historical_claim": "No historical claim",
    "single_buyer": "Single buyer",
    "single_metric": "Single metric",
    "weak_signal": "Weak signal",
    "no_revenue_claim": "No revenue claim",
    "no_named_short": "No named short",
    "fewer_allegations": "Fewer allegations",
    "weaker_allegation": "Weaker allegation",
    "no_dist_signal": "No 'spreading on Slack'",
    "no_pct": "No %",
    "no_segment": "No segment detail",
    "no_form4_jargon": "No 'Form 4' jargon",
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    data = json.load(open(Path(__file__).parent / "results" / "e4_results.json"))
    by_lex = defaultdict(dict)
    for r in data:
        by_lex[r["lex_id"]][r["variant"]] = r["asr"]

    # Plot: heatmap with variants on y-axis, bases on x-axis
    variants = []
    for v in VARIANTS_ORDER:
        if any(v in by_lex[l] for l in LEX_ORDER):
            variants.append(v)
    M = np.full((len(variants), len(LEX_ORDER)), np.nan)
    for i, v in enumerate(variants):
        for j, lex in enumerate(LEX_ORDER):
            if v in by_lex[lex]:
                M[i, j] = by_lex[lex][v]

    fig, ax = plt.subplots(figsize=(4.90, 3.7))
    cmap = plt.get_cmap("Reds")
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(LEX_ORDER)),
                  [LEX_LABEL[l] for l in LEX_ORDER],
                  fontsize=9, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(variants)),
                  [VARIANT_LABEL[v] for v in variants],
                  fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#A0A0A0")
            else:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("ASR (R=5)", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "fig_lex_abl.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
