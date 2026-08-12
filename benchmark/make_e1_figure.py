#!/usr/bin/env python3
"""Make a figure from E1 (event-grounded benchmark) results once the run finishes."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[2] / "paper_chain"
FIG_DIR = PAPER_ROOT / "figures_pdf"
RESULTS_PATH = Path(__file__).parent / "results" / "e1_results.json"


CATEGORY_ORDER = ["CA", "GI", "RG", "EA", "MS", "MA", "NA"]
CAT_NAME = {
    "CA": "Corp.\nactions",
    "GI": "Govern.\n/ Insider",
    "RG": "Regulat.\n/ Legal",
    "EA": "Earnings\n/ Analyst",
    "MS": "Market\nstructure",
    "MA": "Macro\n/ Geopol.",
    "NA": "Narrative\n/ Sent.",
}

EMERGENT_RED = "#DC3232"
PIPE_BLUE = "#4682C8"
CHAIN_ORANGE = "#E6821E"
GRAY = "#A0A0A0"


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    if not RESULTS_PATH.exists():
        print(f"E1 results not found at {RESULTS_PATH}. Run E1 first.")
        return

    rows = json.load(RESULTS_PATH.open())
    print(f"Loaded {len(rows)} E1 rows")

    # Per-category mean ASR + best
    by_cat: dict[str, list[float]] = defaultdict(list)
    by_cat_best: dict[str, float] = defaultdict(float)
    by_cat_event: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["category"]
        by_cat[c].append(r["asr"])
        by_cat_best[c] = max(by_cat_best[c], r["asr"])
        by_cat_event[c][r["event"]].append(r["asr"])

    # Plot 1: per-category mean ASR bar
    fig, ax = plt.subplots(figsize=(4.90, 2.8))
    cats = [c for c in CATEGORY_ORDER if c in by_cat]
    means = [sum(by_cat[c]) / len(by_cat[c]) for c in cats]
    bests = [by_cat_best[c] for c in cats]
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w/2, means, w, label="Mean ASR", color=PIPE_BLUE, edgecolor="white", linewidth=0.6)
    ax.bar(x + w/2, bests, w, label="Best (max-prompt) ASR", color=EMERGENT_RED, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [CAT_NAME[c] for c in cats], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("ASR", fontsize=10)
    ax.grid(True, axis="y", linestyle=":", color=GRAY, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig_event_asr.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")

    # Print per-cat stats for the paper
    print("\nPer-category stats (for Table IV update):")
    print(f'{"Cat":4} {"Mean":>6} {"Best":>6} {"Strong-event":<20} {"Strong-event ASR":>18}')
    for c in cats:
        events = by_cat_event[c]
        ev_means = {e: sum(v)/len(v) for e, v in events.items()}
        strongest = max(ev_means, key=ev_means.get)
        print(f'{c:4} {sum(by_cat[c])/len(by_cat[c]):>6.3f} {by_cat_best[c]:>6.3f} {strongest:<20} {ev_means[strongest]:>18.3f}')


if __name__ == "__main__":
    main()
