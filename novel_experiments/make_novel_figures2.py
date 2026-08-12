#!/usr/bin/env python3
"""Figures for batches 3–4: N8 magnitude-insensitivity, N9 warning-vs-reflection, N7 robust-but-FPR."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
R = HERE / "results"
OUT = (HERE.parent.parent / "paper_chain" / "figures_pdf")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})


def _agg(name, key):
    by = defaultdict(list)
    for x in json.load(open(R / f"{name}_results.json")):
        by[x[key]].append(x["asr"])
    return {k: mean(v) for k, v in by.items()}


def n8_n15b():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    n8 = _agg("n8", "fake_rsi")
    xs = sorted(n8, reverse=True)              # 40 -> 15 (increasingly "extreme")
    ax.plot(xs, [n8[x] for x in xs], "o-", color="#d62728", label="N8: fake RSI in V4 panel")
    n15 = _agg("n15b", "miss_pct")
    xs2 = sorted(n15)
    ax2 = ax.twiny()
    ax2.plot(xs2, [n15[x] for x in xs2], "s--", color="#1f77b4", label="N15b: claimed miss %")
    ax.set_xlabel("fabricated RSI (more extreme →)")
    ax2.set_xlabel("claimed earnings-miss %")
    ax.set_ylabel("ASR")
    ax.set_ylim(0, 1.05)
    ax.set_title("Magnitude-insensitivity: form drives ASR, not the number")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_n8_magnitude.pdf")
    print("wrote fig_n8_magnitude.pdf")


def n9():
    rows = json.load(open(R / "n9_results.json"))
    conds = ["asr_base", "asr_warning", "asr_reflection"]
    labels = ["no defense", "upfront\nwarning", "post-hoc\nreflection"]
    vals = [mean([r[c] for r in rows]) for c in conds]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    colors = ["#7f7f7f", "#2ca02c", "#d62728"]
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(vals[0], ls=":", color="#7f7f7f", lw=1)
    ax.set_ylabel("mean ASR")
    ax.set_ylim(0, 0.7)
    ax.set_title("Reflection backfires; only an upfront prime helps")
    fig.tight_layout()
    fig.savefig(OUT / "fig_n9_mitigations.pdf")
    print("wrote fig_n9_mitigations.pdf")


def n7():
    rows = json.load(open(R / "n7_results.json"))
    variants = ["clean", "paraphrase", "authority"]
    undef = [mean([r["asr_undef"] for r in rows if r["variant"] == v]) for v in variants]
    deff = [mean([r["asr_def"] for r in rows if r["variant"] == v]) for v in variants]
    # benign FPR
    b = json.load(open(R / "n7b_results.json"))
    benign_block = mean([x["benign_block_rate"] for x in b])
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = range(len(variants))
    ax.bar([i - 0.2 for i in x], undef, width=0.4, label="ASR undefended", color="#d62728")
    ax.bar([i + 0.2 for i in x], deff, width=0.4, label="ASR after FinShield", color="#2ca02c")
    ax.axhline(benign_block, ls="--", color="#1f77b4", lw=1.5,
               label=f"benign FALSE-block = {benign_block:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("FinShield: robust to rephrasing, but blocks real news too")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_n7_defense.pdf")
    print("wrote fig_n7_defense.pdf")


if __name__ == "__main__":
    for fn in (n8_n15b, n9, n7):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"{fn.__name__} failed: {e}", file=sys.stderr)
