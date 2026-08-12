#!/usr/bin/env python3
"""Publication figures for the novel experiments (writes PDFs to paper_chain/figures_pdf/)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).parent / "results"
FIGDIR = (Path(__file__).parent.parent.parent / "paper_chain" / "figures_pdf")
FIGDIR.mkdir(parents=True, exist_ok=True)
MODEL_ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
MODEL_SHORT = {"gpt-3.5-turbo": "3.5T", "gpt-4o-mini": "4o-mini", "gpt-4.1-nano": "4.1-nano",
               "gpt-4.1-mini": "4.1-mini", "gpt-4o": "4o", "gpt-4.1": "4.1"}
plt.rcParams.update({"font.size": 10, "figure.dpi": 120})


def _load(name):
    p = R / f"{name}_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_n1():
    rows = _load("n1")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["vector"])].append(r["asr"])
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    vectors = sorted({r["vector"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = list(range(len(models)))
    for v in vectors:
        y = [mean(by[(m, v)]) if by[(m, v)] else float("nan") for m in models]
        ax.plot(x, y, marker="o", label=v, linewidth=1.8, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], rotation=20)
    ax.set_ylabel("Mean ASR")
    ax.set_xlabel("Base model (increasing capability →)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n1_scaling.pdf")
    print("wrote fig_n1_scaling.pdf")


def fig_n3():
    rows = _load("n3")
    if not rows:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["dose"]].append(r["asr"])
    doses = sorted(by)
    y = [mean(by[d]) for d in doses]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(doses, y, marker="o", color="#cc3232", linewidth=2, markersize=6)
    ax.set_xlabel("# corroborating fake sources")
    ax.set_ylabel("Mean ASR")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n3_dose.pdf")
    print("wrote fig_n3_dose.pdf")


def fig_n2():
    rows = _load("n2")
    if not rows:
        return
    conf = defaultdict(list)
    for r in rows:
        conf[r["outcome"]].append(r["confidence"])
    cats = [("benign", "benign"), ("hit", "successful\nattack"), ("miss", "failed\nattack")]
    vals = [mean(conf[c]) if conf[c] else 0 for c, _ in cats]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bars = ax.bar([l for _, l in cats], vals, color=["#50a050", "#cc3232", "#888888"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean self-reported confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n2_calibration.pdf")
    print("wrote fig_n2_calibration.pdf")


def fig_n4():
    rows = _load("n4")
    if not rows:
        return
    order = ["first", "middle", "last"]
    by = defaultdict(list)
    for r in rows:
        by[r["position"]].append(r["asr"])
    y = [mean(by[p]) if by[p] else 0 for p in order]
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bars = ax.bar(order, y, color=["#7088c8", "#b0b0b0", "#cc3232"])
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean ASR")
    ax.set_xlabel("Injection position in feed")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n4_position.pdf")
    print("wrote fig_n4_position.pdf")


if __name__ == "__main__":
    for f in (fig_n1, fig_n2, fig_n3, fig_n4):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("fig failed:", f.__name__, e)
