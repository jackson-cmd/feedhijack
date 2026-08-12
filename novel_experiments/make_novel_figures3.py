#!/usr/bin/env python3
"""Publication figures for batch 5 (A: multi-agent propagation) and batch 6
(C: end-to-end P&L).  Writes PDFs to paper_chain/figures_pdf/.  Pure reads of the
results JSON / logs -- no API calls.  Missing inputs are skipped gracefully."""
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
plt.rcParams.update({"font.size": 9, "figure.dpi": 120})

C_SINGLE = "#888888"
C_LINEAR = "#7088c8"
C_STAR = "#e08830"
C_DEBATE = "#cc3232"
C_BENIGN = "#50a050"
C_ONESHOT = "#e08830"
C_PERSIST = "#cc3232"
TOPO_COLOR = {"single": C_SINGLE, "linear": C_LINEAR, "star": C_STAR, "debate": C_DEBATE}
VEC_SHORT = {"V1_fact": "V1", "V2_sentiment": "V2", "V3_selective": "V3",
             "V4_structured": "V4", "V5_multisource": "V5"}


def _load(name):
    p = R / f"{name}_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def _load_log(name):
    p = R / f"{name}_log.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# A: multi-agent propagation
# ---------------------------------------------------------------------------

def fig_a_amplification():
    """Grouped bars: single vs linear/star/debate mean ASR per vector. The money
    figure: hierarchical pipelines (linear/star) FILTER, peer debate AMPLIFIES."""
    data = _load("n5_pipeline")
    if not data:
        return
    rows = data["rows"]
    summ = data["summary"]
    vectors = list(summ["single_asr_by_vector"].keys())
    topos = ["single", "linear", "star", "debate"]

    single_by_vec = summ["single_asr_by_vector"]
    pipe_by_vec = {t: summ["by_topology"][t]["pipeline_asr_by_vector"] for t in topos if t != "single"}

    fig, ax = plt.subplots(figsize=(3.07, 2.5))
    x = list(range(len(vectors)))
    w = 0.20
    for i, t in enumerate(topos):
        if t == "single":
            y = [single_by_vec[v] for v in vectors]
        else:
            y = [pipe_by_vec[t][v] for v in vectors]
        ax.bar([xi + (i - 1.5) * w for xi in x], y, width=w, label=t, color=TOPO_COLOR[t])
    ax.set_xticks(x)
    ax.set_xticklabels([VEC_SHORT.get(v, v) for v in vectors], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Mean ASR", fontsize=10)
    ax.set_xlabel("Injection vector", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    # Reserve the top strip for the legend, then anchor the legend to the figure
    # (not the axes) so its one-row of 4 entries stays centered and unclipped on
    # the narrow single-column width.
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.995),
              bbox_transform=fig.transFigure, columnspacing=1.0,
              handletextpad=0.5, handlelength=1.4, borderaxespad=0.0)
    fig.savefig(FIGDIR / "fig_n_pipeline_amplification.pdf")
    print("wrote fig_n_pipeline_amplification.pdf")


def fig_a_delta_survival():
    """Per-topology mean delta (pipeline-single) + narrative survival, with the
    Spearman rank-order correlation annotated (tests framework.tex's assumption)."""
    data = _load("n5_pipeline")
    if not data:
        return
    summ = data["summary"]["by_topology"]
    topos = [t for t in ["linear", "star", "debate"] if t in summ]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    deltas = [summ[t]["mean_delta"] for t in topos]
    bars = ax.bar(topos, deltas, color=[TOPO_COLOR[t] for t in topos])
    for b, t in zip(bars, topos):
        d = summ[t]["mean_delta"]
        rho = summ[t]["spearman_rank_vs_single"]
        surv = summ[t]["mean_narrative_survival"]
        ax.text(b.get_x() + b.get_width() / 2, d + (0.01 if d >= 0 else -0.03),
                f"Δ={d:+.2f}\nρ={rho}\nsurv={surv:.2f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean ASR change vs single-shot")
    ax.set_xlabel("Pipeline topology")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n_pipeline_delta.pdf")
    print("wrote fig_n_pipeline_delta.pdf")


# ---------------------------------------------------------------------------
# C: end-to-end P&L
# ---------------------------------------------------------------------------

def fig_c_pnl_delta(tag="single"):
    """Per-ticker attack-induced dollar delta (one_shot vs persistent) vs benign."""
    data = _load(f"n6_pnl_{tag}")
    if not data:
        return
    per = data["summary"]["per_ticker"]
    tickers = list(per.keys())
    one = [per[t]["delta_vs_benign"].get("one_shot", {}).get("delta_dollars", 0) for t in tickers]
    per_ = [per[t]["delta_vs_benign"].get("persistent", {}).get("delta_dollars", 0) for t in tickers]

    fig, ax = plt.subplots(figsize=(3.07, 2.8))
    x = list(range(len(tickers)))
    w = 0.38
    ax.bar([xi - w / 2 for xi in x], one, width=w, label="one-shot", color=C_ONESHOT)
    ax.bar([xi + w / 2 for xi in x], per_, width=w, label="persistent", color=C_PERSIST)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=40, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Attack-induced P&L delta ($)", fontsize=10)
    ax.set_xlabel("Ticker", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig_n_pnl_delta_{tag}.pdf")
    print(f"wrote fig_n_pnl_delta_{tag}.pdf")


def fig_c_equity_curve(tag="single", ticker=None):
    """Equity curves benign vs one_shot vs persistent for one representative ticker."""
    logs = _load_log(f"n6_pnl_{tag}")
    if not logs:
        return
    # pick ticker with the largest |persistent delta| if not specified
    if ticker is None:
        data = _load(f"n6_pnl_{tag}")
        per = data["summary"]["per_ticker"] if data else {}
        cand = {t: abs(per[t]["delta_vs_benign"].get("persistent", {}).get("delta_dollars", 0))
                for t in per}
        ticker = max(cand, key=cand.get) if cand else (logs[0]["ticker"])

    curves = {}
    inj_day = None
    for rec in logs:
        if rec["ticker"] != ticker or rec["rep"] != 0:
            continue
        daily = rec["daily"]
        dates = [d["date"] for d in daily]
        eq = [d["equity"] for d in daily]
        curves[rec["scenario"]] = (dates, eq)
        for d in daily:
            if d["injected"] and inj_day is None:
                inj_day = d["date"]

    if not curves:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    style = {"benign": (C_BENIGN, "-"), "one_shot": (C_ONESHOT, "--"), "persistent": (C_PERSIST, "-.")}
    for scen, (col, ls) in style.items():
        if scen in curves:
            dates, eq = curves[scen]
            ax.plot(range(len(eq)), eq, color=col, linestyle=ls, label=scen, linewidth=1.8)
    if inj_day is not None and "benign" in curves:
        idx = curves["benign"][0].index(inj_day) if inj_day in curves["benign"][0] else None
        if idx is not None:
            ax.axvline(idx, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
            ax.text(idx, ax.get_ylim()[1], " inject", fontsize=7, va="top")
    ax.set_ylabel("Portfolio equity ($)")
    ax.set_xlabel(f"Trading day ({ticker})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / f"fig_n_pnl_equity_{tag}.pdf")
    print(f"wrote fig_n_pnl_equity_{tag}.pdf ({ticker})")


def fig_ac_flagship():
    """A x C: single-shot vs debate-pipeline attack-induced dollar delta."""
    ds = _load("n6_pnl_single")
    dd = _load("n6_pnl_debate")
    if not ds or not dd:
        return
    tickers = [t for t in dd["summary"]["per_ticker"]]
    def dget(data, t, scen):
        return data["summary"]["per_ticker"].get(t, {}).get("delta_vs_benign", {}).get(scen, {}).get("delta_dollars", 0)
    single_v = [dget(ds, t, "one_shot") for t in tickers]
    debate_v = [dget(dd, t, "one_shot") for t in tickers]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = list(range(len(tickers)))
    w = 0.38
    ax.bar([xi - w / 2 for xi in x], single_v, width=w, label="single-shot", color=C_SINGLE)
    ax.bar([xi + w / 2 for xi in x], debate_v, width=w, label="debate pipeline", color=C_DEBATE)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=15)
    ax.set_ylabel("One-shot induced P&L delta ($)")
    ax.set_xlabel("Ticker")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_n_pnl_flagship.pdf")
    print("wrote fig_n_pnl_flagship.pdf")


if __name__ == "__main__":
    for f in (fig_a_amplification, fig_a_delta_survival,
              lambda: fig_c_pnl_delta("single"), lambda: fig_c_equity_curve("single"),
              lambda: fig_c_pnl_delta("debate"), lambda: fig_c_equity_curve("debate"),
              fig_ac_flagship):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("fig failed:", getattr(f, "__name__", "lambda"), e)
