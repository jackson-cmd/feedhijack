#!/usr/bin/env python3
"""
Generate figures for the benchmark paper from the real ASR data.

Inputs:
    paper_chain/data/raw_asr.json   (aggregated from existing experiment)

Outputs (PDFs in paper_chain/figures_pdf/):
    fig_overall_per_ticker.pdf      bar chart of II vs CI mean ASR per ticker
    fig_subtype_heatmap.pdf         subtype × ticker heatmap
    fig_state_dependence.pdf        BUY vs HOLD baseline comparison per vector
    fig_top_attacks.pdf             top 10 attack prompts ranked by mean ASR
    fig_action_flow.pdf             baseline → attacked decision flow (Sankey-ish)

On-page sizing.  The paper (USENIX, 10pt body) shows these as full-width
``figure*`` floats at 0.7*\textwidth, and \textwidth = 7.00in, i.e. a fixed
on-page width of 4.90in.  We therefore save each PDF at exactly 4.90in wide
(figsize width = 4.90 and NO bbox_inches="tight", so the saved size is
deterministic).  With that, matplotlib point sizes equal on-page point sizes,
so the font hierarchy below (<= 10pt = body) renders at body size on the page:
axis labels 10, tick labels 9, legend 9, in-plot annotations 8.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# The paper lived in paper_chain/ before moving to the repo root. Try both and
# use whichever holds data/raw_asr.json.
_CAND = [Path(__file__).resolve().parents[2] / "paper_chain",
         Path(__file__).resolve().parents[2]]
PAPER_ROOT = next((c for c in _CAND if (c / "data" / "raw_asr.json").exists()), _CAND[-1])
DATA_PATH = PAPER_ROOT / "data" / "raw_asr.json"
FIG_DIR = PAPER_ROOT / "figures_pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# Match TikZ palette in main.tex
EMERGENT_RED = "#DC3232"
BENIGN_GREEN = "#32A032"
CHAIN_ORANGE = "#E6821E"
PIPE_BLUE = "#4682C8"
GRAY = "#A0A0A0"


# On-page display width in inches (0.7 * \textwidth, \textwidth = 7.00in).
# Saving at exactly this width makes matplotlib pt == on-page pt.
FIG_W = 4.90
# Font hierarchy, in points; never exceeds the 10pt body size.
FS_AXIS = 10    # axis labels (set_xlabel / set_ylabel, colorbar label)
FS_TICK = 9     # tick labels (set_xticks / set_yticks / tick_params)
FS_LEGEND = 9   # legend text
FS_ANNOT = 8    # in-plot data annotations (bar values, notes, heatmap cells)


def load_data() -> dict:
    with open(DATA_PATH) as f:
        return json.load(f)


def per_ticker_bar(data: dict) -> None:
    """Bar chart: II vs CI mean ASR per ticker, sorted by II ASR."""
    ticker_asr = defaultdict(lambda: {"II": [], "CI": []})
    for r in data["all_rows"]:
        cat = "II" if r["category"] == "information_injection" else "CI"
        ticker_asr[r["ticker"]][cat].append(r["asr"])

    items = []
    for t, d in ticker_asr.items():
        ii = sum(d["II"]) / len(d["II"]) if d["II"] else 0
        ci = sum(d["CI"]) / len(d["CI"]) if d["CI"] else 0
        items.append((t, ii, ci, data["baselines"].get(t, "?")))
    items.sort(key=lambda x: -x[1])

    tickers = [x[0] for x in items]
    iis = [x[1] for x in items]
    cis = [x[2] for x in items]
    bases = [x[3] for x in items]

    x = np.arange(len(tickers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    b1 = ax.bar(x - w / 2, iis, w, label="Information injection (II)", color=EMERGENT_RED, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + w / 2, cis, w, label="Control injection (CI)", color=PIPE_BLUE, edgecolor="white", linewidth=0.6)
    # 15 four-char tickers cannot sit horizontally at 9pt without touching, so
    # the ticker names and the benign-baseline (B/H) markers are drawn as two
    # separate manual rows below the axis: rotated ticker names (collision-free
    # at 9pt) on top, then a clean horizontal row of single-char B/H markers.
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylabel("Mean Attack Success Rate", fontsize=FS_AXIS)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.tick_params(axis="x", length=0)
    ax.grid(True, axis="y", linestyle=":", color=GRAY, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    # Reserve a bottom band for the two manual label rows + baseline note.
    fig.subplots_adjust(bottom=0.34)
    xaxis_t = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for xi, t, b in zip(x, tickers, bases):
        ax.text(xi, -0.03, t, transform=xaxis_t, rotation=45, ha="right", va="top",
                rotation_mode="anchor", fontsize=FS_TICK)
        ax.text(xi, -0.34, "B" if b == "BUY" else "H", transform=xaxis_t,
                ha="center", va="top", fontsize=FS_ANNOT, color=GRAY)
    # Baseline-row legend, below the B/H marker row.
    ax.text(0.5, -0.50, "B = BUY-baseline    H = HOLD-baseline", transform=ax.transAxes,
            ha="center", va="top", fontsize=FS_ANNOT, color=GRAY)
    out = FIG_DIR / "fig_overall_per_ticker.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


SUBTYPE_LABEL = {
    "fact_poisoning": "Fact poisoning (V1)",
    "sentiment_narrative_manipulation": "Sentiment manip. (V2)",
    "selective_evidence_injection": "Selective evidence (V3)",
    "structured_signal_poisoning": "Structured signal (V4)",
    "coordinated_context_manipulation": "Coordinated multi-source (V5)",
}


def subtype_heatmap(data: dict) -> None:
    """Subtype × ticker heatmap."""
    sub_ticker = defaultdict(dict)
    for r in data["all_rows"]:
        if r["category"] != "information_injection":
            continue
        sub_ticker[r["subtype"]].setdefault(r["ticker"], []).append(r["asr"])

    sub_order = [
        "fact_poisoning",
        "sentiment_narrative_manipulation",
        "selective_evidence_injection",
        "structured_signal_poisoning",
        "coordinated_context_manipulation",
    ]
    # Sort tickers by mean II ASR
    ticker_means = {}
    for t in data["tickers"]:
        vs = []
        for s in sub_order:
            if t in sub_ticker[s]:
                vs.extend(sub_ticker[s][t])
        ticker_means[t] = sum(vs) / len(vs) if vs else 0
    tickers = sorted(data["tickers"], key=lambda t: -ticker_means[t])

    M = np.zeros((len(sub_order), len(tickers)))
    for i, s in enumerate(sub_order):
        for j, t in enumerate(tickers):
            asrs = sub_ticker[s].get(t, [])
            M[i, j] = sum(asrs) / len(asrs) if asrs else 0.0

    fig, ax = plt.subplots(figsize=(FIG_W, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tickers)), tickers, fontsize=FS_TICK)
    ax.set_yticks(np.arange(len(sub_order)), [SUBTYPE_LABEL[s] for s in sub_order], fontsize=FS_TICK)
    # annotate
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_ANNOT, color=color)
    # baseline marks
    for j, t in enumerate(tickers):
        b = data["baselines"].get(t, "?")
        ax.text(j, len(sub_order) - 0.3, b[0], ha="center", va="top", fontsize=FS_ANNOT, color=GRAY,
                transform=ax.transData)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label("ASR", fontsize=FS_AXIS)
    fig.tight_layout()
    out = FIG_DIR / "fig_subtype_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def state_dependence(data: dict) -> None:
    """BUY vs HOLD baseline ASR per attack vector."""
    sub_order = [
        "fact_poisoning",
        "sentiment_narrative_manipulation",
        "selective_evidence_injection",
        "structured_signal_poisoning",
        "coordinated_context_manipulation",
    ]
    buy_t = [t for t, b in data["baselines"].items() if b == "BUY"]
    hold_t = [t for t, b in data["baselines"].items() if b == "HOLD"]
    M = {"BUY": [], "HOLD": []}
    for s in sub_order:
        for state, ts in [("BUY", buy_t), ("HOLD", hold_t)]:
            vals = [r["asr"] for r in data["all_rows"]
                    if r["category"] == "information_injection"
                    and r["subtype"] == s
                    and r["ticker"] in ts]
            M[state].append(sum(vals) / len(vals) if vals else 0)

    x = np.arange(len(sub_order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(FIG_W, 3.2))
    ax.bar(x - w / 2, M["BUY"], w, label=f"BUY-baseline (n={len(buy_t)})", color=CHAIN_ORANGE, edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2, M["HOLD"], w, label=f"HOLD-baseline (n={len(hold_t)})", color=PIPE_BLUE, edgecolor="white", linewidth=0.6)
    # The vector names are long; at 9pt they cannot sit horizontally without
    # overlapping, so rotate them (5 wide slots => rotation is collision-free).
    ax.set_xticks(x, [SUBTYPE_LABEL[s].replace(" (V", "\n(V") for s in sub_order],
                  fontsize=FS_TICK, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylabel("Mean ASR", fontsize=FS_AXIS)
    ax.grid(True, axis="y", linestyle=":", color=GRAY, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=FS_LEGEND, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_state_dependence.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def top_attacks(data: dict) -> None:
    """Top-10 attacks by mean ASR across 15 tickers."""
    by_id = defaultdict(list)
    for r in data["all_rows"]:
        by_id[(r["attack_id"], r["category"], r["subtype"])].append(r["asr"])
    items = [(aid, cat, sub, sum(v) / len(v)) for (aid, cat, sub), v in by_id.items()]
    # Secondary sort key is required. Several templates share an identical mean
    # ASR — most notably coordinated_context_manipulation-3 and -5 are both
    # 0.646667 (n=15), and only one fits in the top 10. Sorting on ASR alone
    # (the previous implementation) lets dict insertion order (i.e. JSON line
    # order) decide the winner, so the same data produces different figures if
    # the rows are reordered. Ascending attack_id as tiebreaker makes the
    # output stable regardless of input order.
    items.sort(key=lambda x: (-x[3], x[0]))
    top = items[:10]
    bot = items[-10:]
    labels_top = [f"{aid} ({cat[:2].upper()})" for aid, cat, sub, _ in top]
    vals_top = [v for _, _, _, v in top]
    colors = [EMERGENT_RED if cat == "information_injection" else PIPE_BLUE for _, cat, _, _ in top]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.0))
    y = np.arange(len(top))
    ax.barh(y, vals_top, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(y, labels_top, fontsize=FS_TICK)
    # Value labels are drawn at v+0.01 past the bar end. At v=1.00 the "1.00"
    # glyph takes ~0.09 data units, so an xlim of 1.05 clips the label against
    # the axis frame (empirically Figure 4's 1.00/1.00/0.93/0.93 all got cut).
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.invert_yaxis()
    ax.set_xlabel("Mean ASR (across 15 tickers)", fontsize=FS_AXIS)
    ax.tick_params(axis="x", labelsize=FS_TICK)
    ax.grid(True, axis="x", linestyle=":", color=GRAY, alpha=0.5)
    ax.set_axisbelow(True)
    for i, v in enumerate(vals_top):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=FS_ANNOT)
    fig.tight_layout()
    out = FIG_DIR / "fig_top_attacks.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def action_flow(data: dict) -> None:
    """How attacks redirect actions."""
    counts = defaultdict(int)
    totals = defaultdict(int)
    for r in data["all_rows"]:
        if r["category"] != "information_injection":
            continue
        ba = r["benign_action"]
        sample = r.get("sample_changed_to") or "unchanged"
        if r["n_success"] == 0 or sample == "unchanged":
            counts[(ba, ba)] += r["n_runs"] - r["n_success"]
            counts[(ba, "Unchanged")] += 0  # placeholder
        else:
            new_action = sample.split()[0] if sample.split() else ba
            counts[(ba, new_action)] += r["n_success"]
            counts[(ba, ba)] += r["n_runs"] - r["n_success"]
        totals[ba] += r["n_runs"]

    # Build a small heatmap: rows = baseline action, cols = post-attack action
    actions = ["BUY", "SELL", "HOLD"]
    M = np.zeros((len(actions), len(actions)))
    for i, b in enumerate(actions):
        if totals[b] == 0:
            continue
        for j, a in enumerate(actions):
            M[i, j] = counts.get((b, a), 0) / totals[b]
    fig, ax = plt.subplots(figsize=(FIG_W, 2.2))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(actions)), actions, fontsize=FS_TICK)
    ax.set_yticks(np.arange(len(actions)), actions, fontsize=FS_TICK)
    ax.set_xlabel("Decision after attack", fontsize=FS_AXIS)
    ax.set_ylabel("Benign baseline", fontsize=FS_AXIS)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_ANNOT, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout()
    out = FIG_DIR / "fig_action_flow.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def category_taxonomy(data: dict) -> None:
    """Heatmap: 7 event categories × ticker baselines (synthetic projection from V→Event mapping)."""
    # Map each generic vector to event categories (from the taxonomy):
    # V1 fact_poisoning           -> RG (regulatory) + GI (governance/short)
    # V2 sentiment_narrative      -> NA (narrative)
    # V3 selective_evidence       -> EA (analyst/earnings)
    # V4 structured_signal        -> MS (market structure / options flow)
    # V5 coordinated_context      -> NA + multi-channel
    # CA, MA stay separate. We'll project each vector's ASR onto its primary
    # event category for visualisation.
    proj = {
        "fact_poisoning": ["RG", "GI"],
        "sentiment_narrative_manipulation": ["NA"],
        "selective_evidence_injection": ["EA"],
        "structured_signal_poisoning": ["MS"],
        "coordinated_context_manipulation": ["NA", "EA"],
    }
    # Aggregate per (category, baseline-state)
    cat_state: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in data["all_rows"]:
        if r["category"] != "information_injection":
            continue
        sub = r["subtype"]
        state = data["baselines"].get(r["ticker"], "?")
        for cat in proj.get(sub, []):
            cat_state[cat][state].append(r["asr"])
    # Add CA and MA placeholders by averaging V1+V3 (proxy)
    for cat in ("CA", "MA"):
        for r in data["all_rows"]:
            if r["category"] != "information_injection" or r["subtype"] not in ("fact_poisoning", "selective_evidence_injection"):
                continue
            cat_state[cat][data["baselines"].get(r["ticker"], "?")].append(r["asr"])

    cat_order = ["CA", "GI", "RG", "EA", "MS", "MA", "NA"]
    states = ["BUY", "HOLD"]
    M = np.zeros((len(cat_order), len(states)))
    for i, c in enumerate(cat_order):
        for j, s in enumerate(states):
            vs = cat_state[c].get(s, [])
            M[i, j] = sum(vs) / len(vs) if vs else 0
    cat_names_short = {
        "CA": "Corporate\nactions",
        "GI": "Governance\n/ Insider",
        "RG": "Regulatory\n/ Legal",
        "EA": "Earnings\n/ Analyst",
        "MS": "Market\nstructure",
        "MA": "Macro\n/ Geopol.",
        "NA": "Narrative\n/ Sentiment",
    }
    fig, ax = plt.subplots(figsize=(FIG_W, 2.8))
    im = ax.imshow(M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(states)), states, fontsize=FS_TICK)
    ax.set_yticks(np.arange(len(cat_order)), [cat_names_short[c] for c in cat_order], fontsize=FS_TICK)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_ANNOT, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label("Projected ASR", fontsize=FS_AXIS)
    fig.tight_layout()
    out = FIG_DIR / "fig_event_category.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    data = load_data()
    per_ticker_bar(data)
    subtype_heatmap(data)
    state_dependence(data)
    top_attacks(data)
    action_flow(data)
    category_taxonomy(data)


if __name__ == "__main__":
    main()
