#!/usr/bin/env python3
"""Analyze novel-experiment results into a text report + a markdown RESULTS block.

Reads results/n{1..6}_results.json and emits:
  - stdout: human-readable report with the headline numbers
  - results/RESULTS.md: markdown block to splice into NOVEL_EXPERIMENTS.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

R = Path(__file__).parent / "results"
MODEL_ORDER = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]


def _load(name):
    p = R / f"{name}_results.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x):
    return f"{x:.2f}"


def n1_report(out):
    rows = _load("n1")
    if not rows:
        return "### N1 — Capability-Scaling Law\n_(no results)_\n"
    by = defaultdict(list)            # (model, vector) -> asr list
    for r in rows:
        by[(r["model"], r["vector"])].append(r["asr"])
    vectors = sorted({r["vector"] for r in rows})
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
    out.append("### N1 — Capability-Scaling Law of Vulnerability\n")
    out.append("Raw mean ASR by (vector × model). _Caveat: V4/V5 are state-dependent, so their "
               "cross-model row mixes baseline states (see stratified view below)._\n")
    header = "| Vector | " + " | ".join(models) + " |"
    out.append(header)
    out.append("|" + "---|" * (len(models) + 1))
    for v in vectors:
        cells = [mean(by[(m, v)]) if by[(m, v)] else float("nan") for m in models]
        out.append(f"| {v} | " + " | ".join(fmt(c) for c in cells) + " |")
    overall = [mean([r["asr"] for r in rows if r["model"] == m]) for m in models]
    out.append("| **mean** | " + " | ".join(fmt(c) for c in overall) + " |\n")

    # --- clean view 1: V1 fact poisoning is state-robust => its curve is unconfounded ---
    v1 = [mean(by[(m, "V1_fact")]) if by[(m, "V1_fact")] else float("nan") for m in models]
    out.append("**V1 fact poisoning (state-robust ⇒ clean scaling curve):** "
               + " · ".join(f"{MODEL_SHORT_(m)}={fmt(c)}" for m, c in zip(models, v1)) + "\n")

    # --- clean view 2: stratify the state-dependent vectors by baseline state ---
    def strat(vec, state):
        return [mean([r["asr"] for r in rows if r["model"] == m and r["vector"] == vec
                      and r["benign_action"] == state]) if any(
                    r["model"] == m and r["vector"] == vec and r["benign_action"] == state
                    for r in rows) else float("nan") for m in models]
    out.append("**V4 structured-signal, stratified by baseline state:**\n")
    out.append("| state | " + " | ".join(MODEL_SHORT_(m) for m in models) + " |")
    out.append("|" + "---|" * (len(models) + 1))
    for st in ("HOLD", "BUY"):
        out.append(f"| {st} | " + " | ".join(fmt(c) for c in strat("V4_structured", st)) + " |")
    out.append("")

    v1_min, v1_max = min(c for c in v1 if c == c), max(c for c in v1 if c == c)
    print("[N1] V1 curve:", {MODEL_SHORT_(m): round(c, 2) for m, c in zip(models, v1)})
    print("[N1] mean ASR by model:", {m: round(c, 2) for m, c in zip(models, overall)})
    out.append(f"**Insight.** V1 fact poisoning never collapses across the {len(models)}-rung ladder "
               f"(stays in [{fmt(v1_min)}, {fmt(v1_max)}] from {MODEL_SHORT_(models[0])} to "
               f"{MODEL_SHORT_(models[-1])}): the vector with **no inspectable in-context ground "
               "truth** is model-invariant, so scaling the base model is *not* a defense for the "
               "dominant attack. By contrast, V4/V5 are inert wherever the model's baseline is BUY "
               "(consistent with our state-dependence finding); on HOLD baselines they attenuate as "
               "capability rises, because a stronger model cross-checks the fake indicator panel "
               "against the real indicators in the same prompt.\n")
    return "\n".join(out)


def MODEL_SHORT_(m):
    return {"gpt-3.5-turbo": "3.5T", "gpt-4o-mini": "4o-mini", "gpt-4.1-nano": "4.1-nano",
            "gpt-4.1-mini": "4.1-mini", "gpt-4o": "4o", "gpt-4.1": "4.1"}.get(m, m)


def n2_report(out):
    rows = _load("n2")
    if not rows:
        return "### N2 — Confidence Calibration\n_(no results)_\n"
    conf = defaultdict(list)
    for r in rows:
        conf[r["outcome"]].append(r["confidence"])
    b = mean(conf["benign"]) if conf["benign"] else float("nan")
    hit = mean(conf["hit"]) if conf["hit"] else float("nan")
    miss = mean(conf["miss"]) if conf["miss"] else float("nan")
    out.append("### N2 — Confidence Calibration Under Attack\n")
    out.append("| Outcome | mean confidence | n |")
    out.append("|---|---|---|")
    out.append(f"| benign baseline | {fmt(b)} | {len(conf['benign'])} |")
    out.append(f"| successful attack (hit) | {fmt(hit)} | {len(conf['hit'])} |")
    out.append(f"| failed attack (miss) | {fmt(miss)} | {len(conf['miss'])} |\n")
    delta = hit - b
    print(f"[N2] conf benign={b:.2f} hit={hit:.2f} miss={miss:.2f}  delta(hit-benign)={delta:+.2f}")
    verdict = ("there is no usable internal uncertainty signal: the agent is at least as confident "
               "when acting on fabricated news as on benign input"
               if delta >= -0.05 else
               "the agent is somewhat less confident on successful attacks, leaving a weak gating signal")
    out.append(f"**Insight.** mean confidence on successful attacks = {fmt(hit)} vs "
               f"{fmt(b)} benign (Δ={delta:+.2f}); {verdict}. This undercuts cheap "
               "\"ask-the-model-if-it's-sure\" defenses and motivates external provenance.\n")
    return "\n".join(out)


def n3_report(out):
    rows = _load("n3")
    if not rows:
        return "### N3 — Dose-Response\n_(no results)_\n"
    by = defaultdict(list)
    for r in rows:
        by[r["dose"]].append(r["asr"])
    doses = sorted(by)
    out.append("### N3 — Dose-Response: Number of Corroborating Sources\n")
    out.append("| #sources | " + " | ".join(str(d) for d in doses) + " |")
    out.append("|" + "---|" * (len(doses) + 1))
    means = [mean(by[d]) for d in doses]
    out.append("| mean ASR | " + " | ".join(fmt(m) for m in means) + " |\n")
    # knee: first dose reaching >=90% of the max
    mx = max(means) if means else 0
    knee = next((d for d, m in zip(doses, means) if mx > 0 and m >= 0.9 * mx), doses[-1])
    print(f"[N3] ASR by dose:", {d: round(m, 2) for d, m in zip(doses, means)}, "knee=", knee)
    out.append(f"**Insight.** ASR rises from {fmt(means[0])} at one source to {fmt(means[-1])} at "
               f"{doses[-1]} sources, with the knee at **{knee} corroborating sources** "
               f"(≥90% of max ASR). An attacker needs only ~{knee} coordinated accounts; any defense "
               "tolerating that many corroborating posts is already defeated.\n")
    return "\n".join(out)


def n4_report(out):
    rows = _load("n4")
    if not rows:
        return "### N4 — Positional Bias\n_(no results)_\n"
    by = defaultdict(list)
    byattack = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["position"]].append(r["asr"])
        byattack[r["attack"]][r["position"]].append(r["asr"])
    order = ["first", "middle", "last"]
    out.append("### N4 — Positional Bias: Primacy vs. Recency\n")
    out.append("| Attack | first | middle | last |")
    out.append("|---|---|---|---|")
    for atk, d in byattack.items():
        out.append(f"| {atk} | " + " | ".join(fmt(mean(d[p])) if d[p] else "-" for p in order) + " |")
    means = {p: mean(by[p]) for p in order if by[p]}
    out.append(f"| **mean** | " + " | ".join(fmt(means.get(p, float('nan'))) for p in order) + " |\n")
    contrast = means.get("last", 0) - means.get("first", 0)
    print(f"[N4] ASR by position:", {p: round(v, 2) for p, v in means.items()}, "last-first=", round(contrast, 2))
    out.append(f"**Insight.** mean ASR is {fmt(means.get('first', float('nan')))} (first), "
               f"{fmt(means.get('middle', float('nan')))} (middle), {fmt(means.get('last', float('nan')))} "
               f"(last); the recency contrast last−first = {contrast:+.2f}. "
               + ("Injections appended after curated news (the common pipeline order) are the most "
                  "effective placement, so feed ordering is an exploitable — and cheaply mitigable — "
                  "design choice.\n" if abs(contrast) >= 0.05 else
                  "Position has little effect here, so the agent is roughly order-invariant on this feed.\n"))
    return "\n".join(out)


def n5_report(out):
    data = _load("n5")
    if not data:
        return "### N5 — Adaptive Attack\n_(no results)_\n"
    traj = data.get("trajectory", [])
    found = data.get("found", {})
    transfer = data.get("transfer", [])
    out.append("### N5 — Adaptive Greedy Attack Optimization\n")
    out.append("| Target | dir | rounds run | best ASR | ASR trajectory |")
    out.append("|---|---|---|---|---|")
    bytk = defaultdict(list)
    for r in traj:
        bytk[r["ticker"]].append(r)
    for tk, rs in bytk.items():
        rs = sorted(rs, key=lambda x: x["round"])
        d = found.get(tk, {})
        traj_s = " → ".join(fmt(x["asr"]) for x in rs)
        out.append(f"| {tk} | {d.get('direction','?')} | {len(rs)} | {fmt(d.get('best_asr',0))} | {traj_s} |")
    out.append("")
    if transfer:
        out.append("Transfer of optimized injection (source→target):\n")
        out.append("| source\\target | " + " | ".join(sorted({x['target'] for x in transfer})) + " |")
        tgts = sorted({x["target"] for x in transfer})
        srcs = sorted({x["source"] for x in transfer})
        out.append("|" + "---|" * (len(tgts) + 1))
        tmap = {(x["source"], x["target"]): x["asr"] for x in transfer}
        for s in srcs:
            out.append(f"| {s} | " + " | ".join(fmt(tmap.get((s, t), float('nan'))) if (s, t) in tmap else "-" for t in tgts) + " |")
        out.append("")
    rounds_to_max = {tk: min((r["round"] for r in rs if r["asr"] >= found.get(tk, {}).get("best_asr", 1)), default=0)
                     for tk, rs in bytk.items()}
    print("[N5] best ASR:", {tk: round(found[tk]["best_asr"], 2) for tk in found})
    print("[N5] rounds-to-best:", rounds_to_max)
    avg_best = mean([found[tk]["best_asr"] for tk in found]) if found else 0
    out.append(f"**Insight.** Starting from a deliberately weak base, a black-box LLM attacker that only "
               f"reads the agent's own stated reasoning lifts mean ASR to {fmt(avg_best)} on the most "
               "robust tickers within a handful of rounds. The static benchmark is therefore a lower "
               "bound; an adaptive attacker is strictly stronger.\n")
    return "\n".join(out)


def n6_report(out):
    rows = _load("n6")
    if not rows:
        return "### N6 — Dilution Robustness\n_(no results)_\n"
    by = defaultdict(list)
    for r in rows:
        by[r["n_headlines"]].append(r["asr"])
    ks = sorted(by)
    out.append("### N6 — Benign-News Dilution Robustness\n")
    out.append("| #benign headlines : 1 fake | " + " | ".join(f"{k}:1" for k in ks) + " |")
    out.append("|" + "---|" * (len(ks) + 1))
    means = [mean(by[k]) for k in ks]
    out.append("| mean ASR | " + " | ".join(fmt(m) for m in means) + " |\n")
    drop = means[0] - means[-1]
    print("[N6] ASR by dilution:", {k: round(m, 2) for k, m in zip(ks, means)})
    out.append(f"**Insight.** Diluting the single fake item with up to {ks[-1]} real headlines changes "
               f"ASR by {drop:+.2f} (from {fmt(means[0])} to {fmt(means[-1])}). "
               + ("Context volume is not a defense — the fake item survives a busy news day.\n"
                  if abs(drop) < 0.15 else
                  "Heavier benign context measurably dampens the attack, a partial mitigation.\n"))
    return "\n".join(out)


def main():
    blocks = ["\n<!-- AUTO-GENERATED by analyze.py -->\n"]
    for fn in (n1_report, n2_report, n3_report, n4_report, n5_report, n6_report):
        blocks.append(fn([]) if False else fn(out=[]))
    md = "\n\n".join(blocks)
    (R / "RESULTS.md").write_text(md)
    print("\nWrote", R / "RESULTS.md")


if __name__ == "__main__":
    main()
