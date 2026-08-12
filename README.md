# FeedHijack

Code for our paper on **information injection attacks** against LLM-based trading agents through the news feed.

The idea is simple: LLM traders read news headlines before making trade decisions. We show that by injecting crafted but plausible-looking fake news into the feed — fake M&A rumors, fabricated FDA approvals, bogus earnings leaks — you can reliably flip the agent's BUY/SELL/HOLD output. No jailbreaking, no prompt hacking, just information that *looks real* but isn't.

We test this across GPT-4o, Gemini, Claude, and multiple agent architectures (single-shot, debate, researcher-writer). The attack success rates are uncomfortably high.

We report **ASR** (how often the injected news flips the decision) and end-to-end P&L impact on a $10k+$10k-margin backtest benchmarked against SPY.

## What's in here

- `help_code/` — trading agent, backtester, data pipeline, and the attack evaluation harness
- `benchmark/` — experiments E1–E6 from the paper (event-grounded information injection, ablations, cross-model)
- `novel_experiments/` — extended experiments N1–N7 (scaling laws, dose-response, topology comparison, etc.)

The trader is in `help_code/brain.py` — a single-shot LLM prompt that takes a market snapshot and returns strict JSON. Nothing fancy, which is kind of the point.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

You'll need to set up your own API keys. Create a file `help_code/.env` (this file is local-only, don't commit it):

```
OPENAI_API_KEY=<your key>
GEMINI_API_KEY=<your key>
```

You only need the keys for the models you actually want to run. If you're just testing with OpenAI, the Gemini key isn't needed and vice versa.

Optional keys (only if you want alternative news sources):
```
BENZINGA_API_KEY=<your key>    # for Benzinga news feed
FINVIZ_API_TOKEN=<your key>    # for Finviz Elite (full history)
```

For Claude experiments, the code calls the `claude -p` CLI directly — install Claude Code and log in, no API key needed.

### Run a backtest

```bash
cd help_code
python main.py --tickers AAPL --start 20250501 --end 20250601 --mode medium
```

First run auto-downloads market data (yfinance), fundamentals, and news. Results go to `help_code/trades_data/` — `metrics.json` has returns/Sharpe/drawdown, `orders.jsonl` is the trade log, and `decisions/decision_log.jsonl` has the full LLM reasoning.

Three modes: `aggressive` (trade every day), `medium` (trade when signals align), `conservative` (high-conviction only).

### Run an attack eval

```bash
cd help_code
python main.py --tickers AAPL --start 20250501 --end 20250505 \
  --attack-eval --attack-repeats 10 --no-backtest
```

Takes a market snapshot, runs each attack template N times, and counts how often the decision flips vs the clean baseline.

Attack templates are organized into three families:

- **Information injection** (the main contribution) — plausible fake news designed to manipulate the agent's reasoning: fact poisoning, sentiment/narrative manipulation, selective evidence injection, structured signal poisoning, coordinated context manipulation. These are the ones that work scarily well.
- **Control injection** (baseline comparison) — direct prompt manipulation: instruction hijacking, role authority abuse, policy bypass, tool manipulation. Included to show that information injection is comparably effective without needing any prompt engineering tricks.
- **Event-grounded** — 37 templates modeled after real market-moving events across 7 categories (M&A, FDA, earnings, short reports, macro, etc.), in `benchmark/event_attacks.py`.

## Experiments

All scripts auto-load `help_code/.env`.

**E1–E5** (main paper):
```bash
cd benchmark
python run_comprehensive.py --experiment all --output results/
```

E1 is the main benchmark — 15 tickers x 37 event templates x 5 reps. E2–E4 ablate trading mode, position state, and lexical variants. E5 tests cross-ticker transferability.

**E6** (cross-model):
```bash
python run_e6_cross_model.py
python run_e6_extended.py
```

**N1–N7** (novel experiments):
```bash
cd novel_experiments
python run_novel.py --experiment all --output results/
```

N1 is the scaling law — turns out smarter models aren't much better at resisting information injection. N3 measures how many corroborating fake sources it takes. N4 checks positional bias in the news feed. The topology experiments test whether multi-agent setups (debate, researcher-writer) are more robust than single-shot. (Spoiler: they propagate the injected narrative instead of catching it.)

### Figures

```bash
cd benchmark && python make_figures.py
cd ../novel_experiments && python make_novel_figures.py
```

Outputs land in `figures_pdf/`.

## News sources

Google News RSS is the default — free, no key, supports date ranges. Benzinga is better for large sweeps. Finviz public scrape only covers ~1 week; the Elite token unlocks full history.

All three write the same JSON schema, one file per trading day.

## Cost

The full E1–E5 suite on gpt-4o-mini runs about 3500 LLM calls in ~75 min. Cross-model and novel experiments scale with the number of models.

## Data

Market data and fundamentals come from yfinance, cached locally. The 15 tickers from the paper are already included under `help_code/fundamentals/data/`. News is stored as one JSON per trading day.

## License

Research artifact — see the paper for citation.
