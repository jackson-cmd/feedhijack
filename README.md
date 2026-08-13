# FeedHijack

An attack benchmark for **information-injection attacks** against LLM-based trading agents through the news feed.

LLM traders read news headlines before making trade decisions. Injecting crafted but plausible-looking fake news into the feed — fabricated M&A rumors, fake FDA approvals, bogus earnings leaks — is designed to influence the agent's BUY/SELL/HOLD output through the input feed rather than through prompt-level manipulation.

This artifact contains the attack corpus and benchmark runners for measuring attack success rate (ASR). The backtester, full experimental sweep, and figure code are omitted.

## What's in here

- `help_code/` — trading agent, data pipeline, and attack evaluation harness
- `benchmark/` — event-grounded benchmark runners
- `novel_experiments/` — extended experiment runners

The trader is in `help_code/brain.py`: a single-shot LLM prompt that takes a market snapshot and returns strict JSON.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Create `help_code/.env` for API keys (local-only, do not commit):

```
OPENAI_API_KEY=<your key>
GEMINI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
```

Only keys for the models used are required.

Optional keys for alternative news sources:
```
BENZINGA_API_KEY=<your key>
FINVIZ_API_TOKEN=<your key>
```

### Run an attack eval

```bash
cd help_code
python main.py --tickers AAPL --start 20250501 --end 20250505 \
  --attack-eval --attack-repeats 10 --no-backtest
```

Takes a market snapshot, runs each attack template N times, and counts how often the decision flips vs the clean baseline.

Attack templates are organized into three families:

- **Information injection** — plausible fake news designed to manipulate the agent's reasoning: fact poisoning, sentiment/narrative manipulation, selective evidence injection, structured signal poisoning, coordinated context manipulation.
- **Control injection** — direct prompt manipulation baseline: instruction hijacking, role authority abuse, policy bypass, tool manipulation.
- **Event-grounded** — templates modeled after real market-moving event categories (M&A, FDA, earnings, short reports, macro, etc.), in `benchmark/event_attacks.py`.

## Experiments

All scripts auto-load `help_code/.env`.

**Benchmark campaign**:
```bash
cd benchmark
python run_comprehensive.py --experiment all --output results/
```

Includes the main event-grounded sweep and ablations over trading mode, position state, lexical variants, and cross-ticker transfer.

**Cross-model**:
```bash
python run_e6_cross_model.py
python run_e6_extended.py
```

**Extended experiments**:
```bash
cd novel_experiments
python run_novel.py --experiment all --output results/
```

Covers capability scaling, confidence calibration, dose–response over corroborating fake sources, positional bias, an adaptive attack, and benign-news dilution.

**Multi-agent topologies**:
```bash
python run_novel5.py
python run_novel7.py
```

## News sources

Google News RSS is the default (free, no key, date-range supported). Benzinga is used for large sweeps. Finviz public scrape covers ~1 week; the Elite token unlocks full history.

All three write the same JSON schema, one file per trading day.

## Data

Market data and fundamentals come from yfinance, cached locally under `help_code/fundamentals/data/`. News is stored as one JSON per trading day.
