#!/usr/bin/env python3
"""Batch pipeline: benign backtest + attack ASR eval per ticker."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacks.attack_eval import print_asr_table, run_attack_evaluation, save_asr_json
from main import (
    CAPITAL,
    MARGIN,
    TRADES_DIR,
    _first_trading_day_with_data,
    ensure_data,
    run_backtest,
)
from run_output import (
    DEFAULT_README,
    make_run_directory,
    tee_stdio_to,
    write_readme,
    write_run_meta,
)

DEFAULT_LOOKBACK_DAYS = 92  # ~3 months


def load_tickers(csv_path: Path) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        tcol = None
        for name in reader.fieldnames:
            if name and name.strip().lower() == "ticker":
                tcol = name
                break
        if not tcol:
            raise ValueError(f"CSV must have a 'ticker' column: {csv_path}")
        out: list[str] = []
        for row in reader:
            t = (row.get(tcol) or "").strip().upper()
            if t:
                out.append(t)
    return out


def parse_yyyymmdd(s: str) -> date:
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Date must be YYYYMMDD, got: {s}")
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _run_pipeline_body(
    args: argparse.Namespace,
    start: date,
    end: date,
    tickers: list[str],
    run_dir: Path,
    eval_date_global: date | None,
) -> None:
    """Core pipeline logic (stdout may be teed to run_dir/logs/console.txt)."""
    by_ticker = run_dir / "by_ticker"
    summary_dir = run_dir / "summary"

    print(
        f"[Pipeline] tickers={len(tickers)} | {start.isoformat()} → {end.isoformat()} "
        f"| mode={args.mode} | run_dir={run_dir}",
        flush=True,
    )

    ensure_data(tickers, start, end)
    if args.skip_backtest and args.skip_attack:
        print("Nothing to do (--skip-backtest and --skip-attack).", flush=True)
        return

    summary: dict = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "mode": args.mode,
        "attack_repeats": args.attack_repeats,
        "csv": str(Path(args.csv).resolve()),
        "run_directory": str(run_dir.resolve()),
        "tickers": [],
    }
    t0 = time.perf_counter()

    for ticker in tqdm(tickers, desc="Pipeline tickers", unit="sym"):
        tdir = by_ticker / ticker
        tdir.mkdir(parents=True, exist_ok=True)
        entry: dict = {
            "ticker": ticker,
            "directory": str(tdir.resolve()),
            "backtest_metrics": None,
            "attack_asr": None,
            "errors": [],
        }

        if not args.skip_backtest:
            try:
                run_backtest(
                    [ticker],
                    start,
                    end,
                    mode=args.mode,
                    use_tqdm=not args.no_tqdm_backtest,
                )
                m_src = TRADES_DIR / "metrics.json"
                if m_src.exists():
                    m_dst = tdir / "backtest_metrics.json"
                    shutil.copy(m_src, m_dst)
                    entry["backtest_metrics"] = str(m_dst)
                o_src = TRADES_DIR / "orders.jsonl"
                if o_src.exists():
                    shutil.copy(o_src, tdir / "orders.jsonl")
            except Exception as e:
                entry["errors"].append(f"backtest: {e!s}")

        if not args.skip_attack:
            try:
                eval_d = eval_date_global or _first_trading_day_with_data(ticker, start, end)
                if eval_d is None:
                    entry["errors"].append("attack: no market data in range")
                else:
                    alog = tdir / "attack_study_log.jsonl"
                    asr_path = tdir / "attack_asr.json"
                    rows, elapsed = run_attack_evaluation(
                        ticker,
                        eval_d,
                        args.mode,
                        repeats=args.attack_repeats,
                        buying_power=CAPITAL + MARGIN,
                        attack_log_path=alog,
                    )
                    print_asr_table(rows, elapsed, label=ticker)
                    save_asr_json(rows, elapsed, output_path=asr_path, ticker=ticker)
                    entry["attack_asr"] = str(asr_path)
                    entry["attack_log"] = str(alog)
                    entry["attack_eval_date"] = eval_d.isoformat()
                    entry["attack_elapsed_sec"] = elapsed
            except Exception as e:
                entry["errors"].append(f"attack: {e!s}")

        summary["tickers"].append(entry)

    summary["total_wall_seconds"] = time.perf_counter() - t0
    sum_path = summary_dir / "pipeline_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"\n[Pipeline] Done. Summary: {sum_path} | total time {summary['total_wall_seconds']:.1f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pipeline: benign backtest + attack eval for all tickers in CSV."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "tickers.csv",
        help="CSV with a ticker column (default: ./tickers.csv)",
    )
    ap.add_argument(
        "--start",
        default="",
        help="Start date YYYYMMDD (default: end − %d days)" % DEFAULT_LOOKBACK_DAYS,
    )
    ap.add_argument(
        "--end",
        default="",
        help="End date YYYYMMDD (default: today)",
    )
    ap.add_argument(
        "--mode",
        choices=["aggressive", "medium", "conservative"],
        default="medium",
    )
    ap.add_argument("--attack-repeats", type=int, default=10)
    ap.add_argument("--skip-backtest", action="store_true", help="Only run attack eval")
    ap.add_argument("--skip-attack", action="store_true", help="Only run backtest")
    ap.add_argument(
        "--output-base",
        type=Path,
        default=PROJECT_ROOT / "trades_data" / "runs",
        help="Parent directory; a timestamped subfolder is created each run",
    )
    ap.add_argument(
        "--run-label",
        default="pipeline",
        help="Prefix for the run folder name: {run_label}_{YYYYMMDD_HHMMSS}",
    )
    ap.add_argument(
        "--no-tee",
        action="store_true",
        help="Do not duplicate stdout/stderr to logs/console.txt",
    )
    ap.add_argument(
        "--eval-date",
        default="",
        help="YYYYMMDD snapshot for attack eval (all tickers); default: first trading day with data per ticker",
    )
    ap.add_argument(
        "--no-tqdm-backtest",
        action="store_true",
        help="Disable inner tqdm during each backtest",
    )
    args = ap.parse_args()

    if args.end:
        end = parse_yyyymmdd(args.end)
    else:
        end = date.today()
    if args.start:
        start = parse_yyyymmdd(args.start)
    else:
        start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    if start > end:
        print("Error: --start must be <= --end", flush=True)
        sys.exit(1)

    tickers = load_tickers(args.csv)
    if not tickers:
        print("No tickers found in CSV.", flush=True)
        sys.exit(1)

    args.output_base.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_directory(args.output_base, args.run_label)
    write_readme(run_dir, DEFAULT_README)
    meta = {
        "script": "pipeline.py",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "output_base": str(args.output_base.resolve()),
        "run_directory": str(run_dir.resolve()),
        "csv": str(Path(args.csv).resolve()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "mode": args.mode,
        "attack_repeats": args.attack_repeats,
        "skip_backtest": args.skip_backtest,
        "skip_attack": args.skip_attack,
        "eval_date": args.eval_date or None,
        "tickers": tickers,
    }
    write_run_meta(run_dir, meta)

    eval_date_global: date | None = parse_yyyymmdd(args.eval_date) if args.eval_date else None

    if args.no_tee:
        _run_pipeline_body(args, start, end, tickers, run_dir, eval_date_global)
    else:
        with tee_stdio_to(run_dir):
            print(f"[Pipeline] Console log: {run_dir / 'logs' / 'console.txt'}", flush=True)
            _run_pipeline_body(args, start, end, tickers, run_dir, eval_date_global)

    print(f"[Pipeline] All outputs under: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
