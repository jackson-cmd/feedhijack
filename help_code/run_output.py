"""Timestamped run directories and stdout/stderr tee to log file."""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


class TeeIO:
    """Write to multiple streams (e.g. terminal + file)."""

    def __init__(self, *streams: Any):
        self.streams = list(streams)

    def write(self, s: str) -> int:
        n = len(s)
        for st in self.streams:
            st.write(s)
            st.flush()
        return n

    def flush(self) -> None:
        for st in self.streams:
            st.flush()

    def isatty(self) -> bool:
        return self.streams[0].isatty() if self.streams else False

    def fileno(self) -> int:
        return self.streams[0].fileno()


def make_run_directory(base: Path, label: str = "run") -> Path:
    """Create base/{safe_label}_{YYYYMMDD_HHMMSS}/ with logs/, by_ticker/, summary/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label.strip()) or "run"
    path = base.expanduser().resolve() / f"{safe}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)
    (path / "by_ticker").mkdir(exist_ok=True)
    (path / "summary").mkdir(exist_ok=True)
    return path


def write_run_meta(run_dir: Path, meta: dict[str, Any]) -> Path:
    path = run_dir / "summary" / "run_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def write_readme(run_dir: Path, text: str) -> Path:
    path = run_dir / "README.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


DEFAULT_README = """Run output layout
=================
summary/run_meta.json   — CLI args, time range, mode, paths
summary/pipeline_summary.json — per-ticker results (pipeline only)
logs/console.txt        — full copy of stdout + stderr from this run
by_ticker/<TICKER>/     — metrics, orders, attack ASR, attack logs per symbol
"""


@contextmanager
def tee_stdio_to(run_dir: Path, log_name: str = "console.txt") -> Iterator[Path]:
    """Tee stdout and stderr to run_dir/logs/<log_name>; restore on exit."""
    log_path = run_dir / "logs" / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = TeeIO(old_out, log_f)
        sys.stderr = TeeIO(old_err, log_f)
        yield log_path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        log_f.close()
