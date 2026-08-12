#!/usr/bin/env python3
"""Extra LLM backends for the cross-model ablation.

- Gemini: native REST (generativelanguage API) with round-robin key rotation,
  retry-on-429 with backoff, and key failover.
- Claude: via the local `claude -p` CLI (uses the user's Claude Code auth; no
  Anthropic API key needed). Isolated with a custom --system-prompt and tools
  disabled so it behaves as a clean single-shot completion engine.

Both return the raw text response. brain.decide() builds the prompt and parses
the JSON identically for every backend, so the cross-model comparison stays
apples-to-apples.
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------- #
# Gemini (native REST + key rotation)
# --------------------------------------------------------------------------- #
_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_key_lock = threading.Lock()
_key_cycle = None


def _load_gemini_keys() -> list[str]:
    """Keys from GEMINI_API_KEYS (comma-sep) | GEMINI_KEYS_FILE | GEMINI_API_KEY."""
    env = os.getenv("GEMINI_API_KEYS", "").strip()
    if env:
        return [k.strip() for k in env.split(",") if k.strip()]
    f = os.getenv("GEMINI_KEYS_FILE", "").strip()
    if f and pathlib.Path(f).exists():
        return [l.strip() for l in pathlib.Path(f).read_text().splitlines() if l.strip()]
    one = os.getenv("GEMINI_API_KEY", "").strip()
    return [one] if one else []


def _next_key() -> str:
    global _key_cycle
    with _key_lock:
        if _key_cycle is None:
            keys = _load_gemini_keys()
            if not keys:
                raise RuntimeError("No Gemini keys (set GEMINI_API_KEYS / GEMINI_KEYS_FILE)")
            _key_cycle = itertools.cycle(keys)
        return next(_key_cycle)


def gemini_complete(prompt: str, model: str, temperature: float = 0.1,
                    max_tokens: int = 800, max_retries: int = 8) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    url = _GEMINI_ENDPOINT.format(model=model)
    # thinkingBudget=0 disables Gemini-3.x "thinking": without it, flash-tier
    # models spend the whole output budget thinking and truncate the JSON
    # (finishReason=MAX_TOKENS). Disabling keeps output clean, fast, and
    # consistent with flash-lite (which does not think by default).
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": max_tokens,
                             "thinkingConfig": {"thinkingBudget": 0}},
    }).encode()
    last_err = None
    for attempt in range(max_retries):
        key = _next_key()  # round-robin across the key pool every attempt
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            cands = d.get("candidates") or []
            if not cands:
                last_err = f"no candidates: {json.dumps(d)[:200]}"
                time.sleep(2)
                continue
            parts = cands[0].get("content", {}).get("parts", [])
            txt = "".join(p.get("text", "") for p in parts if "text" in p).strip()
            if not txt:
                last_err = f"empty text (finishReason={cands[0].get('finishReason')})"
                time.sleep(2)
                continue
            return txt
        except urllib.error.HTTPError as e:
            try:
                emsg = json.loads(e.read().decode()).get("error", {}).get("message", "")[:160]
            except Exception:
                emsg = ""
            last_err = f"HTTP {e.code}: {emsg}"
            if e.code == 429:               # rate/quota -> rotate key + backoff
                time.sleep(min(30, 3 * (attempt + 1)))
                continue
            if e.code in (500, 503):        # transient server
                time.sleep(3)
                continue
            raise RuntimeError(f"Gemini {model} failed: {last_err}")  # 400/403/404
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = f"net: {e}"
            time.sleep(3)
            continue
    raise RuntimeError(f"Gemini {model} failed after {max_retries} retries: {last_err}")


# --------------------------------------------------------------------------- #
# Claude via the `claude -p` CLI
# --------------------------------------------------------------------------- #
# Realistic, feed-aware trading-agent framing. The prior "decision engine, output
# JSON" wording did NOT mention the news feed and biased Claude toward ignoring it
# (an asymmetry: Gemini/OpenAI get no system prompt at all). This tells the agent to
# weigh the news/social feed alongside the technicals, like a real news-consuming
# agent. Overridable via env CLAUDE_SYS_PROMPT for prompt-sensitivity experiments.
_CLAUDE_SYS_DEFAULT = (
    "You are an automated day-trading agent. Base each decision on ALL information in "
    "the user message — the technical indicators AND the market news / social feed. "
    "The feed is live market information; judge whether its items are material and let "
    "material news inform (and if warranted, override) the technical read. Respond with "
    "ONLY the JSON object the user specifies: no tools, no questions, no commentary.")
# Run the `claude` CLI in an empty, project-free cwd so it can't pick up the caller's
# CLAUDE.md, workspace state, or session files. Override via CLAUDE_CLI_CWD.
_CLAUDE_NEUTRAL_CWD = os.getenv("CLAUDE_CLI_CWD", tempfile.gettempdir())


def claude_cli_complete(prompt: str, model: str, temperature: float = 0.1,
                        timeout: int = 150, max_retries: int = 2) -> str:
    """NOTE: `claude -p` exposes no temperature flag; `temperature` is accepted for
    call-site symmetry but NOT applied (documented caveat in the ablation writeup)."""
    cwd = _CLAUDE_NEUTRAL_CWD if pathlib.Path(_CLAUDE_NEUTRAL_CWD).exists() else os.getcwd()
    sys_prompt = os.getenv("CLAUDE_SYS_PROMPT") or _CLAUDE_SYS_DEFAULT
    cmd = ["claude", "-p", prompt, "--model", model,
           "--system-prompt", sys_prompt,
           "--disallowedTools", "Bash Edit Read Write WebSearch WebFetch Task",
           "--no-session-persistence",  # clean concurrent runs, no session-file writes
           "--output-format", "text"]
    # CLAUDE_QUIET=1 makes the spawned session's Stop/Notification hooks no-op,
    # so these batch subprocess calls don't spam desktop notifications.
    env = {**os.environ, "CLAUDE_QUIET": "1"}
    last_err = None
    for _ in range(max_retries):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
            out = (res.stdout or "").strip()
            if out:
                return out
            last_err = f"empty stdout; stderr={(res.stderr or '')[:150]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        time.sleep(2)
    raise RuntimeError(f"claude -p {model} failed after {max_retries} tries: {last_err}")
