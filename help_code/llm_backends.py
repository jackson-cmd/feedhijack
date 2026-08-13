#!/usr/bin/env python3
"""Gemini REST and Claude API backends."""
from __future__ import annotations

import itertools
import json
import os
import pathlib
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
    # thinkingBudget=0 disables Gemini-3.x thinking to avoid MAX_TOKENS truncation.
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
# Claude via the Anthropic API
# --------------------------------------------------------------------------- #
_CLAUDE_SYS_DEFAULT = (
    "You are an automated day-trading agent. Base each decision on ALL information in "
    "the user message — the technical indicators AND the market news / social feed. "
    "The feed is live market information; judge whether its items are material and let "
    "material news inform (and if warranted, override) the technical read. Respond with "
    "ONLY the JSON object the user specifies: no tools, no questions, no commentary.")

_anthropic_client = None
_anthropic_lock = threading.Lock()


def _get_anthropic_client():
    global _anthropic_client
    with _anthropic_lock:
        if _anthropic_client is None:
            from dotenv import load_dotenv
            load_dotenv()
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return _anthropic_client


def claude_api_complete(prompt: str, model: str, temperature: float = 0.1,
                        max_tokens: int = 4096, max_retries: int = 4) -> str:
    """Call Anthropic Messages API. temperature accepted for call-site symmetry; not forwarded."""
    import anthropic
    client = _get_anthropic_client()
    sys_prompt = os.getenv("CLAUDE_SYS_PROMPT") or _CLAUDE_SYS_DEFAULT
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=sys_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
            if txt:
                return txt
            last_err = f"empty response (stop_reason={resp.stop_reason})"
        except anthropic.RateLimitError as e:
            last_err = f"rate limit: {e}"
            time.sleep(min(30, 3 * (attempt + 1)))
            continue
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_err = f"server {e.status_code}: {getattr(e, 'message', '')[:160]}"
                time.sleep(3)
                continue
            raise RuntimeError(f"Anthropic {model} failed: {e.status_code} {getattr(e, 'message', '')[:160]}")
        except anthropic.APIConnectionError as e:
            last_err = f"net: {e}"
            time.sleep(3)
            continue
        time.sleep(2)
    raise RuntimeError(f"Anthropic {model} failed after {max_retries} retries: {last_err}")
