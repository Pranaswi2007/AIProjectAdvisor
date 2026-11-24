"""
LLM Client — GroqCloud wrapper
================================
Single entry point for all agent LLM calls.

Configuration (via .env or environment variables)
-------------------------------------------------
    GROQ_API_KEY  — required
    GROQ_MODEL    — default "openai/gpt-oss-20b"
    GROQ_TIMEOUT  — request timeout in seconds, default 45
"""

import hashlib
import json
import logging
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env — load once at module import using an absolute path
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH     = _PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(dotenv_path=_ENV_PATH, override=False)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config — read after .env is loaded
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    return os.environ.get("GROQ_API_KEY", "")

def _get_gemini_api_key() -> str:
    return ""  # Gemini disabled — using Groq

def _is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()

DEFAULT_MODEL   = "openai/gpt-oss-120b"  # separate daily quota from gpt-oss-20b — fresh 200K TPD
DEFAULT_TIMEOUT = int(os.environ.get("GROQ_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Gemini OpenAI-compatible REST client (stdlib urllib — no extra package)
# ---------------------------------------------------------------------------

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class _GeminiMessage:
    def __init__(self, content: str):
        self.content = content


class _GeminiChoice:
    def __init__(self, content: str, finish_reason: str):
        self.message      = _GeminiMessage(content)
        self.finish_reason = finish_reason


class _GeminiResponse:
    def __init__(self, content: str, finish_reason: str):
        self.choices = [_GeminiChoice(content, finish_reason)]


def _call_gemini(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout: int,
) -> _GeminiResponse:
    """Call Gemini via its OpenAI-compatible REST endpoint using urllib."""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise LLMError(
            "GEMINI_API_KEY is not set. Add it to .env or the environment."
        )

    payload: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        # Disable thinking tokens → ~4s response time
        "reasoning_effort": "none",
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        _GEMINI_BASE_URL,
        data=body,
        method="POST",
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        status   = exc.code
        # Map HTTP status codes to the same permanent/retryable semantics as Groq
        if status in (400, 401, 403, 404, 422):
            raise LLMError(f"Gemini permanent error {status}: {raw_body}") from exc
        if status == 429:
            raise LLMError(
                f"Gemini rate limit (429) — wait a moment and retry. Detail: {raw_body}"
            ) from exc
        if status == 413:
            err = Exception(f"Gemini TPM limit 413: {raw_body}")
            err.status_code = 413  # type: ignore[attr-defined]
            raise err from exc
        raise LLMError(f"Gemini HTTP error {status}: {raw_body}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"Gemini connection error: {exc}") from exc

    choice        = data["choices"][0]
    content       = (choice.get("message") or {}).get("content") or ""
    finish_reason = choice.get("finish_reason", "unknown")
    return _GeminiResponse(content, finish_reason)

# ---------------------------------------------------------------------------
# Safe diagnostic
# ---------------------------------------------------------------------------

def check_api_key() -> dict:
    """Return key status without exposing the value."""
    # Prefer Gemini if its key is set
    gemini_key = _get_gemini_api_key()
    if gemini_key:
        return {"configured": True, "source": "GEMINI_API_KEY", "hint": ""}
    key = _get_api_key()
    if key:
        try:
            from dotenv import dotenv_values
            in_dotenv = "GROQ_API_KEY" in dotenv_values(dotenv_path=_ENV_PATH)
            source = str(_ENV_PATH) if in_dotenv else "environment variable"
        except Exception:
            source = "environment"
        return {"configured": True, "source": source, "hint": ""}
    return {
        "configured": False,
        "source":     None,
        "hint":       f"Add GEMINI_API_KEY=<your_key> to {_ENV_PATH}",
    }

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMError(RuntimeError):
    """Raised when the LLM call fails and cannot recover."""


class _TruncatedStr(str):
    """str subclass that marks content returned with finish_reason=length.

    Behaves exactly like a plain str everywhere — JSON parsing, slicing,
    logging all work normally.  call_llm_json checks isinstance(raw, _TruncatedStr)
    to emit a precise error message if repair also fails.
    """

# ---------------------------------------------------------------------------
# Permanent-error detection (no retry on 4xx)
# ---------------------------------------------------------------------------

_PERMANENT_ERRORS: tuple = ()
_RETRYABLE_ERRORS: tuple = ()

try:
    import groq as _groq_pkg
    _PERMANENT_ERRORS = (
        _groq_pkg.BadRequestError,
        _groq_pkg.AuthenticationError,
        _groq_pkg.PermissionDeniedError,
        _groq_pkg.NotFoundError,
        _groq_pkg.UnprocessableEntityError,
    )
    _RETRYABLE_ERRORS = (
        _groq_pkg.RateLimitError,
        _groq_pkg.InternalServerError,
        _groq_pkg.APIConnectionError,
        _groq_pkg.APITimeoutError,
    )
except (ImportError, AttributeError):
    pass


def _is_permanent_error(exc: Exception) -> bool:
    """Return True for errors that should NOT be retried.

    413 used to be treated as permanent, but on Groq it means the TPM window
    is exhausted — wait 60s and the window resets. So 413 is NOT permanent here;
    it is handled as a special retryable in call_llm instead.
    """
    if _PERMANENT_ERRORS and isinstance(exc, _PERMANENT_ERRORS):
        return True
    status = getattr(exc, "status_code", None)
    if status is not None and status in (400, 401, 403, 404, 422):
        return True
    return False

# ---------------------------------------------------------------------------
# Context cache — prevents re-embedding the same queries across agents
# ---------------------------------------------------------------------------

_context_cache: dict[tuple, tuple[float, str]] = {}
_CACHE_TTL = 300  # 5 minutes — covers a single analysis run


def _context_cache_key(project_id: str, queries: list[str], k: int,
                       max_chars: int | None = None) -> tuple:
    h = hashlib.md5("|".join(queries).encode()).hexdigest()
    return (project_id, h, k, max_chars)  # max_chars IS part of the key

# ---------------------------------------------------------------------------
# Core LLM call
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    retries: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    json_mode: bool = False,
) -> str:
    """Call Groq and return the reply text.

    Uses `max_tokens` (the parameter groq==0.9.0 accepts).
    For qwen/qwen3.8-27b: reasoning_effort="none" disables thinking mode so
    all tokens go directly to visible output (instruct mode, ~450 t/s).

    - max_tokens  default 8192 — full output budget.
    - retries     default 3    — rate-limit (429) waits for the API's requested
                                 time (parsed from "try again in Xs" message),
                                 then retries; permanent errors (4xx) never retry.
    - timeout     default 30s  — sufficient for qwen3 instruct mode.
    - json_mode=True requests response_format=json_object.
    """
    _use_gemini = _is_gemini_model(model)
    _messages   = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    if _use_gemini:
        # Gemini path — uses urllib, no groq package needed.
        # Key check is inside _call_gemini; no client object to build here.
        client         = None  # type: ignore[assignment]
        create_kwargs: dict[str, Any] = {}
    else:
        api_key = _get_api_key()
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it to .env or the environment."
            )
        if Groq is None:
            raise LLMError(
                "groq package not installed. "
                "Run: venv\\Scripts\\python.exe -m pip install groq"
            )
        client = Groq(api_key=api_key, timeout=timeout)
        create_kwargs = dict(
            model=model,
            messages=_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # reasoning_effort must go in extra_body — the Groq SDK passes it
        # as a top-level JSON field rather than a named Python parameter.
        if "qwen" in model.lower():
            create_kwargs["extra_body"] = {"reasoning_effort": "none"}
        # Low reasoning for gpt-oss models — fast output, finishes in <10s.
        if "gpt-oss" in model.lower():
            create_kwargs["extra_body"] = {"reasoning_effort": "low"}
        if json_mode:
            create_kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception = RuntimeError("Unknown error")
    _json_mode_active = json_mode

    for attempt in range(1, retries + 1):
        try:
            if _use_gemini:
                resp = _call_gemini(
                    messages=_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=_json_mode_active,
                    timeout=timeout,
                )
            else:
                resp = client.chat.completions.create(**create_kwargs)
            choice        = resp.choices[0]
            content       = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "unknown")

            # Safe diagnostic log — never logs keys or secrets
            logger.info(
                "LLM response: model=%s attempt=%d finish_reason=%s "
                "content_len=%d empty=%s",
                model, attempt, finish_reason, len(content), not bool(content),
            )

            if not content:
                if finish_reason == "length":
                    raise LLMError(
                        "LLM output truncated (finish_reason=length, empty content). "
                        "Increase max_completion_tokens."
                    )
                logger.warning(
                    "LLM returned empty content (finish_reason=%s, model=%s, attempt=%d)",
                    finish_reason, model, attempt,
                )
                return content

            # Non-empty but truncated — return partial content so call_llm_json
            # can attempt JSON repair before giving up.  Attach a sentinel attr
            # so the caller knows the output may be incomplete.
            if finish_reason == "length":
                logger.warning(
                    "LLM output truncated (finish_reason=length, "
                    "content_len=%d, model=%s, attempt=%d). "
                    "Passing partial content to JSON repair.",
                    len(content), model, attempt,
                )
                # Tag the string so call_llm_json can raise a precise error
                # if repair also fails, rather than a generic parse error.
                content = _TruncatedStr(content)
            return content
        except LLMError:
            raise
        except Exception as exc:
            last_exc = exc

            # If json_mode caused a 400, retry once without it
            if _json_mode_active and attempt == 1 and _is_permanent_error(exc):
                logger.warning("JSON mode caused permanent error, retrying without it: %s", exc)
                create_kwargs.pop("response_format", None)
                _json_mode_active = False
                continue

            # 413 — TPM window exhausted.  Wait 60s for the window to reset,
            # then retry with the same request (the window resets per minute).
            status = getattr(exc, "status_code", None)
            if status == 413:
                wait = 62  # slightly over 60s to ensure the window has fully reset
                logger.warning(
                    "413 TPM limit hit (attempt %d/%d) — waiting %.0fs for rate window to reset",
                    attempt, retries, wait,
                )
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise LLMError(
                    f"TPM limit persisted after {retries} attempt(s). "
                    "Reduce max_chars or max_tokens further, or upgrade your Groq plan."
                ) from exc

            # Hard-fail on all other permanent errors — do not retry 400, 401, 404 etc.
            if _is_permanent_error(exc):
                logger.error("LLM permanent error (no retry): %s", exc)
                raise LLMError(f"LLM call failed (permanent): {exc}") from exc

            # 429 RateLimitError: respect Retry-After header if present, else wait 10s
            is_rate_limit = _RETRYABLE_ERRORS and isinstance(exc, _RETRYABLE_ERRORS[:1])
            if not is_rate_limit:
                try:
                    is_rate_limit = isinstance(exc, _groq_pkg.RateLimitError)
                except Exception:
                    pass
            if is_rate_limit:
                # Try to extract wait time from the Retry-After header first,
                # then fall back to parsing the error message body which
                # contains "Please try again in 24.59s".
                wait = 10  # default fallback
                try:
                    headers = getattr(getattr(exc, "response", None), "headers", None)
                    if headers and "retry-after" in headers:
                        wait = float(headers["retry-after"])
                    else:
                        # Parse from error message: "Please try again in 24.5925s"
                        msg = str(exc)
                        m = re.search(r"try again in ([\d.]+)s", msg, re.IGNORECASE)
                        if m:
                            wait = float(m.group(1))
                except (ValueError, TypeError, AttributeError):
                    pass
                wait = min(wait, 45)  # cap at 45s — gives the rate window time to reset
                logger.warning(
                    "Rate limit hit (attempt %d/%d) — waiting %.1fs before retry",
                    attempt, retries, wait,
                )
                if attempt < retries:
                    time.sleep(wait)
                    continue
                raise LLMError(f"Rate limit persisted after {retries} attempt(s): {exc}") from exc

            # Other transient — backoff and retry (at most once)
            if attempt < retries:
                wait = 2   # fixed 2s wait before the one allowed retry
                logger.warning(
                    "LLM transient error attempt %d/%d: %s — retrying in %ds",
                    attempt, retries, exc, wait,
                )
                time.sleep(wait)

    raise LLMError(f"LLM call failed after {retries} attempts: {last_exc}") from last_exc

# ---------------------------------------------------------------------------
# JSON call — with multi-strategy extraction and one repair attempt
# ---------------------------------------------------------------------------

def call_llm_json(
    system: str,
    user: str,
    fallback: dict | None = None,
    json_mode: bool = False,   # False: reasoning model (openai/gpt-oss-20b) returns 400 with json_object mode
    **kwargs,
) -> dict:
    """Call LLM and return parsed JSON.

    Extraction order:
      1. Direct json.loads
      2. Strip markdown fences
      3. Extract first {...} block
      4. Repair trailing commas / truncated braces (one attempt)
    Falls back to `fallback` dict if all parsing fails, or raises LLMError.
    """
    system_json = (
        system.rstrip()
        + "\n\nIMPORTANT: Reply with valid JSON only. "
          "No markdown fences, no extra text — just the JSON object."
    )
    raw = ""
    try:
        raw = call_llm(
            system_json, user,
            temperature=0.0,
            json_mode=json_mode,
            **kwargs,
        )

        if not raw or not raw.strip():
            logger.warning("LLM returned empty content for JSON request")
            if fallback is not None:
                return fallback
            raise LLMError("LLM returned empty content")

        parsed = _extract_json(raw)
        if parsed is not None:
            return parsed

        repaired = _repair_json(raw)
        if repaired is not None:
            logger.info("JSON repair succeeded")
            return repaired

        logger.warning("JSON parse failed. Raw reply (500 chars): %s", raw[:500])
        if fallback is not None:
            return fallback
        # Give a precise error for truncated responses — the caller (scope/blocker
        # agents) converts this into a user-visible "unavailable" note.
        if isinstance(raw, _TruncatedStr):
            raise LLMError(
                f"LLM output truncated (finish_reason=length, content_len={len(raw)}). "
                "Reduce max_chars or max_tokens to stay within the model's output budget."
            )
        raise LLMError("LLM returned invalid JSON that could not be repaired")

    except LLMError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("JSON parse exception: %s | raw: %s", exc, raw[:500])
        if fallback is not None:
            return fallback
        raise LLMError(f"LLM returned invalid JSON: {exc}") from exc


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    # 1. Direct
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Fenced block  ```json {...} ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # 3. Strip leading/trailing fence lines
    stripped = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped).strip()
    if stripped != text:
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass
    # 4. First { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _repair_json(raw: str) -> dict | None:
    text = raw.strip()
    # 1. Remove trailing commas before ] or }
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    if cleaned != text:
        result = _extract_json(cleaned)
        if result is not None:
            return result
        text = cleaned  # work with cleaned for subsequent repairs

    # 2. Close unclosed string then add missing brackets/braces
    # Find unmatched opening brace
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:]

    ob   = candidate.count("{");  cb  = candidate.count("}")
    osb  = candidate.count("[");  csb = candidate.count("]")
    open_str = candidate.count('"') % 2 == 1  # odd quotes → unclosed string

    suffixes_to_try = []
    close_str  = '"' if open_str else ""
    close_arr  = "]" * max(osb - csb, 0)
    close_obj  = "}" * max(ob  - cb,  0)

    # Order matters: close string → close innermost array → close objects
    suffixes_to_try.append(close_str + close_arr + close_obj)
    suffixes_to_try.append(close_str + close_obj + close_arr)
    # Also try dropping the last incomplete item (truncated mid-value)
    last_comma = candidate.rfind(",")
    if last_comma != -1:
        trimmed = candidate[:last_comma]
        t_ob = trimmed.count("{"); t_cb = trimmed.count("}")
        t_osb = trimmed.count("["); t_csb = trimmed.count("]")
        t_close = "]" * max(t_osb - t_csb, 0) + "}" * max(t_ob - t_cb, 0)
        suffixes_to_try.append(t_close)
        # try on trimmed
        for sfx in [t_close]:
            if sfx:
                try:
                    return json.loads(trimmed + sfx)
                except (json.JSONDecodeError, ValueError):
                    pass

    for suffix in suffixes_to_try:
        if suffix:
            try:
                return json.loads(candidate + suffix)
            except (json.JSONDecodeError, ValueError):
                pass

    # 3. Try interleaved closing: } then ] then } (for nested object-in-array)
    if close_obj and close_arr:
        interleaved = close_str + "}" + "]" + "}" * (max(ob - cb, 0) - 1)
        if interleaved:
            try:
                return json.loads(candidate + interleaved)
            except (json.JSONDecodeError, ValueError):
                pass
    return None

# ---------------------------------------------------------------------------
# Context builder — with TTL cache to avoid re-embedding across agents
# ---------------------------------------------------------------------------

# Max characters of context sent per LLM prompt.
# Free tier: 8000 TPM (input + output tokens combined).
# Combined agent passes max_chars explicitly; individual agents use tpm_context_cap().
# This cap applies when no max_chars is passed — set to 5000 chars (~1250 tokens),
# which is safe for any single-agent call within the 8K TPM budget.
_MAX_CONTEXT_CHARS = 5000

# Free-tier TPM cap.  Agents use tpm_context_cap() to compute a per-call
# char limit that fills exactly the space left after system prompt + output.
_TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "8000"))


def tpm_context_cap(system_prompt: str, max_output_tokens: int,
                    framing_tokens: int = 60) -> int:
    """Return the max context chars that keep the total request under _TPM_LIMIT.

    Formula (all in tokens, 1 token ≈ 4 chars):
        context_tokens = TPM_LIMIT - system_tokens - framing_tokens - max_output_tokens
        context_chars  = context_tokens * 4

    Args:
        system_prompt:      the system prompt string that will be sent.
        max_output_tokens:  the max_tokens value passed to the LLM call.
        framing_tokens:     overhead for user-prompt boilerplate (default 60).

    Returns a value between 500 and _MAX_CONTEXT_CHARS (never negative).
    """
    system_tokens = len(system_prompt) // 4
    available     = _TPM_LIMIT - system_tokens - framing_tokens - max_output_tokens
    chars         = max(available, 125) * 4   # at least 500 chars
    return min(int(chars), _MAX_CONTEXT_CHARS)

# ---------------------------------------------------------------------------
# LLM result cache — avoids re-calling Groq when inputs are identical
# ---------------------------------------------------------------------------
# Key: (project_id, agent_name, model, max_tokens, context_hash)
# Value: (cached_at: float, result: dict)
_llm_result_cache: dict[tuple, tuple[float, dict]] = {}
_LLM_RESULT_CACHE_TTL = 86400  # 24 hours — survives the full working day

# Disk cache directory — survives Flask restart
_LLM_CACHE_DIR = _PROJECT_ROOT / "llm_cache"
_LLM_CACHE_DIR.mkdir(exist_ok=True)


def _llm_cache_path(project_id: str) -> Path:
    """Return the disk path for a project's LLM cache file."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", project_id).strip("_-") or "default"
    return _LLM_CACHE_DIR / f"{safe_id}.json"


def _load_disk_cache(project_id: str) -> dict:
    """Load all disk-cached LLM results for a project."""
    path = _llm_cache_path(project_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_disk_cache(project_id: str, entries: dict) -> None:
    """Write disk-cached LLM results for a project."""
    path = _llm_cache_path(project_id)
    try:
        path.write_text(
            json.dumps(entries, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write LLM cache to disk: %s", exc)


def _llm_result_cache_key(agent_name: str, project_id: str,
                          system: str, context: str,
                          max_tokens: int, model: str) -> tuple:
    """Deterministic cache key for an LLM agent call.

    Uses MD5 of system+context so the key is compact and stable.  If the
    project gains new documents (chunk count changes), context will differ and
    produce a different hash — automatically invalidating the cached result.
    """
    payload = f"{agent_name}|{project_id}|{model}|{max_tokens}|{system}|{context}"
    h = hashlib.md5(payload.encode(errors="replace")).hexdigest()
    return (agent_name, project_id, model, max_tokens, h)


def get_cached_llm_result(agent_name: str, project_id: str,
                           system: str, context: str,
                           max_tokens: int, model: str) -> dict | None:
    """Return a cached LLM result if one exists and is still valid."""
    key = _llm_result_cache_key(agent_name, project_id, system, context, max_tokens, model)

    # 1. Check in-memory cache (fast path — no disk I/O)
    entry = _llm_result_cache.get(key)
    if entry is not None:
        cached_at, result = entry
        if time.time() - cached_at < _LLM_RESULT_CACHE_TTL:
            logger.debug("LLM result cache hit (memory) for agent=%s project=%s", agent_name, project_id)
            return result
        del _llm_result_cache[key]

    # 2. Check disk cache (survives Flask restart)
    disk_key = f"{agent_name}|{model}|{max_tokens}|{key[4]}"
    disk_entries = _load_disk_cache(project_id)
    disk_entry = disk_entries.get(disk_key)
    if disk_entry is not None:
        cached_at = disk_entry.get("cached_at", 0)
        if time.time() - cached_at < _LLM_RESULT_CACHE_TTL:
            result = disk_entry.get("result")
            if result is not None:
                # Populate in-memory cache for next time
                _llm_result_cache[key] = (cached_at, result)
                logger.info("LLM result cache hit (DISK) for agent=%s project=%s", agent_name, project_id)
                return result

    return None


def set_cached_llm_result(agent_name: str, project_id: str,
                           system: str, context: str,
                           max_tokens: int, model: str,
                           result: dict) -> None:
    """Store a successful LLM result in both memory and disk."""
    key = _llm_result_cache_key(agent_name, project_id, system, context, max_tokens, model)
    now = time.time()
    _llm_result_cache[key] = (now, result)

    # Also persist to disk so it survives Flask restart
    disk_key = f"{agent_name}|{model}|{max_tokens}|{key[4]}"
    disk_entries = _load_disk_cache(project_id)
    disk_entries[disk_key] = {"cached_at": now, "result": result}
    _save_disk_cache(project_id, disk_entries)


def clear_llm_result_cache(project_id: str | None = None) -> int:
    """Remove LLM result cache entries from both memory and disk.

    If project_id is given, only that project's entries are removed.
    Returns the number of entries cleared (memory count only).
    """
    global _llm_result_cache
    if project_id is None:
        count = len(_llm_result_cache)
        _llm_result_cache = {}
        # Delete all disk cache files
        try:
            for f in _LLM_CACHE_DIR.glob("*.json"):
                f.unlink(missing_ok=True)
        except Exception:
            pass
        return count

    # Clear from memory
    keys_to_del = [k for k in _llm_result_cache if k[1] == project_id]
    for k in keys_to_del:
        del _llm_result_cache[k]

    # Clear from disk
    try:
        _llm_cache_path(project_id).unlink(missing_ok=True)
    except Exception:
        pass

    return len(keys_to_del)


def clear_context_cache(project_id: str | None = None) -> int:
    """Remove context cache entries.

    If project_id is given, only that project's entries are removed.
    Returns the number of entries cleared.
    """
    global _context_cache
    if project_id is None:
        count = len(_context_cache)
        _context_cache = {}
        return count
    keys_to_del = [k for k in _context_cache if k[0] == project_id]
    for k in keys_to_del:
        del _context_cache[k]
    return len(keys_to_del)


def build_context(project_id: str, queries: list[str], k: int = 8,
                  max_chars: int | None = None) -> str:
    """Retrieve relevant chunks from ChromaDB and return them as a text block.

    Results are cached for 5 minutes per (project, queries, k, max_chars)
    combination.  max_chars IS part of the cache key — different agents that
    use different caps get separate cache entries.

    k=8 default: with 7 files and k=4, max_per_source=ceil(4/7)=1, meaning
    each file contributed at most 1 chunk per query.  With k=8, every file
    can contribute across the 4 queries.

    Context is hard-capped at max_chars (default _MAX_CONTEXT_CHARS=8000).
    """
    cap = max_chars if max_chars is not None else _MAX_CONTEXT_CHARS
    cache_key = _context_cache_key(project_id, queries, k, max_chars)
    now = time.time()
    if cache_key in _context_cache:
        cached_at, cached_val = _context_cache[cache_key]
        if now - cached_at < _CACHE_TTL:
            logger.debug("Context cache hit for project '%s'", project_id)
            return cached_val

    try:
        from rag import search as rag_search
    except ImportError:
        return ""

    # Filename patterns that should never be sent as project evidence to agents.
    # README files contain test instructions, not project facts.
    _SKIP_SOURCES = re.compile(
        r"(?:README|INSTRUCTIONS|SETUP|INSTALL|CONTRIBUTING|CHANGELOG|LICENSE)"
        r"(?:\.[a-z]+)?$",
        re.IGNORECASE,
    )

    seen: set[str] = set()
    parts: list[str] = []
    total_chars = 0

    for query in queries:
        if total_chars >= cap:
            logger.debug("Context cap reached at %d chars — skipping remaining queries", total_chars)
            break
        try:
            hits = rag_search(project_id, query, k=k)
            for hit in hits:
                # Skip instruction/README files — they contain test guidance,
                # not project evidence, and cause false-positive extractions.
                if _SKIP_SOURCES.search(hit.source):
                    logger.debug("Skipping instruction file '%s' from context", hit.source)
                    continue
                if hit.text not in seen:
                    chunk = f"[Source: {hit.source} | Chunk #{hit.chunk_index}]\n{hit.text}"
                    if total_chars + len(chunk) > cap:
                        logger.debug(
                            "Context cap reached at %d chars for project '%s'",
                            total_chars, project_id,
                        )
                        break
                    seen.add(hit.text)
                    parts.append(chunk)
                    total_chars += len(chunk)
        except Exception as exc:
            logger.warning("Context retrieval failed for query '%s': %s", query, exc)

    result = "\n\n---\n\n".join(parts) if parts else ""
    logger.info(
        "build_context: project='%s' queries=%d k=%d cap=%d → %d chars, %d chunks",
        project_id, len(queries), k, cap, len(result), len(parts),
    )
    _context_cache[cache_key] = (now, result)
    return result