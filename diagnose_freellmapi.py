"""
diagnose_freellmapi.py -- Standalone FreeLLMAPI Health & Compatibility Diagnostic
==================================================================================

Performs layered checks:
  Check 1  - Server reachability on both localhost and 127.0.0.1
  Check 2  - GET /v1/models  (lists available upstream models)
  Check 3  - POST /v1/chat/completions with model="auto"
  Check 4  - POST /v1/chat/completions with first explicitly-named model from /v1/models
  Check 5  - Repeat Check 3/4 WITHOUT Authorization header (keyless mode)

Run from the GlideText directory:
    python diagnose_freellmapi.py

The script is self-contained -- no GlideText imports needed.
"""

import json
import os
import socket
import sys
import time
import traceback

try:
    import requests
    from requests.exceptions import ConnectionError as ReqConnectionError
    from requests.exceptions import ConnectTimeout, ReadTimeout, RequestException
except ImportError:
    print("[FATAL] 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration (mirrors ai_brain.py defaults)
# ---------------------------------------------------------------------------
PORT = int(os.getenv("FREELLMAPI_PORT", "3001"))
HOSTS = [f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"]
FREELLM_BASE = f"http://127.0.0.1:{PORT}/v1"  # always prefer explicit IPv4
FREELLMAPI_API_KEY = os.getenv("FREELLMAPI_API_KEY", "").strip()

CONNECT_TIMEOUT = 4.0   # seconds to wait for TCP connect
READ_TIMEOUT    = 15.0  # seconds to wait for response body

TEST_PAYLOAD_TEXT = "testing one two three"
TEST_SYSTEM_PROMPT = (
    "You are a silent speech-to-text polish engine. "
    "Clean the raw dictation. Output ONLY the cleaned text, nothing else."
)

SEPARATOR = "=" * 70

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _tcp_reachable(host: str, port: int, timeout: float = 2.0) -> tuple[bool, float]:
    """Return (is_open, latency_ms) for a raw TCP connect."""
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            latency = (time.perf_counter() - t0) * 1000
            return True, round(latency, 1)
    except Exception:
        latency = (time.perf_counter() - t0) * 1000
        return False, round(latency, 1)


def _make_headers(include_auth: bool) -> dict:
    headers = {"Content-Type": "application/json"}
    if include_auth:
        key = FREELLMAPI_API_KEY if FREELLMAPI_API_KEY else "free"
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post_completion(model: str, include_auth: bool) -> dict:
    """
    POST /v1/chat/completions and return a result dict with:
      - ok (bool)
      - status_code (int | None)
      - latency_ms (float)
      - response_headers (dict)
      - body (str)
      - error (str | None)
      - parsed_text (str | None)
    """
    url = f"{FREELLM_BASE}/chat/completions"
    headers = _make_headers(include_auth)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user",   "content": f'Clean this spoken dictation: "{TEST_PAYLOAD_TEXT}"'},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
        "stream": False,
    }

    t0 = time.perf_counter()
    result = {
        "ok": False,
        "status_code": None,
        "latency_ms": 0.0,
        "response_headers": {},
        "body": "",
        "error": None,
        "parsed_text": None,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["status_code"] = resp.status_code
        result["response_headers"] = dict(resp.headers)
        result["body"] = resp.text[:600]

        if resp.status_code == 200:
            try:
                data = resp.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                result["parsed_text"] = text
                result["ok"] = bool(text)
            except Exception as parse_err:
                result["error"] = f"JSON parse error: {parse_err}"
        else:
            result["error"] = f"HTTP {resp.status_code}"

    except ConnectTimeout:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["error"] = f"ConnectTimeout after {CONNECT_TIMEOUT}s — server not reachable or port not open."
    except ReadTimeout:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["error"] = f"ReadTimeout after {READ_TIMEOUT}s — server accepted connection but never responded."
    except ReqConnectionError as e:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["error"] = f"ConnectionError (server not running?): {e}"
    except RequestException as e:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["error"] = f"RequestException: {e}"
    except Exception as e:
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["error"] = f"Unexpected error: {e}\n{traceback.format_exc()}"

    return result


def _print_completion_result(label: str, model: str, with_auth: bool, result: dict) -> None:
    auth_tag = "WITH Auth header" if with_auth else "WITHOUT Auth header"
    print(f"\n  >> {label} | model='{model}' | {auth_tag}")
    print(f"     Latency      : {result['latency_ms']} ms")
    if result["status_code"] is not None:
        print(f"     HTTP Status  : {result['status_code']}")
    if result["error"]:
        print(f"     ERROR        : {result['error']}")
    if result["status_code"] and result["status_code"] != 200:
        print(f"     Resp Headers : {json.dumps(result['response_headers'], indent=8)}")
        print(f"     Resp Body    :\n{result['body']}")
    if result["parsed_text"]:
        print(f"     Polished Text: {repr(result['parsed_text'])}")
    status = "PASS" if result["ok"] else "FAIL"
    print(f"     Result       : [{status}]")


# ---------------------------------------------------------------------------
# Check 1: TCP reachability
# ---------------------------------------------------------------------------

def check_1_reachability() -> list[str]:
    """Return list of reachable base URLs."""
    _print_section("CHECK 1: TCP Reachability (port scan)")
    reachable = []
    for base_url in HOSTS:
        host = base_url.split("//")[1].split(":")[0]
        ok, latency = _tcp_reachable(host, PORT, timeout=CONNECT_TIMEOUT)
        tag = "OPEN" if ok else "CLOSED/TIMEOUT"
        print(f"  {base_url:<35}  TCP {tag}  ({latency} ms)")
        if ok:
            reachable.append(base_url)
    if not reachable:
        print(
            f"\n  [DIAGNOSIS] Port {PORT} is not accepting connections on ANY address.\n"
            f"  FreeLLMAPI server is NOT RUNNING. Start it with:\n"
            f"    cd <freellmapi-dir> && npm run dev\n"
            f"  or let GlideText auto-start it by running main.py."
        )
    return reachable


# ---------------------------------------------------------------------------
# Check 2: GET /v1/models
# ---------------------------------------------------------------------------

def check_2_models() -> list[str]:
    """Return list of model IDs advertised by FreeLLMAPI."""
    _print_section("CHECK 2: GET /v1/models  (available model list)")
    url = f"{FREELLM_BASE}/models"
    model_ids = []
    t0 = time.perf_counter()
    try:
        for include_auth in (True, False):
            headers = _make_headers(include_auth)
            auth_tag = "WITH" if include_auth else "WITHOUT"
            try:
                resp = requests.get(url, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
                latency = round((time.perf_counter() - t0) * 1000, 1)
                print(f"  GET {url}  [{auth_tag} Auth]  -> HTTP {resp.status_code}  ({latency} ms)")

                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        items = data.get("data", [])
                        model_ids = [m.get("id", "") for m in items if m.get("id")]
                        print(f"  Found {len(model_ids)} model(s):")
                        for mid in model_ids:
                            print(f"    - {mid}")
                    except Exception:
                        print(f"  Could not parse JSON: {resp.text[:300]}")
                    break  # success -- no need to retry without auth
                else:
                    print(f"  Resp body: {resp.text[:300]}")
            except (ConnectTimeout, ReqConnectionError) as ce:
                latency = round((time.perf_counter() - t0) * 1000, 1)
                print(f"  GET {url}  [{auth_tag} Auth]  -> CONNECTION ERROR ({latency} ms): {ce}")

    except Exception as e:
        print(f"  Unexpected error in check_2: {e}")

    if not model_ids:
        print("  [NOTE] No models returned. 'auto' routing may still work if FreeLLMAPI has a default provider.")
    return model_ids


# ---------------------------------------------------------------------------
# Check 3/4/5: POST /v1/chat/completions with various models & auth configs
# ---------------------------------------------------------------------------

def check_3_completions(discovered_models: list[str]) -> None:
    _print_section("CHECK 3: POST /v1/chat/completions  (chat completion payloads)")

    test_cases: list[tuple[str, str, bool]] = []

    # Always test model="auto" first, with and without auth
    test_cases.append(("3a", "auto", True))
    test_cases.append(("3b", "auto", False))

    # Test the first named model if available
    if discovered_models:
        first_named = discovered_models[0]
        test_cases.append(("3c", first_named, True))
        test_cases.append(("3d", first_named, False))

    # Test common fallback model names even if not in /v1/models list
    for fallback_model in [
        "groq/llama-3.3-70b-versatile",
        "llama-3.3-70b-versatile",
        "sambanova/Meta-Llama-3.1-8B-Instruct",
    ]:
        if fallback_model not in discovered_models:
            test_cases.append(("3e", fallback_model, True))
            break  # Only test one extra fallback to keep output concise

    any_passed = False
    best_model = None
    for label, model, with_auth in test_cases:
        result = _post_completion(model, with_auth)
        _print_completion_result(label, model, with_auth, result)
        if result["ok"] and not any_passed:
            any_passed = True
            best_model = model

    _print_section("CHECK 3: SUMMARY")
    if any_passed:
        print(f"  [SUCCESS] FreeLLMAPI chat completion works.")
        print(f"  [BEST MODEL] Use model='{best_model}' in ai_brain.py for reliable Tier 1 polish.")
    else:
        print("  [FAILURE] All chat completion attempts failed.")
        print("  Root cause candidates:")
        print("    1. FreeLLMAPI server is not running (most likely -- see Check 1).")
        print("    2. model='auto' has no default route configured in FreeLLMAPI.")
        print("    3. Authorization header is required but API key is missing/wrong.")
        print("    4. Payload format mismatch (unlikely -- standard OpenAI schema used).")
        print("    5. IPv6 localhost resolution issue (use 127.0.0.1 explicitly).")
        print("\n  ACTION REQUIRED:")
        print("    a) Start FreeLLMAPI:  cd <freellmapi-dir> && npm run dev")
        print("    b) Set FREELLMAPI_API_KEY env var if your FreeLLMAPI instance requires a key.")
        print("    c) Re-run this script after starting the server.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(SEPARATOR)
    print("  GlideText -- FreeLLMAPI Diagnostic Script")
    print(f"  Target: {FREELLM_BASE}")
    print(f"  Auth key configured: {'YES (from env FREELLMAPI_API_KEY)' if FREELLMAPI_API_KEY else 'NO  (will try keyless and dummy Bearer free)'}")
    print(SEPARATOR)

    reachable = check_1_reachability()

    if not reachable:
        _print_section("SKIPPING CHECKS 2 & 3 -- Server unreachable")
        print("  Start FreeLLMAPI and re-run this script.")
        print(f"\n  Quick start:\n    cd <freellmapi-dir>\n    npm run dev\n")
        sys.exit(1)

    discovered_models = check_2_models()
    check_3_completions(discovered_models)

    print(f"\n{SEPARATOR}")
    print("  Diagnostic complete. Review the output above for actionable steps.")
    print(SEPARATOR)
