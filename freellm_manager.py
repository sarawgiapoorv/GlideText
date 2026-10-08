"""
freellm_manager.py -- FreeLLMAPI Headless Lifecycle Manager for GlideText
==========================================================================

Responsibilities:
  - Auto-discover the FreeLLMAPI project directory from multiple candidate
    locations (env var → config.txt saved path → common sibling / user folders).
  - Resolve npm on the system PATH (handles Windows npm.cmd vs Unix npm).
  - Spawn FreeLLMAPI's dev server silently with CREATE_NO_WINDOW so no terminal
    windows ever pop up.
  - Poll 127.0.0.1:PORT with a raw TCP socket (no HTTP round-trip overhead)
    until the server is accepting connections or a deadline is reached.
  - Save / load the confirmed FreeLLMAPI directory to config.txt so the path
    is remembered between launches even if env vars change.
  - Terminate the background Node process tree cleanly on app exit (via both
    an atexit handler and an explicit `shutdown()` call from the GUI).

Usage (in main.py):
    import freellm_manager
    freellm_manager.start()          # idempotent: safe to call even if already running
    # ... app runs ...
    freellm_manager.shutdown()       # called automatically via atexit too
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level state  (all mutations are protected by _LOCK)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_START_LOCK = threading.Lock()
_process: Optional[subprocess.Popen] = None   # the npm process we spawned
_server_ready: bool = False                    # set True once port is confirmed open
_startup_thread: Optional[threading.Thread] = None

class ProviderHealthStatus:
    API_HEALTHY = "API_HEALTHY"
    PORT_OPEN = "PORT_OPEN"
    API_OVERLOADED = "API_OVERLOADED"
    API_RATE_LIMITED = "API_RATE_LIMITED"
    API_UNAVAILABLE = "API_UNAVAILABLE"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT: int = int(os.getenv("FREELLMAPI_PORT", "3001"))
HOST: str = "127.0.0.1"                        # always explicit IPv4 (avoids ::1 on Windows)
STARTUP_POLL_TIMEOUT: float = 12.0             # seconds to wait after spawn
POLL_INTERVAL: float = 0.3                     # TCP probe frequency
TCP_CONNECT_TIMEOUT: float = 0.5              # per-probe timeout

# Key used to persist the confirmed FreeLLMAPI directory inside config.txt
_CONFIG_KEY = "FREELLMAPI_DIR"

# All name variants GlideText will scan for (case-insensitive covered by listing both)
_DIR_NAMES: list[str] = [
    "freellmapi",
    "FreeLLMAPI",
    "freellm-api",
    "FreeLLM-API",
    "free-llm-api",
    "freellm",
]

# ---------------------------------------------------------------------------
# Helpers: config.txt persistence
# ---------------------------------------------------------------------------

def _config_path() -> Path:
    return Path(__file__).parent / "config.txt"


def _load_config() -> dict[str, str]:
    """Parse config.txt as KEY=VALUE lines. Lines without '=' are skipped."""
    cfg: dict[str, str] = {}
    try:
        for line in _config_path().read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
    except Exception:
        pass
    return cfg


def _save_config_key(key: str, value: str) -> None:
    """Upsert a KEY=VALUE entry in config.txt without touching other lines."""
    cfg_file = _config_path()
    try:
        existing = cfg_file.read_text(encoding="utf-8").splitlines(keepends=True) if cfg_file.exists() else []
        new_lines: list[str] = []
        found = False
        for line in existing:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{key}={value}\n")
        cfg_file.write_text("".join(new_lines), encoding="utf-8")
    except Exception as e:
        logging.warning(f"[FreeLLM] Could not persist {key} to config.txt: {e}")


# ---------------------------------------------------------------------------
# Helpers: directory discovery
# ---------------------------------------------------------------------------

def _is_valid_freellmapi_dir(path: str) -> bool:
    """Return True if *path* looks like a FreeLLMAPI project root."""
    if not os.path.isdir(path):
        return False
    has_pkg = os.path.isfile(os.path.join(path, "package.json"))
    has_server_sub = os.path.isdir(os.path.join(path, "server"))
    return has_pkg or has_server_sub


def locate_freellmapi_dir() -> Optional[str]:
    """
    Return the FreeLLMAPI project directory, or None if not found.

    Search order (first match wins):
      1. FREELLMAPI_DIR environment variable
      2. Saved path in config.txt  (key = FREELLMAPI_DIR)
      3. Sibling of GlideText on Desktop / OneDrive Desktop / home
      4. Subfolder *inside* GlideText (edge case: bundled setup)
      5. Absolute Desktop paths (Windows: C:\\Users\\<user>\\Desktop\\*)
    """
    # 1. Env var
    env_val = os.getenv("FREELLMAPI_DIR", "").strip()
    if env_val and _is_valid_freellmapi_dir(env_val):
        logging.info(f"[FreeLLM] Directory resolved from FREELLMAPI_DIR env: {env_val}")
        return env_val

    # 2. config.txt saved path
    cfg = _load_config()
    cfg_val = cfg.get(_CONFIG_KEY, "").strip()
    if cfg_val and _is_valid_freellmapi_dir(cfg_val):
        logging.info(f"[FreeLLM] Directory resolved from config.txt: {cfg_val}")
        return cfg_val

    # 3–5. File-system scan
    self_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(self_dir)          # e.g. OneDrive/Desktop
    home = os.path.expanduser("~")

    base_dirs: list[str] = [
        parent_dir,                                              # sibling of GlideText
        self_dir,                                               # subfolder inside GlideText
        home,
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "Documents"),
        os.path.join(home, "Projects"),
        os.path.join(home, "dev"),
        os.path.join(home, "code"),
        # Windows-specific: C:\Users\<user>\Desktop regardless of shell cwd
        os.path.join(os.environ.get("USERPROFILE", home), "Desktop"),
        os.path.join(os.environ.get("USERPROFILE", home), "OneDrive", "Desktop"),
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_bases: list[str] = []
    for b in base_dirs:
        nb = os.path.normcase(os.path.abspath(b))
        if nb not in seen:
            seen.add(nb)
            unique_bases.append(b)

    for base in unique_bases:
        for name in _DIR_NAMES:
            candidate = os.path.join(base, name)
            if _is_valid_freellmapi_dir(candidate):
                logging.info(f"[FreeLLM] Directory found by scan: {candidate}")
                _save_config_key(_CONFIG_KEY, candidate)    # remember it for next launch
                return candidate

    logging.warning(
        "[FreeLLM] Directory not found. "
        "Set FREELLMAPI_DIR env var or place it adjacent to GlideText. "
        "Example: C:\\Users\\<you>\\OneDrive\\Desktop\\freellmapi"
    )
    return None


# ---------------------------------------------------------------------------
# Helpers: npm resolution
# ---------------------------------------------------------------------------

def _find_npm() -> Optional[str]:
    """Return the absolute path to npm (or npm.cmd on Windows), or None."""
    # Prefer the Windows .cmd wrapper; fall back to bare 'npm'
    candidates = ["npm.cmd", "npm"] if sys.platform == "win32" else ["npm"]
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found

    if sys.platform == "darwin":
        # Search common macOS paths where Node/npm is installed
        mac_search_dirs = [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            os.path.expanduser("~/bin"),
            os.path.expanduser("~/.local/bin"),
        ]
        # Check NVM, FNM, Volta Node versions
        nvm_dir = os.path.expanduser("~/.nvm/versions/node")
        if os.path.isdir(nvm_dir):
            try:
                for entry in sorted(os.listdir(nvm_dir), reverse=True):
                    mac_search_dirs.append(os.path.join(nvm_dir, entry, "bin"))
            except Exception:
                pass
        fnm_dir = os.path.expanduser("~/.local/share/fnm/current/bin")
        if os.path.isdir(fnm_dir):
            mac_search_dirs.append(fnm_dir)
        volta_dir = os.path.expanduser("~/.volta/bin")
        if os.path.isdir(volta_dir):
            mac_search_dirs.append(volta_dir)

        for d in mac_search_dirs:
            p = os.path.join(d, "npm")
            if os.path.isfile(p) and os.access(p, os.X_OK):
                return p

    return None


# ---------------------------------------------------------------------------
# Helpers: TCP health check (no HTTP dependency)
# ---------------------------------------------------------------------------

def _tcp_open(host: str = HOST, port: int = PORT, timeout: float = TCP_CONNECT_TIMEOUT) -> bool:
    """Return True if *host:port* accepts a TCP connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core: spawn
# ---------------------------------------------------------------------------

def _spawn(freellm_dir: str, npm_path: str) -> Optional[subprocess.Popen]:
    """
    Launch npm run dev inside *freellm_dir*, completely detached from any
    visible console window.

    Returns the Popen handle on success, None on error.
    """
    # If there is a server/ sub-workspace with its own package.json, run the
    # workspace target so npm resolves scripts correctly.
    server_sub = os.path.join(freellm_dir, "server")
    has_server_workspace = (
        os.path.isdir(server_sub)
        and os.path.isfile(os.path.join(server_sub, "package.json"))
    )
    cmd = [npm_path, "run", "dev"]
    if has_server_workspace:
        cmd = [npm_path, "run", "dev", "-w", "server"]

    import platform_compat
    log_dir = platform_compat.get_logs_dir()
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "freellmapi.log")
    try:
        log_file = open(log_file_path, "a", encoding="utf-8")
    except Exception as log_err:
        logging.warning(f"[FreeLLM] Could not open log file {log_file_path}: {log_err}")
        log_file = subprocess.DEVNULL

    logging.info(f"[FreeLLM] Spawning: {' '.join(cmd)}  cwd={freellm_dir}")
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                cmd,
                cwd=freellm_dir,
                creationflags=0x08000000,
                stdout=log_file,
                stderr=log_file,
                shell=True,
            )
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=freellm_dir,
                start_new_session=True,
                stdout=log_file,
                stderr=log_file,
                shell=False,
            )
        logging.info(f"[FreeLLM] Spawned PID {proc.pid}")
        return proc

    except FileNotFoundError:
        logging.error(f"[FreeLLM] npm not found at '{npm_path}'. Is Node.js installed?")
    except PermissionError as e:
        logging.error(f"[FreeLLM] Permission denied spawning npm: {e}")
    except Exception as e:
        logging.error(f"[FreeLLM] Failed to spawn npm process: {e}")
    return None


# ---------------------------------------------------------------------------
# Core: poll until ready
# ---------------------------------------------------------------------------

def _poll_until_ready(deadline: float) -> bool:
    """Busy-poll TCP port until open or deadline exceeded. Returns True if ready."""
    while time.time() < deadline:
        if _tcp_open():
            return True
        time.sleep(POLL_INTERVAL)
    # One final check right at the deadline
    return _tcp_open()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_health(
    host: str = HOST,
    port: int = PORT,
    timeout: float = 1.5,
    api_key: str = "",
) -> tuple[str, str]:
    """
    Perform a real API-level health check against FreeLLMAPI.

    Distinguishes:
      - API_HEALTHY: HTTP 200 returned from /v1/models (ready to accept requests)
      - API_OVERLOADED: HTTP 503 or 504
      - API_RATE_LIMITED: HTTP 429
      - PORT_OPEN: TCP connected, but API endpoint gave unexpected status or timeout
      - API_UNAVAILABLE: Connection refused or port not open

    Returns (status_enum, details_string).
    """
    if not _tcp_open(host, port, timeout=min(timeout, 0.5)):
        return ProviderHealthStatus.API_UNAVAILABLE, f"Port {port} closed or connection refused"

    url = f"http://{host}:{port}/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return ProviderHealthStatus.API_HEALTHY, "Ready (HTTP 200)"
        elif resp.status_code == 429:
            return ProviderHealthStatus.API_RATE_LIMITED, f"Rate limited (HTTP 429)"
        elif resp.status_code in (502, 503, 504):
            return ProviderHealthStatus.API_OVERLOADED, f"Overloaded (HTTP {resp.status_code})"
        else:
            return ProviderHealthStatus.PORT_OPEN, f"TCP open but API returned HTTP {resp.status_code}"
    except Exception as e:
        return ProviderHealthStatus.PORT_OPEN, f"TCP open but HTTP request failed: {e}"


def is_running() -> bool:
    """Return True if FreeLLMAPI is currently accepting TCP connections."""
    return _tcp_open()


def start(poll_timeout: float = STARTUP_POLL_TIMEOUT) -> bool:
    """
    Ensure FreeLLMAPI is running. This function is **idempotent** and thread-safe.
    """
    global _process, _server_ready

    # Fast path: already up
    if _tcp_open():
        logging.info(f"[FreeLLM] Server already listening on {HOST}:{PORT}.")
        with _LOCK:
            _server_ready = True
        return True

    with _START_LOCK:
        # Re-check inside lock
        if _tcp_open():
            with _LOCK:
                _server_ready = True
            return True

        # --- Find tools ---
        npm_path = _find_npm()
        if not npm_path:
            logging.warning(
                "[FreeLLM] npm not found on PATH. "
                "Install Node.js from https://nodejs.org and restart GlideText."
            )
            return False

        freellm_dir = locate_freellmapi_dir()
        if not freellm_dir:
            return False

        # --- Spawn ---
        with _LOCK:
            if _process is not None and _process.poll() is None:
                logging.info("[FreeLLM] Process already spawned by this session, waiting for ready...")
            else:
                proc = _spawn(freellm_dir, npm_path)
                if proc is None:
                    return False
                _process = proc
                atexit.register(shutdown)

        # --- Poll ---
        deadline = time.time() + poll_timeout
        logging.info(f"[FreeLLM] Waiting up to {poll_timeout:.0f}s for port {PORT} to open...")
        ready = _poll_until_ready(deadline)

        with _LOCK:
            _server_ready = ready

        if ready:
            logging.info(f"[FreeLLM] Server is listening on {HOST}:{PORT}.")
        else:
            logging.warning(
                f"[FreeLLM] Server did not respond within {poll_timeout:.0f}s. "
                f"FreeLLMAPI may still be starting."
            )
        return ready


def start_async(poll_timeout: float = STARTUP_POLL_TIMEOUT) -> None:
    """
    Non-blocking variant of start(). Runs the full lifecycle in a daemon
    thread so main.py can continue loading the GUI immediately.

    The server state is available via is_running() once ready.
    """
    global _startup_thread
    with _LOCK:
        if _startup_thread is not None and _startup_thread.is_alive():
            return  # already starting

    def _worker():
        start(poll_timeout=poll_timeout)

    t = threading.Thread(target=_worker, name="FreeLLM-Starter", daemon=True)
    with _LOCK:
        _startup_thread = t
    t.start()


def shutdown() -> None:
    """
    Terminate the background Node/npm process tree spawned by this session.
    Safe to call multiple times; subsequent calls after the first are no-ops.
    """
    global _process, _server_ready

    with _LOCK:
        proc = _process
        _process = None
        _server_ready = False

    if proc is None:
        return

    logging.info(f"[FreeLLM] Shutting down background server (PID {proc.pid})...")
    try:
        if sys.platform == "win32":
            # Kill the entire process tree (npm spawns node as a child)
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,   # CREATE_NO_WINDOW
                timeout=5,
            )
        else:
            # On macOS / POSIX: terminate the entire process group cleanly
            import signal
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except ProcessLookupError:
                pass
            except Exception as e:
                logging.warning(f"[FreeLLM] Error killing process group {proc.pid}: {e}")
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
    except Exception as e:
        logging.warning(f"[FreeLLM] Error during shutdown: {e}")
    finally:
        logging.info("[FreeLLM] Shutdown complete.")
