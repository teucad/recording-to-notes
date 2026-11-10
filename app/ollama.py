"""Keeps the local Ollama server available: installs it if missing, starts it if not running."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from .compose import OLLAMA_URL

INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
AUTO_INSTALL = os.environ.get("OLLAMA_AUTO_INSTALL", "1") != "0"

_url = urlparse(OLLAMA_URL)
IS_LOCAL = _url.hostname in ("127.0.0.1", "localhost", "::1")

# state: unknown | running | starting | downloading | installing | missing | unreachable | error
_status = {"state": "unknown", "message": "", "progress": None}
_ensure_lock = threading.Lock()
_worker: threading.Thread | None = None


def status() -> dict:
    return dict(_status)


def _set(state: str, message: str = "", progress: float | None = None) -> None:
    _status.update(state=state, message=message, progress=progress)


def reachable() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=2).status_code == 200
    except Exception:
        return False


def find_exe() -> Path | None:
    found = shutil.which("ollama")
    if found:
        return Path(found)
    if os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return candidate
    return None


def _wait_reachable(seconds: int) -> bool:
    for _ in range(seconds * 2):
        if reachable():
            return True
        time.sleep(0.5)
    return False


def _start(exe: Path) -> None:
    """Run `ollama serve` in the background, without a console window."""
    env = dict(os.environ, OLLAMA_HOST=f"{_url.hostname}:{_url.port or 11434}")
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [str(exe), "serve"], env=env, creationflags=flags,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _install() -> None:
    """Download the official Windows installer and run it silently (per-user, no admin rights)."""
    installer = Path(tempfile.gettempdir()) / "OllamaSetup.exe"
    partial = installer.with_suffix(".part")
    _set("downloading", "Downloading Ollama", 0.0)
    with httpx.stream("GET", INSTALLER_URL, follow_redirects=True,
                      timeout=httpx.Timeout(60, connect=15)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with partial.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                got += len(chunk)
                _set("downloading", "Downloading Ollama", got / total if total else None)
    partial.replace(installer)
    _set("installing", "Installing Ollama")
    try:
        subprocess.run(
            [str(installer), "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
            check=True, timeout=1200,
        )
    finally:
        installer.unlink(missing_ok=True)


def ensure() -> bool:
    """Make sure Ollama is installed and running. Blocks; returns True when it is reachable."""
    with _ensure_lock:
        if reachable():
            _set("running")
            return True
        if not IS_LOCAL:
            _set("unreachable", f"Ollama at {OLLAMA_URL} is not reachable.")
            return False
        try:
            exe = find_exe()
            if exe is None:
                if os.name != "nt" or not AUTO_INSTALL:
                    _set("missing", "Ollama is not installed. Get it from https://ollama.com/download")
                    return False
                _install()
                if _wait_reachable(20):  # the installer normally launches Ollama itself
                    _set("running")
                    return True
                exe = find_exe()
                if exe is None:
                    _set("error", "Ollama was installed but could not be found. Start it manually.")
                    return False
            _set("starting", "Starting Ollama")
            _start(exe)
            if _wait_reachable(60):
                _set("running")
                return True
            _set("error", "Ollama was started but did not respond within 60 seconds.")
        except Exception as exc:
            _set("error", f"Could not set up Ollama: {exc}")
        return False


def ensure_async() -> None:
    """Run ensure() in the background (used at startup and by the reload button)."""
    global _worker
    if _worker and _worker.is_alive():
        return
    _worker = threading.Thread(target=ensure, daemon=True)
    _worker.start()
