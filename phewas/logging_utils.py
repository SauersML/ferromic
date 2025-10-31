"""Utilities for mirroring phenotype-specific diagnostics to dedicated log files."""
from __future__ import annotations

import io
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO
import re

LOG_DIRECTORY = Path("logs")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_CLEARED_LOG_PATHS: set[Path] = set()
_CLEARED_LOCK = threading.Lock()
_STATE = threading.local()


def _sanitize_component(value: str) -> str:
    sanitized = _SANITIZE_RE.sub("_", str(value))
    sanitized = sanitized.strip("._-") or "phenotype"
    if len(sanitized) > 128:
        sanitized = sanitized[:128]
    return sanitized


def resolve_log_path(name: str, *, directory: str | Path | None = None) -> Path:
    base = Path(directory) if directory is not None else LOG_DIRECTORY
    return base / f"{_sanitize_component(name)}.log"


def _ensure_prepared(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CLEARED_LOCK:
        if path in _CLEARED_LOG_PATHS:
            return
        path.write_text("", encoding="utf-8")
        _CLEARED_LOG_PATHS.add(path)


class _TeeStream(io.TextIOBase):
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:  # type: ignore[override]
        written = self._primary.write(s)
        self._primary.flush()
        self._secondary.write(s)
        self._secondary.flush()
        return written

    def flush(self) -> None:  # type: ignore[override]
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:  # type: ignore[override]
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:  # type: ignore[override]
        if hasattr(self._primary, "fileno"):
            return self._primary.fileno()  # type: ignore[return-value]
        raise OSError("Underlying stream does not expose fileno().")

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._primary, "encoding", "utf-8")

    def writable(self) -> bool:  # type: ignore[override]
        return True


def _push_active(path: Path) -> None:
    stack = getattr(_STATE, "stack", [])
    stack.append(path)
    _STATE.stack = stack


def _pop_active() -> None:
    stack = getattr(_STATE, "stack", [])
    if stack:
        stack.pop()
    _STATE.stack = stack


def current_log_path() -> Path | None:
    stack = getattr(_STATE, "stack", [])
    return stack[-1] if stack else None


def is_logging_active_for(name: str, *, directory: str | Path | None = None) -> bool:
    active = current_log_path()
    if active is None:
        return False
    expected = resolve_log_path(name, directory=directory)
    return active == expected


def append_line(name: str, text: str, *, directory: str | Path | None = None) -> Path:
    path = resolve_log_path(name, directory=directory)
    _ensure_prepared(path)
    line = text if text.endswith("\n") else f"{text}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return path


@contextmanager
def phenotype_logging(name: str, *, directory: str | Path | None = None) -> Iterator[Path]:
    path = resolve_log_path(name, directory=directory)
    _ensure_prepared(path)
    with path.open("a", encoding="utf-8") as handle:
        tee_stdout = _TeeStream(sys.stdout, handle)
        tee_stderr = _TeeStream(sys.stderr, handle)
        previous_stdout, previous_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_stdout, tee_stderr
        _push_active(path)
        try:
            yield path
        finally:
            _pop_active()
            sys.stdout, sys.stderr = previous_stdout, previous_stderr

