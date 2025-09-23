"""Pytest helpers enforcing optimal Ferromic build settings for benchmarks."""

from __future__ import annotations

import pytest

try:
    import ferromic as fm
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise pytest.UsageError(
        "Ferromic's Python benchmarks require the extension module to be installed. "
        "Build it with `maturin develop --release` before running these benchmarks."
    ) from exc

RUST_PROFILE = getattr(fm, "__rust_profile__", None)
RUST_OPT_LEVEL = getattr(fm, "__rust_opt_level__", None)


def _profile_description() -> str:
    if RUST_PROFILE is None:
        return "an unknown profile"
    if RUST_OPT_LEVEL is None:
        return f"the '{RUST_PROFILE}' profile"
    return f"the '{RUST_PROFILE}' profile (opt-level={RUST_OPT_LEVEL!r})"


def pytest_sessionstart(session: pytest.Session) -> None:  # pragma: no cover - exercised via pytest
    if RUST_PROFILE != "release":
        raise pytest.UsageError(
            "Ferromic's Python benchmarks require the Rust extension module to be built "
            "with Cargo's release profile for meaningful performance comparisons. "
            "Reinstall Ferromic with `maturin develop --release` (or an equivalent) before rerunning "
            f"these benchmarks. Detected {_profile_description()}."
        )


__all__ = ["pytest_sessionstart"]
