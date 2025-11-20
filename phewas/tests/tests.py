"""Compatibility shim for legacy ``phewas.tests`` imports.

The consolidated test suite now lives in :mod:`phewas.test_suite`,
so this module simply re-exports its public API to preserve the old
import path expected by several integration tests.
"""

from .test_suite import *  # noqa: F401,F403

__all__ = [name for name in globals() if not name.startswith("_")]
