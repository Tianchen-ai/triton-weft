"""Shared Canonical compiler and native execution support for embedded DSLs.

This package deliberately provides no Python parser, JIT decorator or tuner.
The embedding language owns those services.
"""

__version__ = "0.1.1"


def sdk_path():
    from pathlib import Path
    return Path(__file__).resolve().parent
