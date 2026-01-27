"""Lazy package initializer for live.kis helpers.

This module exposes helper submodules (`kis_auth`, `domestic_stock_functions`,
`auth_functions`) lazily to avoid import-time failures when optional
dependencies or config files are missing. Accessing attributes like
`live.kis.kis_auth` will import the submodule on demand.
"""
from __future__ import annotations

__all__ = ["kis_auth", "domestic_stock_functions", "auth_functions"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        mod = importlib.import_module('.'.join((__name__, name)))
        globals()[name] = mod
        return mod
    raise AttributeError(name)
