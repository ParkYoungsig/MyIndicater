content = '''"""Lazy package initializer for live.kis helpers.

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
'''

path = r"C:\Users\dudals\Downloads\Team_Project\my_robo_advisor\live\kis\__init__.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("WROTE", path)
