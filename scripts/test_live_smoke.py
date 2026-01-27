"""Simple smoke tests for live KIS helper imports and FDR OHLCV fetch.

Run locally to reproduce the failing areas without running the whole engine.
Prints concise status lines to help debugging.
"""

import importlib
import traceback
import sys
import os

# Ensure project root is on sys.path when running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from infra.logger import get_logger

logger = get_logger("scripts.test_live_smoke")


def test_lazy_imports():
    logger.info("Testing lazy import of live.kis submodules...")
    # Load submodules directly from live/kis files (avoid package __init__ issues)
    import os
    import importlib.util

    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    kis_dir = os.path.join(here, "live", "kis")
    modules = {
        "kis_auth": os.path.join(kis_dir, "kis_auth.py"),
        "domestic_stock_functions": os.path.join(
            kis_dir, "domestic_stock_functions.py"
        ),
        "auth_functions": os.path.join(kis_dir, "auth_functions.py"),
    }

    for name, path in modules.items():
        try:
            spec = importlib.util.spec_from_file_location(f"live.kis.{name}", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            logger.info("Loaded %s from %s", name, path)
        except Exception as e:
            logger.error("Failed loading %s: %s", name, e)
            traceback.print_exc()


def test_kisclient_helpers():
    logger.info("Testing KISClient helper loading (no auth call)...")
    from infra.api.broker.kis import KISClient

    client = KISClient(mock=True)
    try:
        client._load_helpers()
        logger.info("_load_helpers() succeeded")
    except Exception as e:
        logger.error("_load_helpers() failed: %s", e)
        traceback.print_exc()


def test_fdr_ohlcv():
    logger.info("Testing FDR OHLCV fetch for 005930 (Samsung)...")
    from infra.api.broker.kis import KISClient

    client = KISClient(mock=True)
    try:
        df = client.get_ohlcv("005930")
        if df is None or df.empty:
            logger.error("FDR returned empty frame for 005930")
        else:
            logger.info("FDR fetch OK: shape=%s", getattr(df, "shape", "?"))
    except Exception as e:
        logger.error("get_ohlcv failed: %s", e)
        traceback.print_exc()


if __name__ == "__main__":
    test_lazy_imports()
    test_kisclient_helpers()
    test_fdr_ohlcv()
    logger.info("Smoke tests complete")
