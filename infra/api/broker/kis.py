from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class KISClient:
    """KIS client wrapper that uses provided `live.kis` helpers.

    Local simulation has been removed. All account/position/order operations
    require the `live.kis` helpers and successful 모의(vps) auth. If those are
    not available the methods will raise `RuntimeError` to avoid any local
    state simulation.
    """

    def __init__(
        self,
        *,
        mock: bool | None = None,
        initial_cash: float = 10_000_000,
    ) -> None:
        self.mock = True if mock is None else bool(mock)
        self.initial_cash = float(initial_cash)
        self._kis = None
        self._ks = None
        self._auth_ok = False
        # Defer importing live.kis helpers until first use to avoid import-time
        # failures when optional dependencies or config files are missing.
        self._ka = None
        self._ks = None
        self._helpers_loaded = False

    def _require_kis(self) -> None:
        # Try to load helpers lazily if not yet loaded
        if not getattr(self, "_helpers_loaded", False):
            try:
                self._load_helpers()
                self._helpers_loaded = True
            except Exception as exc:
                raise RuntimeError(
                    "KIS 모의 API 헬퍼를 로드할 수 없습니다. live/kis/kis_devlp.yaml 존재 여부와 의존성을 확인하세요. 원인: %s"
                    % exc
                ) from exc

        if getattr(self, "_ks", None) is None:
            raise RuntimeError(
                "KIS 모듈이 로드되었지만 내부 도우미(domestic_stock_functions)가 없습니다. live/kis를 확인하세요."
            )

        # If helpers are present but not authenticated, attempt an auth retry.
        if not getattr(self, "_auth_ok", False):
            try:
                ka = getattr(self, "_ka", None)
                if ka is None:
                    raise RuntimeError("KIS auth helper (kis_auth) 없음")
                logger.info("KIS: 인증 상태가 아닙니다. 인증을 시도합니다...")
                ka.auth(svr="vps")
                self._auth_ok = True
                logger.info("KIS: 인증 성공")
            except Exception as exc:  # pragma: no cover - runtime auth failures
                raise RuntimeError(
                    f"KIS 인증 실패: {exc} — live/kis 설정과 네트워크를 확인하세요."
                ) from exc

    def _load_helpers(self) -> None:
        """Attempt to import the live.kis helper modules and assign them.

        This raises the original ImportError/FileNotFoundError to aid debugging.
        """
        import importlib.util
        import os

        # Resolve project root (three levels up from infra/api/broker)
        here = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
        kis_dir = os.path.join(project_root, "live", "kis")

        ka_path = os.path.join(kis_dir, "kis_auth.py")
        ks_path = os.path.join(kis_dir, "domestic_stock_functions.py")

        if not os.path.exists(ka_path):
            raise FileNotFoundError(f"kis_auth.py not found at {ka_path}")
        if not os.path.exists(ks_path):
            raise FileNotFoundError(
                f"domestic_stock_functions.py not found at {ks_path}"
            )

        try:
            # Ensure the live/kis directory is on sys.path so submodules using
            # plain 'import kis_auth' can resolve local imports.
            import sys

            added = False
            if kis_dir not in sys.path:
                sys.path.insert(0, kis_dir)
                added = True

            spec_ka = importlib.util.spec_from_file_location(
                "live.kis.kis_auth", ka_path
            )
            ka = importlib.util.module_from_spec(spec_ka)
            assert spec_ka.loader is not None
            spec_ka.loader.exec_module(ka)

            spec_ks = importlib.util.spec_from_file_location(
                "live.kis.domestic_stock_functions", ks_path
            )
            ks = importlib.util.module_from_spec(spec_ks)
            assert spec_ks.loader is not None
            spec_ks.loader.exec_module(ks)
        except Exception as exc:
            logger.error("live.kis helpers load failed from files: %s", exc)
            raise
        finally:
            try:
                if added:
                    sys.path.remove(kis_dir)
            except Exception:
                pass

        self._ka = ka
        self._ks = ks

    def get_ohlcv(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        try:
            import FinanceDataReader as fdr
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FinanceDataReader가 필요합니다. `pip install finance-datareader` 실행 후 재시도하세요."
            ) from exc
        # Normalize symbol: prefer 6-digit numeric KRX codes. If symbol
        # contains non-digits, attempt to extract digits; otherwise pass
        # through and handle HTTP errors gracefully.
        sym = str(symbol).strip()
        # extract first 6-digit chunk if present
        import re

        m = re.search(r"(\d{6})", sym)
        if m:
            sym = m.group(1)

        try:
            df = fdr.DataReader(sym, start, end)
            return df
        except Exception as exc:  # pragma: no cover - runtime network errors
            # Fail fast: log error and raise so upstream aborts the run.
            logger.error(
                "get_ohlcv failed for symbol=%s (sym=%s): %s", symbol, sym, exc
            )
            raise RuntimeError(f"get_ohlcv failed for symbol={symbol}: {exc}") from exc

    def get_price(self, symbol: str) -> float:
        df = self.get_ohlcv(symbol, None, None)
        if df.empty:
            return 0.0
        for col in ("Close", "close"):
            if col in df.columns:
                return float(df[col].iloc[-1])
        return float(df.iloc[-1].iloc[0])

    def get_positions(self) -> Dict[str, int]:
        self._require_kis()
        try:
            ka = self._ka
            ks = self._ks
            env = "demo"
            cano = ka.getTREnv().my_acct
            acnt_prdt_cd = ka.getTREnv().my_prod
            df1, df2 = ks.inquire_balance(
                env, cano, acnt_prdt_cd, "N", "02", "01", "N", "N", "00"
            )
            positions: Dict[str, int] = {}
            if df1 is not None and not df1.empty:
                code_col = next(
                    (
                        c
                        for c in df1.columns
                        if str(c).lower() in ("pdno", "code", "symbol", "pd_no")
                    ),
                    None,
                )
                qty_col = next(
                    (
                        c
                        for c in df1.columns
                        if "qty" in str(c).lower()
                        or "hldg" in str(c).lower()
                        or "수량" in str(c)
                    ),
                    None,
                )
                if code_col and qty_col:
                    for _, row in df1.iterrows():
                        try:
                            code = str(row[code_col]).zfill(6)
                            qty = (
                                int(float(row[qty_col]))
                                if pd.notna(row[qty_col])
                                else 0
                            )
                            if qty > 0:
                                positions[code] = qty
                        except Exception:
                            continue
            return positions
        except Exception as e:
            logger.warning("KIS inquire_balance failed: %s", e)
            raise

    def get_cash(self) -> float:
        self._require_kis()
        try:
            ka = self._ka
            ks = self._ks
            env = "demo"
            cano = ka.getTREnv().my_acct
            acnt_prdt_cd = ka.getTREnv().my_prod
            df1, df2 = ks.inquire_account_balance(cano, acnt_prdt_cd)
            if df2 is not None and not df2.empty:
                for c in df2.columns:
                    lc = str(c).lower()
                    if any(
                        k in lc
                        for k in (
                            "cash",
                            "ord_psbl",
                            "ord_psbl_cash",
                            "ord_psbl_amt",
                            "ordpsbl",
                            "예수금",
                            "balance",
                            "avail",
                        )
                    ):
                        val = df2.iloc[0][c]
                        try:
                            return float(val)
                        except Exception:
                            continue
            dfb1, dfb2 = ks.inquire_balance_rlz_pl(
                cano=cano,
                acnt_prdt_cd=acnt_prdt_cd,
                afhr_flpr_yn="N",
                inqr_dvsn="01",
                unpr_dvsn="01",
                fund_sttl_icld_yn="N",
                fncg_amt_auto_rdpt_yn="N",
                prcs_dvsn="00",
            )
            if dfb2 is not None and not dfb2.empty:
                for c in dfb2.columns:
                    lc = str(c).lower()
                    if any(
                        k in lc
                        for k in (
                            "cash",
                            "ord_psbl",
                            "ord_psbl_cash",
                            "ordpsbl",
                            "예수금",
                            "balance",
                            "avail",
                        )
                    ):
                        try:
                            return float(dfb2.iloc[0][c])
                        except Exception:
                            continue
        except Exception as e:
            logger.warning("KIS cash inquiry failed: %s", e)
            raise
        raise RuntimeError("계좌 잔고를 조회할 수 없습니다.")

    def list_universe(self, *, market: str = "KRX") -> list[str]:
        # No local listing; if KIS helpers provide a listing, attempt it.
        if getattr(self, "_ks", None) is not None and getattr(self, "_auth_ok", False):
            try:
                # no standard listing helper in provided kis modules; return empty
                return []
            except Exception:
                return []
        raise RuntimeError("KIS 모듈 사용 불가: 유니버스 목록을 얻을 수 없습니다.")

    def place_order(
        self, symbol: str, qty: int, side: str, order_type: str = "market"
    ) -> dict:
        self._require_kis()
        try:
            ka = self._ka
            ks = self._ks
            env = "demo"
            cano = ka.getTREnv().my_acct
            acnt_prdt_cd = ka.getTREnv().my_prod
            price = float(self.get_price(symbol))
            ord_dvsn = "00"
            df = ks.order_cash(
                env,
                side.lower(),
                cano,
                acnt_prdt_cd,
                symbol,
                ord_dvsn,
                str(int(qty)),
                str(int(price)),
                "KRX",
            )
            if df is not None and not df.empty:
                out = {c: df.iloc[0].get(c) for c in df.columns}
                out["status"] = "submitted"
                return out
            return {"status": "rejected", "reason": "empty response"}
        except Exception as e:
            logger.warning("KIS order_cash failed: %s", e)
            raise

    def cancel_order(self, order_id: str) -> dict:
        self._require_kis()
        try:
            ka = self._ka
            ks = self._ks
            env = "demo"
            cano = ka.getTREnv().my_acct
            acnt_prdt_cd = ka.getTREnv().my_prod
            df = ks.order_resv_rvsecncl(env, cano, acnt_prdt_cd, order_id)
            if df is not None and not df.empty:
                out = {c: df.iloc[0].get(c) for c in df.columns}
                out["status"] = "cancel_submitted"
                return out
            return {"status": "rejected", "reason": "empty response"}
        except Exception as e:
            logger.warning("KIS cancel failed: %s", e)
            raise
