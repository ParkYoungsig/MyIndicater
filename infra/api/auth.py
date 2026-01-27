from __future__ import annotations

import os


def get_env_credentials() -> dict:
    """환경변수 기반 인증 정보 조회(스켈레톤)."""
    return {
        "app_key": os.getenv("KIS_APP_KEY"),
        "app_secret": os.getenv("KIS_APP_SECRET"),
    }
