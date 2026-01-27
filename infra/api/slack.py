from __future__ import annotations

from typing import Optional


def send_slack_message(text: str, webhook_url: Optional[str]) -> bool:
    if not webhook_url:
        return False
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "requests가 필요합니다. `pip install requests` 실행 후 재시도하세요."
        ) from exc

    resp = requests.post(webhook_url, json={"text": text}, timeout=5)
    return resp.status_code == 200
