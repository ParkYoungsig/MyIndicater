from __future__ import annotations

from typing import Optional

import requests


def send_slack_message(message: str, webhook_url: Optional[str]) -> bool:
    if not webhook_url:
        return False
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=10)
        return 200 <= resp.status_code < 300
    except Exception:
        return False
