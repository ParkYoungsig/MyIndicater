from __future__ import annotations

from typing import Optional

from infra.api.slack import send_slack_message


class SlackNotifier:
    def __init__(self, webhook_url: Optional[str]) -> None:
        self.webhook_url = webhook_url

    def notify(self, message: str) -> bool:
        return send_slack_message(message, self.webhook_url)
