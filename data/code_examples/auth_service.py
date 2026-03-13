from __future__ import annotations

import hashlib
import hmac
import time


class AuthService:
    def __init__(self, secret_key: str, token_ttl_seconds: int = 3600) -> None:
        self.secret_key = secret_key.encode("utf-8")
        self.token_ttl_seconds = token_ttl_seconds

    def create_token(self, user_id: str, issued_at: int | None = None) -> str:
        issued_at = issued_at or int(time.time())
        payload = f"{user_id}:{issued_at}"
        signature = hmac.new(self.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}:{signature}"

    def validate_token(self, token: str) -> tuple[bool, str]:
        parts = token.split(":")
        if len(parts) != 3:
            return False, "invalid token format"

        user_id, issued_at_raw, signature = parts
        payload = f"{user_id}:{issued_at_raw}"
        expected_signature = hmac.new(
            self.secret_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            return False, "signature mismatch"

        issued_at = int(issued_at_raw)
        if int(time.time()) - issued_at > self.token_ttl_seconds:
            return False, "token expired"

        return True, user_id
