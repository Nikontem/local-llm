"""Talking to the Hugging Face Hub. This milestone: is there a usable token?"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass


@dataclass
class TokenStatus:
    state: str  # "valid" | "invalid" | "absent" | "unreachable"
    username: str | None = None


def token_status(timeout: float = 10.0) -> TokenStatus:
    """Validate the locally stored token against the Hub, without hanging."""
    import huggingface_hub
    from huggingface_hub.errors import HfHubHTTPError

    token = huggingface_hub.get_token()
    if not token:
        return TokenStatus("absent")

    def whoami() -> dict:
        return huggingface_hub.HfApi(token=token).whoami()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(whoami)
        try:
            info = future.result(timeout=timeout)
        except FutureTimeout:
            return TokenStatus("unreachable")
        except HfHubHTTPError as error:
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) in (401, 403):
                return TokenStatus("invalid")
            return TokenStatus("unreachable")
        except Exception:  # noqa: BLE001 - any network failure means "could not check"
            return TokenStatus("unreachable")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return TokenStatus("valid", info.get("name") if isinstance(info, dict) else None)
