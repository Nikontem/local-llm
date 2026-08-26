"""Talking to the Hugging Face Hub. This milestone: is there a usable token?"""

from __future__ import annotations

import threading
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

    # Run the probe on a daemon thread rather than a ThreadPoolExecutor: the pool's
    # worker is a non-daemon thread that concurrent.futures joins at interpreter
    # exit, so shutdown(wait=False) does not stop a blackholed whoami() call from
    # stalling process exit long past `timeout`. A daemon thread is abandoned
    # cleanly when the process exits, whether or not it ever finishes.
    done = threading.Event()
    outcome: dict[str, object] = {}

    def whoami() -> None:
        try:
            outcome["info"] = huggingface_hub.HfApi(token=token).whoami()
        except Exception as error:  # noqa: BLE001 - forwarded to the caller thread below
            outcome["error"] = error
        finally:
            done.set()

    threading.Thread(target=whoami, daemon=True).start()
    if not done.wait(timeout):
        return TokenStatus("unreachable")

    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, HfHubHTTPError):
            response = getattr(error, "response", None)
            if response is not None and getattr(response, "status_code", None) in (401, 403):
                return TokenStatus("invalid")
        return TokenStatus("unreachable")

    info = outcome.get("info")
    return TokenStatus("valid", info.get("name") if isinstance(info, dict) else None)
