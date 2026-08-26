import time

import huggingface_hub
from huggingface_hub.errors import HfHubHTTPError

from local_llm.hub import TokenStatus, token_status


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code
        self.headers = {}
        self.request = None


def _api_with(whoami):
    class Api:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return whoami()

    return Api


def test_absent_token(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    assert token_status() == TokenStatus("absent")


def test_valid_token(monkeypatch):
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(lambda: {"name": "nikos"}))
    assert token_status() == TokenStatus("valid", "nikos")


def test_invalid_token(monkeypatch):
    def boom():
        raise HfHubHTTPError("Invalid user token", response=_Response(401))

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(boom))
    assert token_status() == TokenStatus("invalid")


def test_unreachable_on_other_errors_and_timeouts(monkeypatch):
    def offline():
        raise OSError("no network")

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_x")
    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(offline))
    assert token_status() == TokenStatus("unreachable")

    def slow():
        time.sleep(0.3)
        return {"name": "late"}

    monkeypatch.setattr(huggingface_hub, "HfApi", _api_with(slow))
    assert token_status(timeout=0.05) == TokenStatus("unreachable")
