from pathlib import Path

import pytest

from local_llm.agents import (
    AgentError,
    aider_args,
    aider_env,
    claude_env,
    copilot_env,
    exec_with_env,
    export_lines,
    qwen_env,
    resolve_model,
)
from local_llm.preset import Preset
from local_llm.settings import Settings

PRESET = Preset.parse(
    "[*]\nc = 8192\n[big]\nmodel = /b.gguf\nc = 65536\nn-predict = 32768\n"
    "[small]\nmodel = /s.gguf\n"
)


def test_resolve_model_uses_default_and_validates():
    assert resolve_model(None, PRESET, Settings(default_model="big")) == "big"
    assert resolve_model("small", PRESET, Settings(default_model="big")) == "small"
    with pytest.raises(AgentError, match="No model given"):
        resolve_model(None, PRESET, Settings())
    with pytest.raises(AgentError, match="Unknown model: nope"):
        resolve_model("nope", PRESET, Settings())


def test_claude_env():
    env = claude_env("big", PRESET, Settings(port=7000))
    assert env == {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:7000",
        "ANTHROPIC_MODEL": "big",
        "ANTHROPIC_API_KEY": "dummy",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "big",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_TELEMETRY": "1",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "65536",
    }
    assert claude_env("small", PRESET, Settings(api_key="k"))["ANTHROPIC_API_KEY"] == "k"
    assert claude_env("small", PRESET, Settings())["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "8192"


def test_claude_env_sets_the_small_model_and_quietens_extra_traffic():
    env = claude_env("small", PRESET, Settings())
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:5678"
    assert env["ANTHROPIC_MODEL"] == "small"
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "small"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "8192"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_aider_env_and_args():
    env = aider_env(Settings())
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:5678/v1"
    assert env["OPENAI_API_KEY"] == "dummy"
    assert aider_args("small", ["--no-auto-commits"]) == [
        "--model", "openai/small", "--no-auto-commits",
    ]


def test_qwen_env_sets_all_three_or_qwen_ignores_them():
    env = qwen_env("small", Settings(api_key="secret"))
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:5678/v1"
    assert env["OPENAI_MODEL"] == "small"
    assert env["OPENAI_API_KEY"] == "secret"
    assert all(value for value in env.values()), "Qwen Code needs all three non-empty"


def test_copilot_env():
    env = copilot_env("big", PRESET, Settings(), offline=False)
    assert env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:5678/v1",
        "COPILOT_PROVIDER_API_KEY": "no-key",
        "COPILOT_MODEL": "big",
        "COPILOT_OFFLINE": "false",
        "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": "65536",
        "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "32768",
    }
    small = copilot_env("small", PRESET, Settings())
    assert small["COPILOT_OFFLINE"] == "true"
    assert "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS" not in small


def test_export_lines_for_shells():
    zsh = export_lines("big", PRESET, Settings(), Path("/c/models.ini"), shell="zsh")
    assert "export OPENAI_BASE_URL=http://127.0.0.1:5678/v1\n" in zsh
    assert "export OPENAI_API_KEY=dummy\n" in zsh
    assert "export ANTHROPIC_BASE_URL=http://127.0.0.1:5678\n" in zsh
    assert "export ANTHROPIC_MODEL=big\n" in zsh
    assert "export LOCAL_LLM_OPENAI_BASE_URL=http://127.0.0.1:5678/v1\n" in zsh
    assert "export LOCAL_LLM_ANTHROPIC_BASE_URL=http://127.0.0.1:5678\n" in zsh
    assert "export LOCAL_LLM_PRESET=/c/models.ini\n" in zsh
    fish = export_lines("big", PRESET, Settings(), Path("/c/models.ini"), shell="fish")
    assert "set -gx ANTHROPIC_MODEL big\n" in fish
    quoted = export_lines("big", PRESET, Settings(api_key="a b"), Path("/c/x y.ini"))
    assert "export LOCAL_LLM_PRESET='/c/x y.ini'\n" in quoted


def test_the_key_is_referenced_not_reproduced():
    """These lines end up in scrollback, shell history and pasted terminal output.

    The key can only ever have come from LOCAL_LLM_API_KEY in the environment - it is
    the one setting settings.toml will not hold - so the variable is already there to
    point at, and eval of this output still works.
    """
    secret = Settings(api_key="sk-do-not-print-me")
    zsh = export_lines("big", PRESET, secret, Path("/c/models.ini"), shell="zsh")

    assert "sk-do-not-print-me" not in zsh
    assert 'export OPENAI_API_KEY="$LOCAL_LLM_API_KEY"\n' in zsh
    assert 'export ANTHROPIC_API_KEY="$LOCAL_LLM_API_KEY"\n' in zsh

    fish = export_lines("big", PRESET, secret, Path("/c/models.ini"), shell="fish")

    assert "sk-do-not-print-me" not in fish
    assert 'set -gx OPENAI_API_KEY "$LOCAL_LLM_API_KEY"\n' in fish

    plain = export_lines("big", PRESET, Settings(), Path("/c/models.ini"))
    assert "export OPENAI_API_KEY=dummy\n" in plain, "no key, nothing to hide"


def test_exec_with_env_replaces_process(monkeypatch):
    calls = []
    monkeypatch.setattr("local_llm.agents.shutil.which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        "local_llm.agents.os.execve",
        lambda path, argv, env: calls.append((path, argv, env)),
    )
    monkeypatch.setenv("KEEP", "1")
    exec_with_env("claude", ["--resume"], {"ANTHROPIC_MODEL": "big"})
    path, argv, env = calls[0]
    assert path == "/bin/claude" and argv == ["claude", "--resume"]
    assert env["KEEP"] == "1" and env["ANTHROPIC_MODEL"] == "big"


def test_exec_with_env_reports_missing_program(monkeypatch):
    monkeypatch.setattr("local_llm.agents.shutil.which", lambda name: None)
    with pytest.raises(AgentError, match="copilot not found in PATH"):
        exec_with_env("copilot", [], {})


def test_only_the_key_lines_are_turned_into_references():
    """A one-character key made every line whose value equalled it come out as one."""
    lines = export_lines("big", PRESET, Settings(api_key="1"), Path("/c/models.ini"))

    assert 'export OPENAI_API_KEY="$LOCAL_LLM_API_KEY"\n' in lines
    assert "export DISABLE_TELEMETRY=1\n" in lines
    assert "export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1\n" in lines
