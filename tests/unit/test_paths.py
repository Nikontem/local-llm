from pathlib import Path

from local_llm.paths import SERVICE, Paths


def test_defaults_follow_xdg_layout(tmp_path):
    p = Paths.from_env(env={}, home=tmp_path)
    assert p.config_dir == tmp_path / ".config" / "local-llm"
    assert p.preset == tmp_path / ".config" / "local-llm" / "models.ini"
    assert p.settings_file == tmp_path / ".config" / "local-llm" / "settings.toml"
    assert p.state_dir == tmp_path / ".local" / "state" / "local-llm"
    assert p.log_dir == tmp_path / ".local" / "state" / "local-llm" / "logs"
    assert p.pid_file == p.state_dir / f"{SERVICE}.pid"
    assert p.ui_file == p.state_dir / f"{SERVICE}.ui"
    assert p.hub_cache_file == p.state_dir / "hub-cache.json"


def test_xdg_variables_are_honoured(tmp_path):
    env = {"XDG_CONFIG_HOME": str(tmp_path / "cfg"), "XDG_STATE_HOME": str(tmp_path / "st")}
    p = Paths.from_env(env=env, home=tmp_path)
    assert p.config_dir == tmp_path / "cfg" / "local-llm"
    assert p.state_dir == tmp_path / "st" / "local-llm"


def test_local_llm_variables_override_everything(tmp_path):
    env = {
        "XDG_CONFIG_HOME": str(tmp_path / "cfg"),
        "LOCAL_LLM_CONFIG_DIR": "/opt/llm",
        "LOCAL_LLM_PRESET": "/opt/llm/other.ini",
        "LOCAL_LLM_STATE_DIR": "/var/llm",
        "LOCAL_LLM_LOG_DIR": "/var/log/llm",
    }
    p = Paths.from_env(env=env, home=tmp_path)
    assert p.config_dir == Path("/opt/llm")
    assert p.preset == Path("/opt/llm/other.ini")
    assert p.settings_file == Path("/opt/llm/settings.toml")
    assert p.state_dir == Path("/var/llm")
    assert p.log_dir == Path("/var/log/llm")


def test_ensure_state_dirs_creates_private_dirs(tmp_path):
    p = Paths.from_env(env={}, home=tmp_path)
    p.ensure_state_dirs()
    assert p.state_dir.is_dir() and p.log_dir.is_dir()
    assert oct(p.state_dir.stat().st_mode & 0o777) == "0o700"
    assert oct(p.log_dir.stat().st_mode & 0o777) == "0o700"
