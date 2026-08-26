import tomllib

from local_llm.paths import Paths
from local_llm.settings import Settings, dump_settings, load_settings, save_settings


def paths_for(tmp_path):
    return Paths.from_env(env={}, home=tmp_path)


def test_defaults_when_nothing_is_configured(tmp_path):
    s = load_settings(paths_for(tmp_path), env={})
    assert s == Settings()
    assert s.port == 5678 and s.host == "127.0.0.1" and s.max_models == 2
    assert s.reserve_gb == 10 and s.ui is False and s.default_model == ""
    assert s.allow_remote is False and s.api_key == ""
    assert s.openai_base_url == "http://127.0.0.1:5678/v1"
    assert s.anthropic_base_url == "http://127.0.0.1:5678"
    assert s.is_local


def test_file_then_env_then_overrides(tmp_path):
    p = paths_for(tmp_path)
    p.config_dir.mkdir(parents=True)
    p.settings_file.write_text('port = 6000\nmax_models = 3\ndefault_model = "a"\nui = true\n')
    env = {"LOCAL_LLM_PORT": "6001", "LOCAL_LLM_UI": "0", "LOCAL_LLM_API_KEY": "sekrit"}
    s = load_settings(p, env=env, overrides={"port": 6002, "host": None})
    assert s.port == 6002          # override wins over env and file
    assert s.max_models == 3       # from file
    assert s.default_model == "a"  # from file
    assert s.ui is False           # env "0" beats file true
    assert s.api_key == "sekrit"   # env only
    assert s.host == "127.0.0.1"   # None override means "not given"


def test_bool_parsing_from_env(tmp_path):
    p = paths_for(tmp_path)
    for raw, expected in [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False)]:
        assert load_settings(p, env={"LOCAL_LLM_ALLOW_REMOTE": raw}).allow_remote is expected


def test_is_local_for_loopback_names_only():
    assert Settings(host="localhost").is_local
    assert Settings(host="::1").is_local
    assert not Settings(host="0.0.0.0").is_local


def test_dump_is_valid_toml_without_the_api_key():
    text = dump_settings(Settings(port=7000, default_model='q"uoted', api_key="never"))
    data = tomllib.loads(text)
    assert data["port"] == 7000 and data["default_model"] == 'q"uoted'
    assert "api_key" not in data and "never" not in text


def test_save_writes_file_and_round_trips(tmp_path):
    p = paths_for(tmp_path)
    written = save_settings(p, Settings(port=7001, reserve_gb=4))
    assert written == p.settings_file
    assert load_settings(p, env={}) == Settings(port=7001, reserve_gb=4)
