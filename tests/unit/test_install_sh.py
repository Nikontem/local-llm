import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"


def run(*args, env=None, path_tools=()):
    """Run install.sh in dry-run mode with a PATH that only contains the given fake tools."""
    suffix = "-".join(path_tools) or "none"
    bindir = Path(os.environ.get("TMPDIR", "/tmp")) / f"install-sh-{os.getpid()}-{suffix}"
    bindir.mkdir(parents=True, exist_ok=True)
    for tool in path_tools:
        fake = bindir / tool
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
    merged = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(bindir),
        "LOCAL_LLM_INSTALL_DRY_RUN": "1",
        **(env or {}),
    }
    return subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=merged)


def test_script_is_posix_sh():
    assert subprocess.run(["sh", "-n", str(SCRIPT)]).returncode == 0
    assert SCRIPT.read_text().startswith("#!/bin/sh\n")


def test_help_and_bad_flag():
    assert run("--help").returncode == 0 and "Usage" in run("--help").stdout
    bad = run("--bogus")
    assert bad.returncode == 2 and "Usage" in bad.stderr + bad.stdout


def test_brew_path_is_preferred_when_brew_exists():
    result = run("-y", path_tools=("brew",))
    assert result.returncode == 0, result.stderr
    assert "brew tap nikontem/tap" in result.stdout and "brew install local-llm" in result.stdout
    assert "uv tool install" not in result.stdout


def test_uv_path_installs_uv_when_missing():
    result = run("--uv", "-y", path_tools=())
    assert result.returncode == 0, result.stderr
    assert "astral.sh/uv/install.sh" in result.stdout
    assert "uv tool install git+https://github.com/Nikontem/local-llm" in result.stdout
    assert "local-llm setup" in result.stdout


def test_source_override_and_upgrade():
    result = run(
        "--uv", "--upgrade", "-y", env={"LOCAL_LLM_SOURCE": "/src/local-llm"}, path_tools=("uv",)
    )
    assert result.returncode == 0, result.stderr
    assert (
        "uv tool install --upgrade /src/local-llm" in result.stdout
        or "uv tool install --reinstall /src/local-llm" in result.stdout
    )
    assert "astral.sh" not in result.stdout


def test_brew_flag_without_brew_fails_with_instructions():
    result = run("--brew", "-y", path_tools=())
    assert result.returncode == 1 and "brew.sh" in result.stderr + result.stdout
