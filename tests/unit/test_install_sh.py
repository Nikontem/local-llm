import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "install.sh"


def run(*args, env=None, path_tools=()):
    """Run install.sh in dry-run mode with a PATH that only contains the given fake tools."""
    bindir = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"install-sh-{os.getpid()}-{'-'.join(path_tools) or 'none'}"
    )
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


def test_uv_is_the_only_install_path_and_brew_is_only_mentioned():
    with_brew = run(path_tools=("brew", "uv"))
    assert with_brew.returncode == 0, with_brew.stderr
    assert "Homebrew found" in with_brew.stdout
    assert "llama.cpp and hf through it" in with_brew.stdout
    assert "uv tool install git+https://github.com/Nikontem/local-llm" in with_brew.stdout
    assert "brew tap" not in with_brew.stdout and "brew install local-llm" not in with_brew.stdout
    assert "local-llm setup" in with_brew.stdout


def test_uv_is_installed_when_missing_and_brew_absence_is_explained():
    result = run(path_tools=())
    assert result.returncode == 0, result.stderr
    assert "Homebrew not found" in result.stdout and "brew.sh" in result.stdout
    assert "astral.sh/uv/install.sh" in result.stdout
    assert "uv tool install git+https://github.com/Nikontem/local-llm" in result.stdout


def test_source_override_and_upgrade():
    result = run("--upgrade", env={"LOCAL_LLM_SOURCE": "/src/local-llm"}, path_tools=("uv",))
    assert result.returncode == 0, result.stderr
    assert "uv tool install --upgrade /src/local-llm" in result.stdout
    assert "astral.sh" not in result.stdout


def test_piped_invocation_never_executes_trailing_input():
    """`curl ... | sh` puts the script on stdin; nothing after it may run."""
    bindir = Path(os.environ.get("TMPDIR", "/tmp")) / f"install-sh-{os.getpid()}-piped"
    bindir.mkdir(parents=True, exist_ok=True)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(bindir), "LOCAL_LLM_INSTALL_DRY_RUN": "1"}
    result = subprocess.run(
        ["sh"], input=SCRIPT.read_text() + "echo LEAKED\n", capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "LEAKED" not in result.stdout
