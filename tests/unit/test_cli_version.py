from typer.testing import CliRunner

from local_llm import __version__
from local_llm.cli import app


def test_version_flag_prints_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"local-llm {__version__}"
