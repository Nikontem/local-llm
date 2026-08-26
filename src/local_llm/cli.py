"""Command-line entry point."""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    help="One llama.cpp router serving every model in models.ini.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        typer.echo(f"local-llm {__version__}")
        raise typer.Exit()
