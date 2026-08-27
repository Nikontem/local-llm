#!/bin/sh
# Install local-llm: one llama.cpp router serving every model in models.ini,
# plus a guided setup.
#
#   curl -fsSL https://raw.githubusercontent.com/Nikontem/local-llm/main/install.sh | sh
#   sh install.sh --upgrade     # reinstall the latest version
#
# The tool is installed with uv (https://astral.sh/uv), which is installed
# first if it is missing. Homebrew is not required for the tool itself: when
# it is present, `local-llm setup` installs llama.cpp and hf through it; when
# it is not, setup shows the routes (Homebrew, release binaries, a build).
#
# Environment:
#   LOCAL_LLM_SOURCE            what uv installs (default: git+https://github.com/Nikontem/local-llm)
#   LOCAL_LLM_INSTALL_DRY_RUN   set to 1 to print the commands instead of running them
set -eu

SOURCE="${LOCAL_LLM_SOURCE:-git+https://github.com/Nikontem/local-llm}"
DRY="${LOCAL_LLM_INSTALL_DRY_RUN:-0}"
UPGRADE=0

usage() {
  cat <<EOF
Usage: install.sh [--upgrade]

  --upgrade    reinstall the latest version
  --help       this text

Installs local-llm with uv (installing uv first if needed). Homebrew, when
present, is used later by \`local-llm setup\` to install llama.cpp and hf.
EOF
}

say() { printf '%s\n' "$*"; }
run() {
  say "+ $*"
  if [ "$DRY" != "1" ]; then
    "$@"
  fi
}
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --upgrade) UPGRADE=1 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; say "install.sh: unknown option '$1'" >&2; exit 2 ;;
  esac
  shift
done

OS="$(uname -s)"
case "$OS" in
  Darwin|Linux) ;;
  *) say "install.sh: $OS is untested; continuing anyway." >&2 ;;
esac

if have brew; then
  say "Homebrew found: local-llm setup will install llama.cpp and hf through it."
else
  say "Homebrew not found. That is fine: local-llm setup will show how to get llama.cpp"
  say "  (Homebrew from https://brew.sh is one route, release binaries or a build are others)."
fi

if ! have uv; then
  say "uv is not installed; installing it (https://astral.sh/uv)."
  if [ "$DRY" = "1" ]; then
    say "+ curl -LsSf https://astral.sh/uv/install.sh | sh"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin; make it visible to this script.
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
fi
if [ "$UPGRADE" = "1" ]; then
  run uv tool install --upgrade "$SOURCE"
else
  run uv tool install "$SOURCE"
fi
run uv tool update-shell

say ""
say "Installed. Next:"
say "  local-llm setup     guided first run (checks tools, picks models for this machine, wires agents)"
say "  local-llm --help    every command"

# Stop here: under "curl ... | sh" anything after this line would be executed as commands.
exit 0
