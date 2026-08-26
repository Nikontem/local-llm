#!/bin/sh
# Install local-llm: one llama.cpp router serving every model in models.ini,
# plus a guided setup. Homebrew when it is there, otherwise uv.
#
#   curl -fsSL https://raw.githubusercontent.com/Nikontem/local-llm/main/install.sh | sh
#   sh install.sh --uv          # skip Homebrew even if installed
#   sh install.sh --brew        # insist on Homebrew
#   sh install.sh --upgrade     # reinstall the latest version
#
# Environment:
#   LOCAL_LLM_SOURCE            what uv installs (default: git+https://github.com/Nikontem/local-llm)
#   LOCAL_LLM_INSTALL_DRY_RUN   set to 1 to print the commands instead of running them
set -eu

SOURCE="${LOCAL_LLM_SOURCE:-git+https://github.com/Nikontem/local-llm}"
TAP="nikontem/tap"
DRY="${LOCAL_LLM_INSTALL_DRY_RUN:-0}"
MODE=""
YES=0
UPGRADE=0

usage() {
  cat <<EOF
Usage: install.sh [--brew | --uv] [-y] [--upgrade]

  --brew       install with Homebrew (brew tap $TAP && brew install local-llm)
  --uv         install with uv (uv tool install $SOURCE); installs uv first if needed
  -y, --yes    do not ask which one; Homebrew when present, uv otherwise
  --upgrade    reinstall the latest version
  --help       this text

Without --brew or --uv the script asks when both are possible.
EOF
}

say() { printf '%s\n' "$*"; }
run() {
  if [ "$DRY" = "1" ]; then
    say "+ $*"
  else
    say "+ $*"
    "$@"
  fi
}
have() { command -v "$1" >/dev/null 2>&1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --brew) MODE=brew ;;
    --uv) MODE=uv ;;
    -y|--yes) YES=1 ;;
    --upgrade) UPGRADE=1 ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; say "install.sh: unknown option '$1'" >&2; exit 2 ;;
  esac
  shift
done

OS="$(uname -s)"
case "$OS" in
  Darwin|Linux) ;;
  *) say "install.sh: $OS is untested; continuing with the uv path." >&2; MODE="${MODE:-uv}" ;;
esac

if [ -z "$MODE" ]; then
  if have brew; then
    if [ "$YES" = "1" ]; then
      MODE=brew
    else
      # Under "curl ... | sh" the script itself is on stdin, so ask on the terminal
      # when there is one and take the default otherwise.
      answer=brew
      if [ -t 0 ]; then
        printf 'Homebrew found. Install with Homebrew (recommended, brew manages updates) or uv? [brew/uv] '
        read -r answer || answer=brew
      elif ( : < /dev/tty ) 2>/dev/null; then
        printf 'Homebrew found. Install with Homebrew (recommended, brew manages updates) or uv? [brew/uv] ' > /dev/tty 2>/dev/null || true
        read -r answer < /dev/tty || answer=brew
      fi
      case "$answer" in
        uv|UV) MODE=uv ;;
        *) MODE=brew ;;
      esac
    fi
  else
    MODE=uv
  fi
fi

if [ "$MODE" = "brew" ]; then
  if ! have brew; then
    say "install.sh: --brew given but Homebrew is not installed." >&2
    say "  Install it from https://brew.sh, or run this script with --uv." >&2
    exit 1
  fi
  run brew tap "$TAP"
  if [ "$UPGRADE" = "1" ]; then
    run brew upgrade local-llm || run brew install local-llm
  else
    run brew install local-llm
  fi
else
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
fi

say ""
say "Installed. Next:"
say "  local-llm setup     guided first run (checks tools, picks models for this machine, wires agents)"
say "  local-llm --help    every command"

# Stop here: under "curl ... | sh" anything after this line would be executed as commands.
exit 0
