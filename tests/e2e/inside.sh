#!/bin/bash
# Runs inside the container as the "tester" user. Every step must pass.
set -euo pipefail
# Any small GGUF repo works; override when a CDN edge throttles the default.
MODEL="${E2E_MODEL:-Qwen/Qwen2.5-0.5B-Instruct-GGUF}"
QUANT="${E2E_QUANT:-Q4_K_M}"
SECTION="${MODEL}:${QUANT}"

step() { printf '\n=== %s ===\n' "$*"; }

step "version"; local-llm --version
step "doctor"; local-llm doctor || true      # warns about brew/token on Linux; must not crash
local-llm doctor --json | jq -e '.[] | select(.name=="llama-server") | .status == "ok"' >/dev/null
step "recommend"; local-llm recommend --use small --limit 2
step "pull"; local-llm pull "${MODEL}:${QUANT}" --yes
grep -q "^\[${SECTION}\]" "$HOME/.config/local-llm/models.ini"
step "up"; local-llm up
step "load"; local-llm load "$SECTION"
step "chat"
reply=$(curl -sS --max-time 300 http://127.0.0.1:5678/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SECTION}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}")
echo "$reply" | jq -e '.choices[0].message.content | length > 0' >/dev/null
echo "$reply" | jq -c '.choices[0].message.content'
step "status"; local-llm status | tee /tmp/status.txt; grep -q "state:    running" /tmp/status.txt
step "completion"; local-llm completion install --shell bash --yes; grep -q "local-llm" "$HOME/.bashrc"
step "down"; local-llm down
if pgrep -f llama-server >/dev/null; then echo "llama-server still running" >&2; exit 1; fi
step "remove"; local-llm remove "$SECTION" --delete-files --yes
step "ALL PASSED"
