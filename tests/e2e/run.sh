#!/bin/bash
# Build the clean-machine image and run the end-to-end flow. Needs Docker and network.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE=local-llm-e2e
docker build -f tests/e2e/Dockerfile -t "$IMAGE" .
# Forward the knobs a CI job or a throttled network may need.
args=()
for var in E2E_MODEL E2E_QUANT HF_TOKEN; do
  if [ -n "${!var:-}" ]; then args+=(-e "$var=${!var}"); fi
done
docker run --rm ${args[@]+"${args[@]}"} "$IMAGE"
