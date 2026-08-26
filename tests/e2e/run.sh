#!/bin/bash
# Build the clean-machine image and run the end-to-end flow. Needs Docker and network.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMAGE=local-llm-e2e
docker build -f tests/e2e/Dockerfile -t "$IMAGE" .
docker run --rm "$IMAGE"
