#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Scanning for docker compose files under $ROOT"
echo

find "$ROOT" \
  -maxdepth 2 \
  -type f \
  \( -name "docker-compose.yml" -o -name "docker-compose.yaml" \) |
while read -r compose_file; do
  project_dir="$(dirname "$compose_file")"
  compose_name="$(basename "$compose_file")"

  echo "===================================="
  echo "Project: $project_dir"
  echo "Compose: $compose_name"
  echo "===================================="

  (
    cd "$project_dir"
    echo "Running: docker compose build"
    docker compose build
    echo "Running: docker compose push"
    docker compose push
  )

  echo
done

echo "All compose projects processed."

