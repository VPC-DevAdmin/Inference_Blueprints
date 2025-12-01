#!/usr/bin/env bash
set -euo pipefail

REGISTRY="vectorpathconsulting"

# Find all docker compose files one or two levels deep
find . \
  -type f \
  \( -name "docker-compose.yml" -o -name "docker-compose.yaml" \) | while read -r compose; do

  project_dir="$(basename "$(dirname "$compose")")"

  echo "Processing $compose (project: $project_dir)"

  # Get list of service names
  services=$(yq e '.services | keys | .[]' "$compose")

  for svc in $services; do
    image="${REGISTRY}/${project_dir}-${svc}:latest"
    echo "  setting image for service '$svc' -> $image"

    # Write image field into the service
    yq e -i ".services.\"$svc\".image = \"$image\"" "$compose"
  done

done

