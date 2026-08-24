#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo_root}/ops/jepa_wm_insertion_artifact_milestone.sh" world_model "$@"
