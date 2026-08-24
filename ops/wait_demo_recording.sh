#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
recording_id="${1:-}"
timeout_seconds="${DEMO_RECORDING_TIMEOUT_SECONDS:-1200}"
job_file="${HOME}/docker/isaac-sim/data/quantis/recording_jobs/${recording_id}.json"

is_safe_identifier "${recording_id}" || {
  printf 'error: invalid recording ID: %s\n' "${recording_id}" >&2
  exit 1
}
[[ "${timeout_seconds}" =~ ^[0-9]+$ && "${timeout_seconds}" -gt 0 ]] || {
  printf 'error: invalid recording timeout: %s\n' "${timeout_seconds}" >&2
  exit 1
}

deadline=$((SECONDS + timeout_seconds))
while true; do
  if (( SECONDS >= deadline )); then
    printf 'error: recording job timed out: %s\n' "${recording_id}" >&2
    exit 1
  fi
  if [[ -f "${job_file}" ]]; then
    status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${job_file}")"
    if [[ "${status}" != "running" ]]; then
      break
    fi
  fi
  sleep 1
done

cat "${job_file}"
[[ "${status}" == "complete" ]] || exit 1
