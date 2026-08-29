#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
context_index="$(resolve_insertion_context \
  "${4:-}" "${repo_dir}" "${venv_python}")"
load_mode="${5:-attached}"

for identifier in "${session_id}" "${reference_name}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid insertion resolution identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "context index" "${context_index}" || exit 1
load_mode="$(control_resolution_profile_field \
  "${repo_dir}" python3 load "${load_mode}")" || {
  printf 'error: invalid insertion resolution load mode\n' >&2
  exit 1
}
measurement_timeout_seconds="$(control_resolution_profile_field \
  "${repo_dir}" python3 measurement-timeout)" || {
  printf 'error: invalid insertion resolution measurement timeout\n' >&2
  exit 1
}
require_positive_integer \
  "insertion resolution measurement timeout" \
  "${measurement_timeout_seconds}" || exit 1

cd "${repo_dir}"
start_and_wait_control_capture \
  "${session_id}" "${reference_name}" "${exploration_seed}" \
  control-resolution-measurement insertion_resolution_measurement \
  "${context_index}" None standard \
  "${isaac_control_capture_timeout_seconds}" true
isaac_server_call \
  "await demo.measure_insertion_control_resolution('${session_id}','${load_mode}')" \
  "${measurement_timeout_seconds}"
"${venv_python}" -m jepa_wm.control_resolution \
  "${HOME}/docker/isaac-sim/data/quantis/control_sessions/${session_id}/control_resolution.json"
