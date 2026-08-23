#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
proposal_name="${4:-}"

is_safe_identifier "${session_id}" || {
  printf 'error: invalid control session\n' >&2
  exit 1
}
is_safe_identifier "${reference_name}" || {
  printf 'error: invalid reference recording\n' >&2
  exit 1
}
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
is_safe_identifier "${proposal_name}" || {
  printf 'error: invalid proposal name\n' >&2
  exit 1
}

isaac_call() {
  local expression="$1"
  local timeout_seconds="$2"
  local code
  local response
  code="$(isaac_demo_code "${expression}")"
  response="$(printf '%s\n' "${code}" \
    | timeout "${timeout_seconds}" nc -N 127.0.0.1 8226)"
  print_checked_isaac_response "${response}"
}

cd "${repo_dir}"
isaac_call \
  "await demo.capture_control_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}')" \
  180
bash "${repo_dir}/ops/jepa_wm.sh" control-infer-session --session "${session_id}"
isaac_call "await demo.apply_control_response('${session_id}')" 180
