#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
proposal_name="${4:-}"
shadow_mode="${5:-immediate}"

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
[[ "${shadow_mode}" == "immediate" || "${shadow_mode}" == "deferred" ]] || {
  printf 'error: shadow mode must be immediate or deferred\n' >&2
  exit 1
}

cd "${repo_dir}"
isaac_server_call \
  "await demo.capture_control_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}')" \
  180 true
bash "${repo_dir}/ops/jepa_wm.sh" control-infer-session --session "${session_id}"
isaac_server_call "await demo.apply_control_response('${session_id}')" 180
if [[ "${shadow_mode}" == "immediate" ]]; then
  capture_shadow_control_evidence "${repo_dir}" "${session_id}"
fi
