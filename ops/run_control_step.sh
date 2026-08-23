#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
control_identity="${4:-}"
shadow_mode="${5:-immediate}"
policy="${6:-direct}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

is_safe_identifier "${session_id}" || {
  printf 'error: invalid control session\n' >&2
  exit 1
}
is_safe_identifier "${reference_name}" || {
  printf 'error: invalid reference recording\n' >&2
  exit 1
}
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
is_safe_identifier "${control_identity}" || {
  printf 'error: invalid control identity\n' >&2
  exit 1
}
[[ "${shadow_mode}" == "immediate" || "${shadow_mode}" == "deferred" ]] || {
  printf 'error: shadow mode must be immediate or deferred\n' >&2
  exit 1
}
validate_control_policy "${policy}" || exit 1
cd "${repo_dir}"
proposal_name="$(control_proposal_from_identity \
  "${policy}" "${control_identity}" "${checkpoint_dir}" "${venv_python}")"

isaac_server_call \
  "await demo.capture_control_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}','${policy}')" \
  180 true
respond_to_control_session "${repo_dir}" "${session_id}" "${policy}"
isaac_server_call "await demo.apply_control_response('${session_id}')" 180
if [[ "${shadow_mode}" == "immediate" && "${policy}" == "direct" ]]; then
  capture_shadow_control_evidence "${repo_dir}" "${session_id}"
fi
