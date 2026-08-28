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
context_index="${7:-4}"
context_purpose="${8:-standard}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

[[ "${shadow_mode}" == "immediate" || "${shadow_mode}" == "deferred" ]] || {
  printf 'error: shadow mode must be immediate or deferred\n' >&2
  exit 1
}
capture_and_respond_control_session \
  "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" "${policy}" "${context_index}" \
  "${checkpoint_dir}" "${venv_python}" "" "" "${context_purpose}"
isaac_server_call "await demo.apply_control_response('${session_id}')" 180
if [[ "${shadow_mode}" == "immediate" && "${policy}" == "direct" ]]; then
  capture_shadow_control_evidence "${repo_dir}" "${session_id}"
fi
