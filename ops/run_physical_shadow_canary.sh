#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
control_root="${HOME}/docker/isaac-sim/data/quantis"
config="${repo_dir}/.scratch/jepa-physical-shadow-canary-v1/experiment-config.json"
session_id="${1:-}"
reference_name="contact-insertion-v10-drive-slow-2600-held-01"
exploration_seed="12601"
control_identity="contact-insertion-v10-physical-shadow-canary-v1"
context_index="110"
output="${checkpoint_dir}/quantis_physical_state_residual_v1/known-start-shadow-canary-v1.json"
phase="claim"

is_safe_identifier "${session_id}" || {
  printf 'error: invalid physical shadow canary session\n' >&2
  exit 1
}

terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  "${venv_python}" -m jepa_wm.physical_shadow_canary failure \
    --session "${session_id}" --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

cd "${repo_dir}"
"${venv_python}" -m jepa_wm.physical_shadow_canary claim \
  --config "${config}" --session "${session_id}"

phase="capture"
capture_and_respond_control_session \
  "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" direct "${context_index}" \
  "${checkpoint_dir}" "${venv_python}" "" "" contact_grasp

phase="shadow_planning"
bash "${repo_dir}/ops/jepa_wm.sh" \
  control-shadow-session --session "${session_id}"

phase="counterfactual_safety"
isaac_server_call \
  "await demo.evaluate_shadow_candidate('${session_id}')" 180

phase="terminal_evaluation"
evaluator_revision="$("${venv_python}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["evaluator"]["implementation_revision"])' \
  "${config}")"
"${venv_python}" -m jepa_wm.physical_shadow_canary evaluate \
  --config "${config}" \
  --session-path "${control_root}/control_sessions/${session_id}" \
  --output "${output}" \
  --evaluator-revision "${evaluator_revision}"
trap - ERR
