#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
rollout_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
step_count="${4:-}"
proposal_name="${5:-}"
data_root="${HOME}/docker/isaac-sim/data/quantis"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

cd "${repo_dir}"

for value in "${rollout_id}" "${reference_name}" "${proposal_name}"; do
  is_safe_identifier "${value}" || {
    printf 'error: invalid control rollout identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "step count" "${step_count}" || exit 1
(( step_count <= 8 )) || {
  printf 'error: control rollout is capped at eight steps\n' >&2
  exit 1
}

step_status() {
  "${venv_python}" -m jepa_wm.control_rollout_cli status \
    --data-root "${data_root}" \
    --session "$1"
}

sessions=""
current_phase="initialization"

finalize_rollout() {
  local command_status=$?
  local report_status=0
  local -a error_arguments=()
  trap - EXIT
  if [[ -n "${sessions}" ]]; then
    if (( command_status != 0 )); then
      error_arguments=(
        --orchestration-error "${current_phase}:exit_${command_status}"
      )
    fi
    set +e
    bash "${repo_dir}/ops/jepa_wm.sh" control-rollout-report \
      --rollout "${rollout_id}" \
      --reference "${reference_name}" \
      --seed "${exploration_seed}" \
      --proposal "${proposal_name}" \
      --sessions "${sessions}" \
      --requested-steps "${step_count}" \
      "${error_arguments[@]}"
    report_status=$?
    set -e
  fi
  if (( command_status == 0 && report_status != 0 )); then
    command_status=${report_status}
  fi
  exit "${command_status}"
}

trap finalize_rollout EXIT

first_session="${rollout_id}-00"
sessions="${first_session}"
current_phase="initial_control_step"
bash "${repo_dir}/ops/run_control_step.sh" \
  "${first_session}" "${reference_name}" "${exploration_seed}" "${proposal_name}"
previous_session="${first_session}"
current_phase="initial_status"
status="$(step_status "${first_session}")"

for (( index = 1; index < step_count; index++ )); do
  [[ "${status}" == "applied" ]] || break
  printf -v suffix '%02d' "${index}"
  session_id="${rollout_id}-${suffix}"
  sessions="${sessions},${session_id}"
  current_phase="followup_capture_${suffix}"
  isaac_server_call \
    "await demo.capture_followup_observation('${session_id}','${previous_session}','${proposal_name}')" \
    120
  current_phase="followup_inference_${suffix}"
  bash "${repo_dir}/ops/jepa_wm.sh" control-infer-session --session "${session_id}"
  current_phase="followup_apply_${suffix}"
  isaac_server_call "await demo.apply_control_response('${session_id}')" 180
  previous_session="${session_id}"
  current_phase="followup_status_${suffix}"
  status="$(step_status "${session_id}")"
done
current_phase="complete"
