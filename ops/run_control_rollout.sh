#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
rollout_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
step_count="${4:-}"
control_identity="${5:-}"
policy="${6:-direct}"
context_index="${7:-4}"
context_purpose="${8:-standard}"
data_root="${HOME}/docker/isaac-sim/data/quantis"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

cd "${repo_dir}"

for value in "${rollout_id}" "${reference_name}" "${control_identity}"; do
  is_safe_identifier "${value}" || {
    printf 'error: invalid control rollout identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "step count" "${step_count}" || exit 1
maximum_step_count=8
if [[ "${context_purpose}" == "contact_grasp" ]]; then
  maximum_step_count="$(contact_grasp_maximum_actions \
    "${repo_dir}" "${venv_python}")"
fi
(( step_count <= maximum_step_count )) || {
  printf 'error: control rollout exceeds its task-specific step cap\n' >&2
  exit 1
}
validate_control_policy "${policy}" || exit 1
require_positive_integer "context index" "${context_index}" || exit 1
proposal_name="$(control_proposal_from_identity \
  "${policy}" "${control_identity}" \
  "${HOME}/docker/jepa-wm/checkpoints" "${venv_python}")"
expected_proposal="$(control_proposal_for_policy "${policy}" "${proposal_name}")"
[[ "${proposal_name}" == "${expected_proposal}" ]] || {
  printf 'error: proposal does not match control policy\n' >&2
  exit 1
}

step_status() {
  "${venv_python}" -m jepa_wm.control_rollout_cli status \
    --data-root "${data_root}" \
    --session "$1"
}

reach_and_grasp_status() {
  "${venv_python}" -m jepa_wm.control_rollout_cli reach-and-grasp-status \
    --data-root "${data_root}" \
    --rollout-id "${rollout_id}" \
    --reference-recording "${reference_name}" \
    --seed "${exploration_seed}" \
    --proposal "${HOME}/docker/jepa-wm/checkpoints/${proposal_name}.pth" \
    --sessions "${sessions}" \
    --requested-steps "${step_count}"
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
        --orchestration-failure "${current_phase}:exit_${command_status}"
      )
    fi
    set +e
    bash "${repo_dir}/ops/jepa_wm.sh" control-rollout-report \
      --rollout "${rollout_id}" \
      --reference "${reference_name}" \
      --seed "${exploration_seed}" \
      --proposal "${proposal_name}" \
      --policy "${policy}" \
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
  "${first_session}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" deferred "${policy}" "${context_index}" \
  "${context_purpose}"
previous_session="${first_session}"
current_phase="initial_status"
status="$(step_status "${first_session}")"
task_terminal=false
if [[ "${context_purpose}" == "contact_grasp" && "${status}" == "applied" ]]; then
  if [[ "$(reach_and_grasp_status)" == "ready" ]]; then
    task_terminal=true
  fi
fi

for (( index = 1; index < step_count; index++ )); do
  [[ "${status}" == "applied" && "${task_terminal}" == "false" ]] || break
  printf -v suffix '%02d' "${index}"
  session_id="${rollout_id}-${suffix}"
  sessions="${sessions},${session_id}"
  current_phase="followup_capture_${suffix}"
  isaac_server_call \
    "await demo.capture_followup_observation('${session_id}','${previous_session}','${proposal_name}')" \
    120
  current_phase="followup_inference_${suffix}"
  respond_to_control_session "${repo_dir}" "${session_id}" "${policy}"
  current_phase="followup_apply_${suffix}"
  isaac_server_call "await demo.apply_control_response('${session_id}')" 180
  previous_session="${session_id}"
  current_phase="followup_status_${suffix}"
  status="$(step_status "${session_id}")"
  if [[ "${context_purpose}" == "contact_grasp" && "${status}" == "applied" ]]; then
    if [[ "$(reach_and_grasp_status)" == "ready" ]]; then
      task_terminal=true
    fi
  fi
done
if [[ "${policy}" == "direct" ]]; then
  current_phase="shadow_evidence"
  shadow_roster="$(control_rollout_shadow_session_roster \
    "${context_purpose}" "${sessions}")"
  IFS=',' read -r -a shadow_sessions <<<"${shadow_roster}"
  for session_id in "${shadow_sessions[@]}"; do
    capture_shadow_control_evidence "${repo_dir}" "${session_id}"
  done
fi
current_phase="complete"
