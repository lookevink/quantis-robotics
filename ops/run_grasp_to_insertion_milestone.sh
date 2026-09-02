#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
run_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
grasp_identity="${4:-}"
insertion_identity="${5:-}"
demo_spec_id="${6:-}"
demo_spec_fingerprint="${7:-}"
source_revision="${8:-}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
data_root="${HOME}/docker/isaac-sim/data/quantis"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

for identifier in \
  "${run_id}" "${reference_name}" "${grasp_identity}" "${insertion_identity}" \
  "${demo_spec_id}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid grasp-to-insertion identifier\n' >&2
    exit 1
  }
done
[[ "${demo_spec_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'error: invalid frozen demo run fingerprint\n' >&2
  exit 1
}
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'error: invalid deployed source revision\n' >&2
  exit 1
}
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1

grasp_rollout_id="${run_id}-grasp"
grasp_context_index="$(contact_grasp_initial_context \
  "${repo_dir}" "${venv_python}")"
grasp_steps="$(contact_grasp_maximum_actions \
  "${repo_dir}" "${venv_python}")"
insertion_steps="$(insertion_rollout_profile_field \
  "${repo_dir}" "${venv_python}" grasp-to-insertion maximum-steps)"
cd "${repo_dir}"
grasp_proposal="$(control_proposal_from_identity \
  direct "${grasp_identity}" "${checkpoint_dir}" "${venv_python}")"
insertion_proposal="$(control_proposal_from_identity \
  insertion_followup_trial "${insertion_identity}" \
  "${checkpoint_dir}" "${venv_python}")"
validate_demo_run_spec \
  "${source_revision}" "${venv_python}" \
  "${data_root}/demo_runs/${demo_spec_id}.json" \
  "${demo_spec_fingerprint}" "${data_root}/recordings" \
  "${data_root}/scenes/datacenter_demo.usda" \
  "${grasp_identity}" "${checkpoint_dir}/${grasp_identity}.worker.json" \
  "${insertion_identity}" \
  "${checkpoint_dir}/${insertion_identity}.worker.json" \
  "${reference_name}" "${exploration_seed}" \
  "${run_id}" "${data_root}/demo_runs/${run_id}.binding.json" \
  "${grasp_steps}" "${insertion_steps}"
bash "${repo_dir}/ops/isaac_container.sh" \
  checkpoint-readable "${grasp_proposal}"
bash "${repo_dir}/ops/isaac_container.sh" \
  checkpoint-readable "${insertion_proposal}"

cleanup_control_worker() {
  local exit_status=$?
  local cleanup_status=0
  trap - EXIT
  bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop \
    || cleanup_status=$?
  if (( exit_status == 0 && cleanup_status != 0 )); then
    exit_status=${cleanup_status}
  fi
  exit "${exit_status}"
}
trap cleanup_control_worker EXIT

bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
bash "${repo_dir}/ops/jepa_wm.sh" \
  control-worker-start --artifacts "${grasp_identity}"
bash "${repo_dir}/ops/run_control_rollout.sh" \
  "${grasp_rollout_id}" "${reference_name}" "${exploration_seed}" \
  "${grasp_steps}" "${grasp_identity}" direct "${grasp_context_index}" \
  contact_grasp
grasp_report="${HOME}/docker/isaac-sim/data/quantis/control_rollouts/${grasp_rollout_id}/report.json"
require_control_rollout_reach_and_grasp "${venv_python}" "${grasp_report}"
grasp_final_session="$(control_rollout_terminal_session \
  "${venv_python}" "${grasp_report}")"
isaac_server_call \
  "demo.verify_grasp_to_insertion_source('${grasp_final_session}')" 180 true

bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
bash "${repo_dir}/ops/jepa_wm.sh" \
  control-worker-start --artifacts "${insertion_identity}"
declare -a safety_sessions=()
declare -a action_sessions=()
for ((step=1; step<=insertion_steps; step++)); do
  safety_sessions+=("${run_id}-safety${step}")
  action_sessions+=("${run_id}-action${step}")
done

previous_session="${grasp_final_session}"
for ((step=1; step<=insertion_steps; step++)); do
  prefix_sessions=("${action_sessions[@]:0:step}")
  prefix_roster="$(IFS=,; printf '%s' "${prefix_sessions[*]}")"
  if (( step > 1 )); then
    isaac_server_call \
      "demo.verify_insertion_followup_source('${previous_session}')" 180
  fi
  run_insertion_followup_trial \
    "${repo_dir}" "${safety_sessions[step-1]}" "${action_sessions[step-1]}" \
    "${previous_session}" "${reference_name}" "${exploration_seed}" \
    "${insertion_proposal}" "${prefix_roster}" "${insertion_steps}" \
    "${grasp_final_session}" false "" "${insertion_steps}"
  previous_session="${action_sessions[step-1]}"
done

session_roster="$(IFS=,; printf '%s' "${action_sessions[*]}")"
isaac_server_call \
  "demo.verify_grasp_to_insertion_result('${run_id}','${grasp_rollout_id}','${session_roster}','${reference_name}',${exploration_seed})" \
  180
