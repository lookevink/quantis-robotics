#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
run_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
control_identity="${4:-}"
context_index="$(resolve_insertion_context \
  "${5:-}" "${repo_dir}" "${venv_python}")"

for identifier in "${run_id}" "${reference_name}" "${control_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid insertion demo rollout identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "context index" "${context_index}" || exit 1
maximum_steps="$(insertion_rollout_profile_field \
  "${repo_dir}" "${venv_python}" demo maximum-steps)"
require_positive_integer "maximum rollout steps" "${maximum_steps}" || exit 1

declare -a safety_sessions=()
declare -a action_sessions=()
for (( step=1; step<=maximum_steps; step++ )); do
  safety_sessions+=("${run_id}-safety${step}")
  action_sessions+=("${run_id}-action${step}")
  printf 'Insertion demo safety %d: %s\n' \
    "${step}" "${safety_sessions[step-1]}"
  printf 'Insertion demo action %d: %s\n' \
    "${step}" "${action_sessions[step-1]}"
done

cd "${repo_dir}"
bash "${repo_dir}/ops/run_insertion_safety_check.sh" \
  "${safety_sessions[0]}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" "${context_index}" demo
bash "${repo_dir}/ops/run_insertion_reset_trial.sh" \
  "${action_sessions[0]}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" "${safety_sessions[0]}" "${context_index}" \
  demo

for (( step=2; step<=maximum_steps; step++ )); do
  previous_session_id="${action_sessions[step-2]}"
  prefix_sessions=("${action_sessions[@]:0:step}")
  prefix_roster="$(IFS=,; printf '%s' "${prefix_sessions[*]}")"
  isaac_server_call \
    "demo.verify_insertion_followup_source('${previous_session_id}')" 180
  bash "${repo_dir}/ops/run_insertion_followup_trial.sh" \
    "${safety_sessions[step-1]}" "${action_sessions[step-1]}" \
    "${previous_session_id}" "${reference_name}" "${exploration_seed}" \
    "${control_identity}" "${prefix_roster}" "${step}"
done

session_roster="$(IFS=,; printf '%s' "${action_sessions[*]}")"
isaac_server_call \
  "demo.verify_insertion_demo_rollout_result('${session_roster}','${reference_name}',${exploration_seed})" \
  180
