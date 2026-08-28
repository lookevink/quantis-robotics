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
first_safety_session_id="${run_id}-safety1"
first_action_session_id="${run_id}-action1"
second_safety_session_id="${run_id}-safety2"
second_action_session_id="${run_id}-action2"

for identifier in "${run_id}" "${reference_name}" "${control_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid insertion two-step identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "context index" "${context_index}" || exit 1

printf 'Insertion two-step safety 1: %s\n' "${first_safety_session_id}"
printf 'Insertion two-step action 1: %s\n' "${first_action_session_id}"
printf 'Insertion two-step safety 2: %s\n' "${second_safety_session_id}"
printf 'Insertion two-step action 2: %s\n' "${second_action_session_id}"

cd "${repo_dir}"
bash "${repo_dir}/ops/run_insertion_safety_check.sh" \
  "${first_safety_session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" "${context_index}" two-step
bash "${repo_dir}/ops/run_insertion_reset_trial.sh" \
  "${first_action_session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" "${first_safety_session_id}" "${context_index}" \
  two-step
isaac_server_call \
  "demo.verify_insertion_followup_source('${first_action_session_id}')" 180
bash "${repo_dir}/ops/run_insertion_followup_trial.sh" \
  "${second_safety_session_id}" "${second_action_session_id}" \
  "${first_action_session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}"
isaac_server_call \
  "demo.verify_insertion_two_step_result('${first_action_session_id}','${second_action_session_id}','${reference_name}',${exploration_seed})" \
  180
