#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
control_identity="${4:-}"
source_session_id="${5:-}"
context_index="${6:-43}"
insertion_rollout_profile="${7:-}"
policy="insertion_reset_trial"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
insertion_rollout_maximum_steps="$(insertion_rollout_profile_field \
  "${repo_dir}" "${venv_python}" "${insertion_rollout_profile}" maximum-steps)"

for identifier in \
  "${session_id}" "${reference_name}" "${control_identity}" "${source_session_id}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid insertion reset-trial identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "context index" "${context_index}" || exit 1

cd "${repo_dir}"
proposal_name="$(control_proposal_from_identity \
  "${policy}" "${control_identity}" "${checkpoint_dir}" "${venv_python}")"
run_reset_trial_control_session \
  "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
  "${proposal_name}" "${policy}" "${source_session_id}" "${context_index}" \
  "${isaac_control_capture_timeout_seconds}" prepare_insertion_trial_source persist_insertion_trial_response \
  "${insertion_rollout_maximum_steps}" \
  "${isaac_insertion_trial_apply_timeout_seconds}"
