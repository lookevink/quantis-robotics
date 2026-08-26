#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
safety_session_id="${1:-}"
execution_session_id="${2:-}"
previous_session_id="${3:-}"
reference_name="${4:-}"
exploration_seed="${5:-}"
control_identity="${6:-}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

for identifier in \
  "${safety_session_id}" "${execution_session_id}" "${previous_session_id}" \
  "${reference_name}" "${control_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid insertion follow-up identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1

cd "${repo_dir}"
proposal_name="$(control_proposal_from_identity \
  insertion_followup_trial "${control_identity}" \
  "${checkpoint_dir}" "${venv_python}")"
run_insertion_followup_trial \
  "${repo_dir}" "${safety_session_id}" "${execution_session_id}" \
  "${previous_session_id}" "${reference_name}" "${exploration_seed}" \
  "${proposal_name}"
