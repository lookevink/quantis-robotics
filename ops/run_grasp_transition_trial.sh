#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
run_id="${1:-}"
previous_session="${2:-}"
reference_name="${3:-}"
exploration_seed="${4:-}"
insertion_identity="${5:-}"
rolled_back_session="${6:-}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

for identifier in \
  "${run_id}" "${previous_session}" "${reference_name}" "${insertion_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid grasp transition identifier\n' >&2
    exit 1
  }
done
if [[ -n "${rolled_back_session}" ]]; then
  is_safe_identifier "${rolled_back_session}" || {
    printf 'error: invalid rolled-back transition identifier\n' >&2
    exit 1
  }
fi
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1

cd "${repo_dir}"
if [[ -n "${rolled_back_session}" ]]; then
  isaac_server_call \
    "demo.restore_grasp_transition_retry('${previous_session}','${rolled_back_session}')" \
    180 true
fi
proposal_name="$(control_proposal_from_identity \
  insertion_followup_trial "${insertion_identity}" \
  "${checkpoint_dir}" "${venv_python}")"
safety_session="${run_id}-safety1"
action_session="${run_id}-action1"

run_insertion_followup_trial \
  "${repo_dir}" "${safety_session}" "${action_session}" \
  "${previous_session}" "${reference_name}" "${exploration_seed}" \
  "${proposal_name}" "${action_session}" 1 "${previous_session}"
report="${HOME}/docker/isaac-sim/data/quantis/control_rollouts/${action_session}/report.json"
require_control_rollout_applied "${venv_python}" "${report}"
printf 'Grasp transition trial: %s\nSafety session: %s\nAction session: %s\n' \
  "${run_id}" "${safety_session}" "${action_session}"
