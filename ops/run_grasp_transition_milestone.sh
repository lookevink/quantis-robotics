#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
run_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
grasp_identity="${4:-}"
transition_identity="${5:-}"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"

for identifier in \
  "${run_id}" "${reference_name}" "${grasp_identity}" \
  "${transition_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid grasp transition milestone identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1

cd "${repo_dir}"
grasp_rollout_id="${run_id}-grasp"
grasp_context_index="$(contact_grasp_initial_context \
  "${repo_dir}" "${venv_python}")"
grasp_steps="$(contact_grasp_maximum_actions \
  "${repo_dir}" "${venv_python}")"

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
  control-worker-start --artifacts "${transition_identity}"
bash "${repo_dir}/ops/run_grasp_transition_trial.sh" \
  "${run_id}-transition" "${grasp_final_session}" "${reference_name}" \
  "${exploration_seed}" "${transition_identity}"

printf 'Grasp transition milestone: %s\nGrasp report: %s\n' \
  "${run_id}" "${grasp_report}"
