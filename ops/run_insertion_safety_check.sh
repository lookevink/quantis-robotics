#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
control_identity="${4:-}"
context_index="$(resolve_insertion_context \
  "${5:-}" "${repo_dir}" "${venv_python}")"
insertion_rollout_profile="${6:-}"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
insertion_rollout_maximum_steps="$(insertion_rollout_profile_field \
  "${repo_dir}" "${venv_python}" "${insertion_rollout_profile}" maximum-steps)"

capture_and_respond_control_session \
  "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
  "${control_identity}" insertion_safety_evaluation "${context_index}" \
  "${checkpoint_dir}" "${venv_python}" "" \
  "${insertion_rollout_maximum_steps}"
isaac_server_call \
  "await demo.evaluate_direct_insertion_candidate('${session_id}')" 180
