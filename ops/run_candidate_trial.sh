#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
session_id="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
source_session_id="${4:-}"
policy="reset_trial_candidate"
proposal_name="$(control_proposal_for_policy "${policy}")"

for identifier in "${session_id}" "${reference_name}" "${source_session_id}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid candidate trial identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1

cd "${repo_dir}"
run_reset_trial_control_session \
  "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
  "${proposal_name}" "${policy}" "${source_session_id}" 4 \
  180 prepare_experimental_candidate_source persist_experimental_candidate_response
