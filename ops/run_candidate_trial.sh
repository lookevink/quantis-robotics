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

current_phase="candidate_capture"
report_started=true

finalize_candidate_trial() {
  local command_status=$?
  local report_status=0
  local -a error_arguments=()
  trap - EXIT
  if [[ "${report_started}" == "true" ]]; then
    if (( command_status != 0 )); then
      error_arguments=(
        --orchestration-failure "${current_phase}:exit_${command_status}"
      )
    fi
    set +e
    bash "${repo_dir}/ops/jepa_wm.sh" control-rollout-report \
      --rollout "${session_id}" \
      --reference "${reference_name}" \
      --seed "${exploration_seed}" \
      --proposal "${proposal_name}" \
      --policy "${policy}" \
      --sessions "${session_id}" \
      --requested-steps 1 \
      "${error_arguments[@]}"
    report_status=$?
    set -e
  fi
  if (( command_status == 0 && report_status != 0 )); then
    command_status=${report_status}
  fi
  exit "${command_status}"
}

trap finalize_candidate_trial EXIT
cd "${repo_dir}"
isaac_server_call \
  "await demo.capture_control_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}','${policy}')" \
  180 true
current_phase="candidate_binding"
respond_to_control_session \
  "${repo_dir}" "${session_id}" "${policy}" "${source_session_id}"
current_phase="candidate_apply"
isaac_server_call "await demo.apply_control_response('${session_id}')" 180
current_phase="complete"
