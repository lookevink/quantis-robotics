#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
checkpoint_dir="${HOME}/docker/jepa-wm/checkpoints"
control_root="${HOME}/docker/isaac-sim/data/quantis"
config="${PHYSICAL_SHADOW_CANARY_CONFIG:-${repo_dir}/.scratch/jepa-physical-shadow-canary-v1/experiment-config.json}"
deployed_revision="${1:-}"
mapfile -t frozen_contract < <(
  "${venv_python}" - "${config}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
for value in (
    config["schema"],
    config["session_id"],
    config["known_start"]["reference"],
    config["known_start"]["seed"],
    config["worker"]["name"],
    config["known_start"]["context_index"],
    config["output"],
    config.get("unknown_start", {}).get("recording_id", ""),
    config.get("unknown_start", {}).get("result_fingerprint", ""),
):
    print(value)
PY
)
(( ${#frozen_contract[@]} == 9 )) || {
  printf 'error: incomplete physical shadow canary contract\n' >&2
  exit 1
}
experiment_schema="${frozen_contract[0]}"
session_id="${frozen_contract[1]}"
reference_name="${frozen_contract[2]}"
exploration_seed="${frozen_contract[3]}"
control_identity="${frozen_contract[4]}"
context_index="${frozen_contract[5]}"
output="${frozen_contract[6]}"
reset_recording_id="${frozen_contract[7]}"
reset_result_fingerprint="${frozen_contract[8]}"
[[ "${deployed_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'error: invalid physical shadow canary deployment revision\n' >&2
  exit 1
}
worker_manifest="${checkpoint_dir}/${control_identity}.worker.json"
for identifier in "${session_id}" "${reference_name}" "${control_identity}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid physical shadow canary identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}"
require_positive_integer "context index" "${context_index}"
if [[ "${experiment_schema}" == "quantis.jepa_wm_physical_shadow_canary_experiment.v3" || "${experiment_schema}" == "quantis.jepa_wm_physical_shadow_canary_experiment.v4" ]]; then
  is_safe_identifier "${reset_recording_id}" || {
    printf 'error: invalid unknown-start reset recording ID\n' >&2
    exit 1
  }
  [[ "${reset_result_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || {
    printf 'error: invalid unknown-start reset result fingerprint\n' >&2
    exit 1
  }
fi

cd "${repo_dir}"
"${venv_python}" -m jepa_wm.physical_shadow_canary prepare-worker \
  --config "${config}" --output "${worker_manifest}" \
  --recording-root "${control_root}/recordings"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
bash "${repo_dir}/ops/jepa_wm.sh" \
  control-worker-start --artifacts "${control_identity}"

if [[ "${experiment_schema}" == "quantis.jepa_wm_physical_shadow_canary_experiment.v4" ]]; then
  isaac_server_call \
    "await demo.preflight_unknown_start_shadow('${reset_recording_id}','${reset_result_fingerprint}')" \
    180 true
fi

phase="claim"

terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  "${venv_python}" -m jepa_wm.physical_shadow_canary failure \
    --config "${config}" --session "${session_id}" \
    --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

"${venv_python}" -m jepa_wm.physical_shadow_canary claim \
  --config "${config}" --session "${session_id}"

phase="capture"
if [[ "${experiment_schema}" == "quantis.jepa_wm_physical_shadow_canary_experiment.v3" || "${experiment_schema}" == "quantis.jepa_wm_physical_shadow_canary_experiment.v4" ]]; then
  proposal_name="$(control_proposal_from_identity \
    direct "${control_identity}" "${checkpoint_dir}" "${venv_python}")"
  isaac_server_call \
    "await demo.capture_unknown_start_shadow_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}','${reset_recording_id}','${reset_result_fingerprint}')" \
    180 true
  respond_to_control_session "${repo_dir}" "${session_id}" direct
else
  capture_and_respond_control_session \
    "${repo_dir}" "${session_id}" "${reference_name}" "${exploration_seed}" \
    "${control_identity}" direct "${context_index}" \
    "${checkpoint_dir}" "${venv_python}" "" "" contact_grasp
fi

phase="shadow_planning"
bash "${repo_dir}/ops/jepa_wm.sh" \
  control-shadow-session --session "${session_id}"

phase="counterfactual_safety"
isaac_server_call \
  "await demo.evaluate_shadow_candidate('${session_id}')" 180

phase="terminal_evaluation"
"${venv_python}" -m jepa_wm.physical_shadow_canary evaluate \
  --config "${config}" \
  --session-path "${control_root}/control_sessions/${session_id}" \
  --output "${output}" \
  --deployed-revision "${deployed_revision}"
trap - ERR
