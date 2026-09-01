#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
source "${repo_dir}/ops/shell_helpers.sh"
python_bin="${HOME}/.venvs/quantis-jepa-wm/bin/python"
checkpoint_root="${HOME}/docker/jepa-wm/checkpoints"
data_root="${HOME}/docker/isaac-sim/data/quantis"
source_revision="${1:-}"
runtime_fingerprint="${2:-}"
run_id="unknown-start-e2e-v2-62605-grasp"
previous_session="unknown-start-live-action-v7-62605"
reference_name="contact-insertion-v10-drive-slow-2600-held-00"
reference_seed="12600"
proposal_name="contact-grasp-v10-drive-slow-2600_task12_h256_s3000"
worker_identity="contact-insertion-v10-unknown-start-shadow-canary-v5"
maximum_actions="51"

cd "${repo_dir}"
phase="claim"
"${python_bin}" -m jepa_wm.unknown_start_grasp_continuation claim \
  --checkpoint-root "${checkpoint_root}" \
  --data-root "${data_root}" \
  --source-revision "${source_revision}" \
  --runtime-fingerprint "${runtime_fingerprint}"

terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop >/dev/null 2>&1
  "${python_bin}" -m jepa_wm.unknown_start_grasp_continuation failure \
    --checkpoint-root "${checkpoint_root}" \
    --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

phase="worker_start"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-start \
  --artifacts "${worker_identity}"

sessions=""
status="applied"
grasp_ready="false"
for ((index=0; index<maximum_actions; index++)); do
  printf -v suffix '%02d' "${index}"
  session_id="${run_id}-${suffix}"
  phase="followup_capture_${suffix}"
  isaac_server_call \
    "await demo.capture_followup_observation('${session_id}','${previous_session}','${proposal_name}')" \
    120
  phase="followup_inference_${suffix}"
  respond_to_control_session "${repo_dir}" "${session_id}" direct
  phase="followup_apply_${suffix}"
  isaac_server_call "await demo.apply_control_response('${session_id}')" 180
  if [[ -z "${sessions}" ]]; then
    sessions="${session_id}"
  else
    sessions="${sessions},${session_id}"
  fi
  phase="followup_status_${suffix}"
  status="$(${python_bin} -m jepa_wm.control_rollout_cli status \
    --data-root "${data_root}" --session "${session_id}")"
  [[ "${status}" == "applied" ]] || break
  grasp_status="$(${python_bin} -m jepa_wm.control_rollout_cli \
    reach-and-grasp-status \
    --data-root "${data_root}" \
    --rollout-id "${run_id}" \
    --reference-recording "${reference_name}" \
    --seed "${reference_seed}" \
    --proposal "${checkpoint_root}/${proposal_name}.pth" \
    --sessions "${sessions}" \
    --requested-steps "${maximum_actions}")"
  if [[ "${grasp_status}" == "ready" ]]; then
    grasp_ready="true"
    break
  fi
  previous_session="${session_id}"
done

phase="report"
bash "${repo_dir}/ops/jepa_wm.sh" control-rollout-report \
  --rollout "${run_id}" \
  --reference "${reference_name}" \
  --seed "${reference_seed}" \
  --proposal "${proposal_name}" \
  --policy direct \
  --sessions "${sessions}" \
  --requested-steps "${maximum_actions}" \
  --predecessor-session "unknown-start-live-action-v7-62605"
[[ "${status}" == "applied" && "${grasp_ready}" == "true" ]] || {
  printf 'error: unknown-start grasp continuation did not reach retained grasp\n' >&2
  false
}

phase="evaluation"
"${python_bin}" -m jepa_wm.unknown_start_grasp_continuation evaluate \
  --checkpoint-root "${checkpoint_root}" --data-root "${data_root}"
phase="worker_stop"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
trap - ERR
