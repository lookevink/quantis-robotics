#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
source "${repo_dir}/ops/shell_helpers.sh"
python_bin="${HOME}/.venvs/quantis-jepa-wm/bin/python"
checkpoint_root="${HOME}/docker/jepa-wm/checkpoints"
recovery_checkpoint_root="/mnt/quantis-assets/quantis-state/jepa-wm/checkpoints"
data_root="${HOME}/docker/isaac-sim/data/quantis"
recovery_data_root="/mnt/quantis-assets/quantis-state/isaac"
source_revision="${1:-}"
runtime_fingerprint="${2:-}"
run_id="unknown-start-e2e-v31-62605-grasp"
source_session="unknown-start-e2e-v30-62605-grasp-001"
reference_name="contact-insertion-v10-drive-slow-2600-held-00"
reference_seed="12600"
proposal_name="contact-grasp-acquisition-v10-drive-slow-2600_task12_h256_s3000_cfopen-v3-retained"
worker_identity="contact-insertion-v10-unknown-start-acquisition-v3-retained"
global_maximum_actions="$(contact_grasp_maximum_actions "${repo_dir}" "${python_bin}")"
source_horizon_actions="137"
maximum_actions="55"
first_session="${run_id}-001"
module="jepa_wm.contact_grasp_horizon_completion"

cd "${repo_dir}"
[[ "$((maximum_actions + source_horizon_actions))" == "${global_maximum_actions}" ]] || {
  printf 'error: frozen V31 cumulative action allocation changed\n' >&2
  exit 1
}
phase="claim"
"${python_bin}" -m "${module}" claim \
  --checkpoint-root "${checkpoint_root}" \
  --recovery-checkpoint-root "${recovery_checkpoint_root}" \
  --data-root "${data_root}" --recovery-data-root "${recovery_data_root}" \
  --followup-session "${first_session}" --source-revision "${source_revision}" \
  --runtime-fingerprint "${runtime_fingerprint}"

terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop >/dev/null 2>&1
  "${python_bin}" -m "${module}" failure --checkpoint-root "${checkpoint_root}" \
    --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

phase="encode_handoff"
encoded_handoff="$("${python_bin}" -m "${module}" encode \
  --checkpoint-root "${checkpoint_root}")"

phase="worker_start"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-start --artifacts "${worker_identity}"

sessions=""
previous_session="${source_session}"
status=""
grasp_ready="false"
for ((index=1; index<=maximum_actions; index++)); do
  printf -v suffix '%03d' "${index}"
  session_id="${run_id}-${suffix}"
  phase="capture_${suffix}"
  if (( index == 1 )); then
    isaac_server_call \
      "await demo.capture_contact_grasp_acquisition_handoff('${session_id}','${source_session}','${proposal_name}','${encoded_handoff}')" \
      180 true
    phase="handoff_validation"
    "${python_bin}" -m "${module}" validate-handoff \
      --checkpoint-root "${checkpoint_root}" --data-root "${data_root}"
  else
    isaac_server_call \
      "await demo.capture_followup_observation('${session_id}','${previous_session}','${proposal_name}')" 180
  fi
  phase="inference_${suffix}"
  respond_to_control_session "${repo_dir}" "${session_id}" direct
  phase="apply_${suffix}"
  isaac_server_call "await demo.apply_control_response('${session_id}')" 180
  sessions="${sessions:+${sessions},}${session_id}"
  phase="status_${suffix}"
  status="$("${python_bin}" -m jepa_wm.control_rollout_cli status \
    --data-root "${data_root}" --session "${session_id}")"
  [[ "${status}" == "applied" ]] || break
  grasp_status="$("${python_bin}" -m jepa_wm.control_rollout_cli reach-and-grasp-status \
    --data-root "${data_root}" --rollout-id "${run_id}" \
    --reference-recording "${reference_name}" --seed "${reference_seed}" \
    --proposal "${checkpoint_root}/${proposal_name}.pth" --sessions "${sessions}" \
    --requested-steps "${maximum_actions}" --predecessor-session "${source_session}")"
  if [[ "${grasp_status}" == "ready" ]]; then
    grasp_ready="true"
    break
  fi
  previous_session="${session_id}"
done

phase="report"
bash "${repo_dir}/ops/jepa_wm.sh" control-rollout-report \
  --rollout "${run_id}" --reference "${reference_name}" --seed "${reference_seed}" \
  --proposal "${proposal_name}" --policy direct --sessions "${sessions}" \
  --requested-steps "${maximum_actions}" --predecessor-session "${source_session}"
[[ "${status}" == "applied" && "${grasp_ready}" == "true" ]] || {
  printf 'error: horizon-complete acquisition did not retain grasp\n' >&2
  false
}

phase="evaluation"
"${python_bin}" -m "${module}" evaluate --checkpoint-root "${checkpoint_root}" \
  --data-root "${data_root}"
phase="worker_stop"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-stop
trap - ERR
