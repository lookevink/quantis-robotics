#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
source "${repo_dir}/ops/shell_helpers.sh"
python_bin="${HOME}/.venvs/quantis-jepa-wm/bin/python"
checkpoint_root="${HOME}/docker/jepa-wm/checkpoints"
data_root="${HOME}/docker/isaac-sim/data/quantis"
source_revision="${1:-}"
runtime_fingerprint="${2:-}"
session_id="unknown-start-live-action-v1-62605"
source_session_id="unknown-start-shadow-canary-v5-62605"
reset_recording_id="unknown-start-reset-v6-62605"
reset_result_fingerprint="70a8fba8022e687c2fc9ecd78f8d63924a8a5840497af9249c60bb781a0a6d58"
source_shadow_fingerprint="75b77d011c314db3118993723755ea1524ec2eab22bd141d98543d1162475ce2"
source_safety_fingerprint="467a62dae7ed8728e536e81f3b6b58dad3aa3702b12477ab6cdeac52b7506bae"

cd "${repo_dir}"
phase="preflight"
isaac_server_call \
  "await demo.preflight_unknown_start_shadow('${reset_recording_id}','${reset_result_fingerprint}')" \
  180 true
phase="source"
isaac_server_call \
  "demo.prepare_experimental_candidate_source('${source_session_id}','${source_shadow_fingerprint}','${source_safety_fingerprint}')" 180

"${python_bin}" -m jepa_wm.unknown_start_live_action claim \
  --checkpoint-root "${checkpoint_root}" \
  --source-revision "${source_revision}" \
  --runtime-fingerprint "${runtime_fingerprint}"

phase="capture"
terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  "${python_bin}" -m jepa_wm.unknown_start_live_action failure \
    --checkpoint-root "${checkpoint_root}" --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

isaac_server_call \
  "await demo.capture_unknown_start_candidate_observation('${session_id}','contact-insertion-v10-drive-slow-2600-held-00',12600,'experimental_shadow_candidate','${reset_recording_id}','${reset_result_fingerprint}')" \
  180
phase="binding"
isaac_server_call \
  "demo.persist_experimental_candidate_response('${session_id}','${source_session_id}')" \
  180
phase="apply"
isaac_server_call "await demo.apply_control_response('${session_id}')" 180
phase="evaluation"
"${python_bin}" -m jepa_wm.unknown_start_live_action evaluate \
  --checkpoint-root "${checkpoint_root}" --data-root "${data_root}"
trap - ERR
