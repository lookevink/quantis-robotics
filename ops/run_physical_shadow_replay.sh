#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
venv_python="${HOME}/.venvs/quantis-jepa-wm/bin/python"
control_root="${HOME}/docker/isaac-sim/data/quantis"
checkpoint_root="${HOME}/docker/jepa-wm/checkpoints"
config="${repo_dir}/.scratch/jepa-physical-shadow-replay-v1/experiment-config.json"
phase="claim"

mapfile -t replay_contract < <(
  "${venv_python}" - "${config}" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
print(config["source"]["session"])
print(config["worker"]["name"])
print(config["worker"]["manifest"])
print(config["output"])
PY
)
(( ${#replay_contract[@]} == 4 )) || exit 1
source_session="${replay_contract[0]}"
worker_name="${replay_contract[1]}"
worker_manifest="${replay_contract[2]}"
output="${replay_contract[3]}"

terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  cd "${repo_dir}"
  "${venv_python}" -m jepa_wm.physical_shadow_replay failure \
    --config "${config}" --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

cd "${repo_dir}"
"${venv_python}" -m jepa_wm.physical_shadow_replay claim --config "${config}"

phase="model_replay"
bash "${repo_dir}/ops/jepa_wm.sh" control-worker-start --artifacts "${worker_name}"
"${venv_python}" -m jepa_wm.control_client \
  --socket "${HOME}/docker/jepa-wm/run/control.sock" \
  --request "${source_session}/request.json" \
  --state "${source_session}/state.json" \
  --recording-root "${control_root}" \
  --direct-response "${source_session}/response.json" \
  --artifacts "${worker_manifest}" \
  --shadow-request-output "${output}/shadow_request.json" \
  --shadow-response-output "${output}/shadow.json"

phase="adjudication"
"${venv_python}" -m jepa_wm.physical_shadow_replay evaluate --config "${config}"
trap - ERR
