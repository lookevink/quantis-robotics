#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
cd "${repo_dir}"
recording_id="${1:-}"
seed="${2:-}"
source_revision="${3:-}"
runtime_source_fingerprint="${4:-}"
data_root="${HOME}/docker/isaac-sim/data/quantis"
python_bin="${HOME}/.venvs/quantis-jepa-wm/bin/python"
descriptor_recording_id="$(
  "${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle \
    describe --field recording-id
)"
descriptor_seed="$(
  "${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle describe --field seed
)"
ledger_name="$(
  "${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle \
    describe --field ledger-name
)"
recording_id="${recording_id:-${descriptor_recording_id}}"
seed="${seed:-${descriptor_seed}}"
terminal_root="${data_root}/${ledger_name}"

is_safe_identifier "${recording_id}" || {
  printf 'error: invalid unknown-start reset recording id\n' >&2
  exit 1
}
require_nonnegative_integer "unknown-start reset seed" "${seed}" || exit 1
[[ "${recording_id}" == "${descriptor_recording_id}" \
  && "${seed}" == "${descriptor_seed}" ]] || {
  printf 'error: unknown-start reset run does not match its descriptor\n' >&2
  exit 1
}
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'error: invalid unknown-start reset source revision\n' >&2
  exit 1
}
[[ "${runtime_source_fingerprint}" =~ ^[0-9a-f]{64}$ ]] || {
  printf 'error: invalid unknown-start reset runtime fingerprint\n' >&2
  exit 1
}

"${python_bin}" -m jepa_wm.unknown_start_reset_runtime authenticate \
  --expected "${runtime_source_fingerprint}"
if ! mkdir -p "${terminal_root}" 2>/dev/null \
  || [[ ! -w "${terminal_root}" ]]; then
  sudo install -d -o "$(id -u)" -g "$(id -g)" "${terminal_root}"
fi
"${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle claim \
  --ledger-root "${terminal_root}" --recording-id "${recording_id}" \
  --seed "${seed}" --source-revision "${source_revision}" \
  --runtime-source-fingerprint "${runtime_source_fingerprint}"

phase="simulator_reset"
terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  "${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle failure \
    --ledger-root "${terminal_root}" --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

isaac_server_call \
  "await demo.authenticate_unknown_start_reset('${recording_id}',${seed},'${source_revision}','${runtime_source_fingerprint}')" \
  300 true
phase="artifact_ownership"
sudo chown -R "$(id -u):$(id -g)" \
  "${data_root}/recordings/${recording_id}" "${terminal_root}"
trap - ERR
