#!/usr/bin/env bash
set -eEuo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
recording_id="${1:-unknown-start-reset-v1-62600}"
seed="${2:-62600}"
source_revision="${3:-}"
data_root="${HOME}/docker/isaac-sim/data/quantis"
python_bin="${HOME}/.venvs/quantis-jepa-wm/bin/python"
terminal_root="${data_root}/unknown_start_reset_claims"
claim_path="${terminal_root}/${recording_id}-claim.json"
failure_path="${terminal_root}/${recording_id}-failure.json"

is_safe_identifier "${recording_id}" || {
  printf 'error: invalid unknown-start reset recording id\n' >&2
  exit 1
}
require_nonnegative_integer "unknown-start reset seed" "${seed}" || exit 1
[[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || {
  printf 'error: invalid unknown-start reset source revision\n' >&2
  exit 1
}

cd "${repo_dir}"
"${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle claim \
  --path "${claim_path}" --recording-id "${recording_id}" --seed "${seed}" \
  --source-revision "${source_revision}"

phase="simulator_reset"
terminalize_failure() {
  local exit_status=$?
  trap - ERR
  set +e
  "${python_bin}" -m jepa_wm.unknown_start_reset_lifecycle failure \
    --path "${failure_path}" --claim-path "${claim_path}" \
    --error "${phase}:exit_${exit_status}" >&2
  exit "${exit_status}"
}
trap terminalize_failure ERR

isaac_server_call \
  "await demo.authenticate_unknown_start_reset('${recording_id}',${seed},'${source_revision}')" 300
trap - ERR
