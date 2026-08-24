#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"

calibration_name="${1:-}"
reference_name="${2:-}"
exploration_seed="${3:-}"
trial_count="${4:-}"
artifacts_name="${5:-}"

for identifier in "${calibration_name}" "${reference_name}" "${artifacts_name}"; do
  is_safe_identifier "${identifier}" || {
    printf 'error: invalid control calibration identifier\n' >&2
    exit 1
  }
done
require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
require_positive_integer "trial count" "${trial_count}" || exit 1
(( trial_count >= 3 && trial_count <= 12 )) || {
  printf 'error: control calibration requires 3 to 12 trials\n' >&2
  exit 1
}

collection_id="calibration-$(date -u +%Y%m%dT%H%M%SZ)-${exploration_seed}"
sessions=""
for (( index = 0; index < trial_count; index++ )); do
  printf -v suffix '%02d' "${index}"
  session_id="${collection_id}-${suffix}"
  sessions+="${sessions:+,}${session_id}"
  bash "${repo_dir}/ops/run_control_step.sh" \
    "${session_id}" "${reference_name}" "${exploration_seed}" \
    "${artifacts_name}" deferred calibration_collection
done

bash "${repo_dir}/ops/jepa_wm.sh" control-objective-calibrate \
  --sessions "${sessions}" \
  --output "${calibration_name}"
printf 'Calibration collection: %s\nCalibration sessions: %s\n' \
  "${collection_id}" "${sessions}"
