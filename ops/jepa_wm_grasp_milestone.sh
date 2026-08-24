#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"
aws_workflow="${repo_root}/ops/aws.sh"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

training_count="${1:-12}"
held_out_count="${2:-2}"
training_steps="${3:-3000}"
base_seed="${4:-2400}"
require_positive_integer "training recording count" "${training_count}" || exit 1
require_positive_integer "held-out recording count" "${held_out_count}" || exit 1
require_positive_integer "training steps" "${training_steps}" || exit 1
require_nonnegative_integer "base seed" "${base_seed}" || exit 1
(( training_count >= 12 && training_count <= 20 )) \
  || die "grasp proposal readiness requires 12 to 20 training seeds"
(( held_out_count == 2 )) || die "grasp readiness requires exactly two held-out seeds"
(( training_steps <= 5000 )) || die "training steps must not exceed 5000"

experiment_id="grasp-$(date -u +%Y%m%dT%H%M%SZ)-${base_seed}"
proposal_name="${experiment_id}_proposal"
training_recording_names=()
held_out_recording_names=()
for ((index = 0; index < training_count; index++)); do
  printf -v recording_id '%s-train-%02d' "${experiment_id}" "${index}"
  seed=$((base_seed + index))
  training_recording_names+=("${recording_id}")
  "${aws_workflow}" demo-record-grasp "${recording_id}" "${seed}" train
  "${aws_workflow}" jepa-wm-grasp-validate "${recording_id}" train
done
for ((index = 0; index < held_out_count; index++)); do
  printf -v recording_id '%s-held-%02d' "${experiment_id}" "${index}"
  seed=$((base_seed + 10000 + index))
  held_out_recording_names+=("${recording_id}")
  "${aws_workflow}" demo-record-grasp "${recording_id}" "${seed}" held_out
  "${aws_workflow}" jepa-wm-grasp-validate "${recording_id}" held_out
done
training_recordings="$(IFS=,; printf '%s' "${training_recording_names[*]}")"
held_out_recordings="$(IFS=,; printf '%s' "${held_out_recording_names[*]}")"
"${aws_workflow}" jepa-wm-proposal-train \
  "${training_recordings}" wrist "${training_steps}" "${proposal_name}"
for recording_id in "${held_out_recording_names[@]}"; do
  "${aws_workflow}" jepa-wm-proposal-eval \
    "${recording_id}" wrist 69 30 1 "${proposal_name}"
done
"${aws_workflow}" jepa-wm-proposal-summarize \
  "${held_out_recordings}" wrist 69 30 1 "${proposal_name}"
"${aws_workflow}" backup-state
printf 'Grasp experiment: %s\nProposal: %s\n' \
  "${experiment_id}" "${proposal_name}"
