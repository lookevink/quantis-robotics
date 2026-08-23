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

training_count="${1:-4}"
held_out_count="${2:-2}"
training_steps="${3:-500}"
base_seed="${4:-1200}"
require_positive_integer "training recording count" "${training_count}" || exit 1
require_positive_integer "held-out recording count" "${held_out_count}" || exit 1
require_positive_integer "training steps" "${training_steps}" || exit 1
require_nonnegative_integer "base seed" "${base_seed}" || exit 1
(( training_count <= 20 && held_out_count <= 20 )) \
  || die "recording counts must not exceed 20 per split"
(( training_steps <= 5000 )) || die "training steps must not exceed 5000"

experiment_id="domain-$(date -u +%Y%m%dT%H%M%SZ)-${base_seed}"
training_recording_names=()
held_out_recording_names=()
for ((index = 0; index < training_count; index++)); do
  printf -v recording_id '%s-train-%02d' "${experiment_id}" "${index}"
  exploration_seed=$((base_seed + index))
  training_recording_names+=("${recording_id}")
  "${aws_workflow}" demo-record-exploration \
    "${recording_id}" "${exploration_seed}" train
done
for ((index = 0; index < held_out_count; index++)); do
  printf -v recording_id '%s-held-%02d' "${experiment_id}" "${index}"
  exploration_seed=$((base_seed + 10000 + index))
  held_out_recording_names+=("${recording_id}")
  "${aws_workflow}" demo-record-exploration \
    "${recording_id}" "${exploration_seed}" held_out
done
training_recordings="$(IFS=,; printf '%s' "${training_recording_names[*]}")"
held_out_recordings="$(IFS=,; printf '%s' "${held_out_recording_names[*]}")"
"${aws_workflow}" jepa-wm-adapt-set \
  "${training_recordings}" wrist "${training_steps}"
for recording_id in "${held_out_recording_names[@]}"; do
  "${aws_workflow}" jepa-wm-eval-adapted \
    "${recording_id}" wrist 0 40 1
done
"${aws_workflow}" jepa-wm-summarize \
  "${experiment_id}" "${training_recordings}" "${held_out_recordings}" wrist 40
"${aws_workflow}" backup-state
printf 'Experiment ID: %s\n' "${experiment_id}"
