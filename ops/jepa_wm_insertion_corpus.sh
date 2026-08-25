#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"
aws_workflow="${AWS_WORKFLOW:-${repo_root}/ops/aws.sh}"
capture_workflow="${CONTACT_INSERTION_CAPTURE_WORKFLOW:-${repo_root}/ops/jepa_wm_contact_insertion_capture.sh}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

training_count="${1:-12}"
held_out_count="${2:-2}"
base_seed="${3:-2600}"
experiment_id="${4:-contact-insertion-v9-${base_seed}}"
roster_path="${INSERTION_CORPUS_ROSTER:-/tmp/${experiment_id}_insertion_corpus.json}"
require_positive_integer "training recording count" "${training_count}" || exit 1
require_positive_integer "held-out recording count" "${held_out_count}" || exit 1
require_nonnegative_integer "base seed" "${base_seed}" || exit 1
is_safe_identifier "${experiment_id}" || die "experiment ID must be a safe identifier"
(( training_count == 12 )) || die "insertion corpus requires exactly 12 TRAIN seeds"
(( held_out_count == 2 )) || die "insertion corpus requires exactly two HELD_OUT seeds"
cd "${repo_root}"
python3 -m jepa_wm.insertion_corpus create \
  --experiment-id "${experiment_id}" \
  --base-seed "${base_seed}" \
  --output "${roster_path}"

backup_on_exit() {
  local status=$?
  trap - EXIT
  if ! "${aws_workflow}" backup-state; then
    printf 'error: insertion corpus recovery backup failed\n' >&2
    (( status != 0 )) || status=1
  fi
  exit "${status}"
}
trap backup_on_exit EXIT

while IFS=$'\t' read -r -u 3 recording_id seed split; do
  AWS_WORKFLOW="${aws_workflow}" \
    "${capture_workflow}" "${recording_id}" "${seed}" "${split}"
done 3< <(
  python3 -m jepa_wm.insertion_corpus show \
    --roster "${roster_path}" --format tsv
)
training_recordings="$(
  python3 -m jepa_wm.insertion_corpus show \
    --roster "${roster_path}" --format train-csv
)"
held_out_recordings="$(
  python3 -m jepa_wm.insertion_corpus show \
    --roster "${roster_path}" --format held-out-csv
)"

printf 'Insertion corpus: %s\nRoster: %s\nTRAIN: %s\nHELD_OUT: %s\n' \
  "${experiment_id}" "${roster_path}" \
  "${training_recordings}" "${held_out_recordings}"
