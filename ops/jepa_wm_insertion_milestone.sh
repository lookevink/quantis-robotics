#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"
aws_workflow="${AWS_WORKFLOW:-${repo_root}/ops/aws.sh}"
corpus_workflow="${CORPUS_WORKFLOW:-${repo_root}/ops/jepa_wm_insertion_corpus.sh}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

training_steps="${1:-3000}"
base_seed="${2:-2600}"
experiment_id="${3:-contact-insertion-v9-${base_seed}}"
require_positive_integer "training steps" "${training_steps}" || exit 1
require_nonnegative_integer "base seed" "${base_seed}" || exit 1
(( training_steps <= 5000 )) || die "training steps must not exceed 5000"
is_safe_identifier "${experiment_id}" || die "experiment ID must be safe"
roster_path="${INSERTION_CORPUS_ROSTER:-/tmp/${experiment_id}_insertion_corpus.json}"

backup_on_exit() {
  local status=$?
  trap - EXIT
  if ! "${aws_workflow}" backup-state; then
    printf 'error: insertion milestone recovery backup failed\n' >&2
    (( status != 0 )) || status=1
  fi
  exit "${status}"
}
trap backup_on_exit EXIT

INSERTION_CORPUS_ROSTER="${roster_path}" \
  "${corpus_workflow}" 12 2 "${base_seed}" "${experiment_id}"
cd "${repo_root}"
training_list="$(
  python3 -m jepa_wm.insertion_corpus show \
    --roster "${roster_path}" --format train-csv
)"
held_out_list="$(
  python3 -m jepa_wm.insertion_corpus show \
    --roster "${roster_path}" --format held-out-csv
)"
IFS=',' read -r -a held_out_recordings <<<"${held_out_list}"
proposal_name="${experiment_id}_insertion_proposal_h256_s${training_steps}"

"${aws_workflow}" jepa-wm-insertion-proposal-train \
  "${training_list}" "${training_steps}" "${proposal_name}" \
  256 0.001 0.0001 "${base_seed}"
for recording_id in "${held_out_recordings[@]}"; do
  "${aws_workflow}" jepa-wm-insertion-proposal-eval \
    "${recording_id}" "${proposal_name}"
done
set +e
"${aws_workflow}" jepa-wm-insertion-proposal-summarize \
  "${held_out_list}" "${proposal_name}" "${experiment_id}" "${base_seed}"
readiness_status=$?
set -e
printf 'Insertion experiment: %s\nProposal: %s\n' \
  "${experiment_id}" "${proposal_name}"
exit "${readiness_status}"
