#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"
aws_workflow="${AWS_WORKFLOW:-${repo_root}/ops/aws.sh}"

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

capture_recording() {
  local recording_id="$1"
  local seed="$2"
  local split="$3"
  local status
  status="$("${aws_workflow}" jepa-wm-contact-insertion-status \
    "${recording_id}" "${split}" "${seed}")"
  case "${status}" in
    valid)
      printf 'Reusing validated %s recording %s (seed %s)\n' \
        "${split}" "${recording_id}" "${seed}"
      return
      ;;
    running)
      printf 'Reconnecting to active recording %s\n' "${recording_id}"
      "${aws_workflow}" demo-wait-recording "${recording_id}"
      "${aws_workflow}" jepa-wm-contact-insertion-validate \
        "${recording_id}" "${split}" "${seed}"
      return
      ;;
    missing)
      ;;
    partial)
      printf 'Quarantining incomplete recording %s before retry\n' "${recording_id}"
      "${aws_workflow}" demo-quarantine-partial-recording "${recording_id}"
      ;;
    invalid)
      die "recording ${recording_id} exists but fails its exact split/seed/v9 contract"
      ;;
    *)
      die "recording status returned an invalid state for ${recording_id}: ${status}"
      ;;
  esac
  "${aws_workflow}" demo-record-contact-insertion \
    "${recording_id}" "${seed}" "${split}"
  "${aws_workflow}" jepa-wm-contact-insertion-validate \
    "${recording_id}" "${split}" "${seed}"
}

while IFS=$'\t' read -r recording_id seed split; do
  capture_recording "${recording_id}" "${seed}" "${split}"
done < <(
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
