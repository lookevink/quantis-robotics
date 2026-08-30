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

backup_on_exit() {
  local status=$?
  trap - EXIT
  if ! "${aws_workflow}" backup-state; then
    printf 'error: fresh insertion evaluation recovery backup failed\n' >&2
    (( status != 0 )) || status=1
  fi
  exit "${status}"
}
trap backup_on_exit EXIT

source_base_seed="${1:-2600}"
source_experiment="${2:-contact-insertion-v10-drive-slow-${source_base_seed}}"
fresh_base_seed="${3:-22600}"
evaluation_id="${4:-${source_experiment}-fresh-${fresh_base_seed}}"
epoch_steps="$(insertion_epoch_steps "${repo_root}" python3)"
adapter_name="${5:-${source_experiment}_insertion_adapter_goal_aligned_relative_finetune_s${epoch_steps}}"
adapter_profile="${6:-goal_aligned_relative_finetune}"
adapter_fingerprint="${7:-}"
require_nonnegative_integer "source base seed" "${source_base_seed}" || exit 1
require_nonnegative_integer "fresh base seed" "${fresh_base_seed}" || exit 1
is_safe_identifier "${source_experiment}" || die "source experiment ID must be safe"
is_safe_identifier "${evaluation_id}" || die "evaluation ID must be safe"
is_safe_identifier "${adapter_name}" || die "adapter name must be safe"
[[ "${adapter_fingerprint}" =~ ^[0-9a-f]{64}$ ]] \
  || die "the frozen adapter fingerprint must be supplied as a lowercase SHA-256"
cd "${repo_root}"
python3 -m jepa_wm.insertion_adapter_profile \
  "${adapter_profile}" artifact-stem >/dev/null

source_roster="${INSERTION_CORPUS_ROSTER:-/tmp/${source_experiment}_insertion_corpus.json}"
fresh_roster="${INSERTION_FRESH_ROSTER:-/tmp/${evaluation_id}_insertion_fresh_evaluation.json}"
python3 -m jepa_wm.insertion_corpus create \
  --experiment-id "${source_experiment}" \
  --base-seed "${source_base_seed}" \
  --output "${source_roster}"
python3 -m jepa_wm.insertion_corpus create-fresh \
  --evaluation-id "${evaluation_id}" \
  --base-seed "${fresh_base_seed}" \
  --source-roster "${source_roster}" \
  --adapter-name "${adapter_name}" \
  --adapter-fingerprint "${adapter_fingerprint}" \
  --output "${fresh_roster}"

while IFS=$'\t' read -r -u 3 recording_id seed split; do
  AWS_WORKFLOW="${aws_workflow}" \
    "${capture_workflow}" "${recording_id}" "${seed}" "${split}"
done 3< <(
  python3 -m jepa_wm.insertion_corpus show-fresh \
    --roster "${fresh_roster}" --format tsv
)

held_out_list="$(
  python3 -m jepa_wm.insertion_corpus show-fresh \
    --roster "${fresh_roster}" --format held-out-csv
)"
IFS=',' read -r -a held_out_recordings <<<"${held_out_list}"
"${aws_workflow}" jepa-wm-control-worker-stop
for recording_id in "${held_out_recordings[@]}"; do
  "${aws_workflow}" jepa-wm-insertion-wm-eval \
    "${recording_id}" "${adapter_name}"
done

set +e
"${aws_workflow}" jepa-wm-insertion-wm-fresh-summarize \
  "${fresh_roster}" "${adapter_profile}"
readiness_status=$?
set -e
printf 'Fresh insertion world-model evaluation: %s\nAdapter: %s\n' \
  "${evaluation_id}" "${adapter_name}"
exit "${readiness_status}"
