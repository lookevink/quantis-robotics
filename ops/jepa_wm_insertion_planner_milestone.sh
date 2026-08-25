#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"
aws_workflow="${AWS_WORKFLOW:-${repo_root}/ops/aws.sh}"

backup_on_exit() {
  local status=$?
  trap - EXIT
  if ! "${aws_workflow}" backup-state; then
    printf 'error: insertion planner recovery backup failed\n' >&2
    (( status != 0 )) || status=1
  fi
  exit "${status}"
}
trap backup_on_exit EXIT

fresh_roster="${1:-/tmp/contact-insertion-v9-2600-fresh-22600_insertion_fresh_evaluation.json}"
proposal_name="${2:-contact-insertion-v9-2600_insertion_proposal_h256_s3000}"
[[ -f "${fresh_roster}" ]] || {
  printf 'error: fresh insertion roster does not exist: %s\n' "${fresh_roster}" >&2
  exit 1
}
is_safe_identifier "${proposal_name}" || {
  printf 'error: proposal name must be safe\n' >&2
  exit 1
}
cd "${repo_root}"
adapter_name="$(python3 -m jepa_wm.insertion_corpus show-fresh \
  --roster "${fresh_roster}" --format adapter-name)"
is_safe_identifier "${adapter_name}" || {
  printf 'error: adapter name must be safe\n' >&2
  exit 1
}
while IFS=$'\t' read -r -u 3 recording_id _seed split; do
  [[ "${split}" == "held_out" ]] || {
    printf 'error: planner roster must contain held-out recordings\n' >&2
    exit 1
  }
  "${aws_workflow}" jepa-wm-insertion-plan-benchmark \
    "${recording_id}" "${adapter_name}" "${proposal_name}"
done 3< <(
  python3 -m jepa_wm.insertion_corpus show-fresh \
    --roster "${fresh_roster}" --format tsv
)
"${aws_workflow}" jepa-wm-insertion-plan-summarize \
  "${fresh_roster}" "${proposal_name}"
