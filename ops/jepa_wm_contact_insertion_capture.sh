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

recording_id="${1:-}"
seed="${2:-}"
split="${3:-}"
is_safe_identifier "${recording_id}" || die "recording ID must be safe"
require_nonnegative_integer "recording seed" "${seed}" || exit 1
[[ "${split}" == train || "${split}" == held_out ]] \
  || die "recording split must be train or held_out"

status="$("${aws_workflow}" jepa-wm-contact-insertion-status \
  "${recording_id}" "${split}" "${seed}")"
case "${status}" in
  valid)
    printf 'Reusing validated %s recording %s (seed %s)\n' \
      "${split}" "${recording_id}" "${seed}"
    exit 0
    ;;
  running)
    printf 'Reconnecting to active recording %s\n' "${recording_id}"
    "${aws_workflow}" demo-wait-recording "${recording_id}"
    ;;
  missing)
    "${aws_workflow}" demo-record-contact-insertion \
      "${recording_id}" "${seed}" "${split}"
    ;;
  partial)
    printf 'Quarantining incomplete recording %s before retry\n' "${recording_id}"
    "${aws_workflow}" demo-quarantine-partial-recording "${recording_id}"
    "${aws_workflow}" demo-record-contact-insertion \
      "${recording_id}" "${seed}" "${split}"
    ;;
  invalid)
    die "recording ${recording_id} exists but fails its exact split/seed/v9 contract"
    ;;
  *) die "recording status returned an invalid state for ${recording_id}: ${status}" ;;
esac
"${aws_workflow}" jepa-wm-contact-insertion-validate \
  "${recording_id}" "${split}" "${seed}"
