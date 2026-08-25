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
    printf 'error: insertion resolution recovery backup failed\n' >&2
    (( status != 0 )) || status=1
  fi
  exit "${status}"
}
trap backup_on_exit EXIT

reference_recording="${1:-contact-insertion-v9-2600-fresh-52600-held-00}"
seed="${2:-52600}"
is_safe_identifier "${reference_recording}" || {
  printf 'error: reference recording must be safe\n' >&2
  exit 1
}
require_nonnegative_integer "exploration seed" "${seed}" || exit 1

# First, middle, and final command contexts, with and without the attached load.
while IFS=$'\t' read -r -u 3 context load; do
  "${aws_workflow}" jepa-wm-insertion-resolution \
    "${reference_recording}" "${seed}" "${context}" "${load}"
done 3< <(
  control_resolution_profile_field "${repo_root}" python3 roster
)
