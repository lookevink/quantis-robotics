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

configure_artifact() {
  local kind="$1"
  train_extra=()
  stop_worker=false
  case "${kind}" in
    proposal)
      default_steps=3000
      artifact_label="Proposal"
      artifact_stem="insertion_proposal_h256_s"
      training_command=jepa-wm-insertion-proposal-train
      evaluation_command=jepa-wm-insertion-proposal-eval
      summary_command=jepa-wm-insertion-proposal-summarize
      train_extra=(256 0.001 0.0001)
      include_seed=true
      ;;
    world_model)
      default_steps=500
      artifact_label="Adapter"
      artifact_stem="insertion_adapter_s"
      training_command=jepa-wm-insertion-adapt
      evaluation_command=jepa-wm-insertion-wm-eval
      summary_command=jepa-wm-insertion-wm-summarize
      include_seed=false
      stop_worker=true
      ;;
    *) die "artifact kind must be proposal or world_model" ;;
  esac
}

artifact_kind="${1:-}"
shift || true
configure_artifact "${artifact_kind}"

training_steps="${1:-${default_steps}}"
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
    printf 'error: insertion %s recovery backup failed\n' "${artifact_kind}" >&2
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

artifact_name="${experiment_id}_${artifact_stem}${training_steps}"
[[ "${stop_worker}" == false ]] || "${aws_workflow}" jepa-wm-control-worker-stop
train_arguments=("${training_list}" "${training_steps}" "${artifact_name}")
if [[ "${include_seed}" == true ]]; then
  train_arguments+=("${train_extra[@]}" "${base_seed}")
fi
"${aws_workflow}" "${training_command}" "${train_arguments[@]}"

for recording_id in "${held_out_recordings[@]}"; do
  "${aws_workflow}" "${evaluation_command}" "${recording_id}" "${artifact_name}"
done
set +e
"${aws_workflow}" "${summary_command}" \
  "${held_out_list}" "${artifact_name}" "${experiment_id}" "${base_seed}"
readiness_status=$?
set -e
printf 'Insertion %s experiment: %s\n%s: %s\n' \
  "${artifact_kind}" "${experiment_id}" "${artifact_label}" "${artifact_name}"
exit "${readiness_status}"
