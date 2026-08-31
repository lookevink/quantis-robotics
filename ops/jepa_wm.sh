#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
jepa_wm_home="${JEPA_WM_HOME:-${HOME}/docker/jepa-wm}"
source_dir="${jepa_wm_home}/source/jepa-wms"
dinov3_dir="${jepa_wm_home}/source/dinov3"
venv_dir="${HOME}/.venvs/quantis-jepa-wm"
bootstrap_venv="${jepa_wm_home}/bootstrap-venv"
checkpoint_dir="${jepa_wm_home}/checkpoints"
cache_dir="${jepa_wm_home}/cache"
model_id="jepa_wm_droid"
jepa_checkpoint="${checkpoint_dir}/${model_id}.pth.tar"
dinov3_checkpoint_dir="${checkpoint_dir}/dinov3"
dinov3_checkpoint="${dinov3_checkpoint_dir}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
dinov3_expected_checkpoint="${dinov3_checkpoint_dir}/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
dinov3_cached_checkpoint="${cache_dir}/torch/hub/checkpoints/$(basename "${dinov3_checkpoint}")"
control_run_dir="${jepa_wm_home}/run"
control_socket="${control_run_dir}/control.sock"
control_pid_file="${control_run_dir}/control.pid"
control_artifacts_file="${control_run_dir}/control.artifacts"
control_log="${jepa_wm_home}/logs/control-worker.log"
control_frame_root="${HOME}/docker/isaac-sim/data/quantis"
jepa_revision="${JEPA_WM_REVISION:-13cf1d9c7e476f53c17714d2e0f1dc239a883ce0}"
dinov3_revision="${DINOV3_REVISION:-6876159a11b4df116f30f667f8c9888617df0751}"
uv_version="${JEPA_WM_UV_VERSION:-0.8.15}"
python_version="${JEPA_WM_PYTHON_VERSION:-3.10.18}"

export JEPAWM_HOME="${jepa_wm_home}/source"
export JEPAWM_DSET="${jepa_wm_home}/datasets"
export JEPAWM_LOGS="${jepa_wm_home}/logs"
export JEPAWM_CKPT="${checkpoint_dir}"
export JEPAWM_OSSCKPT="${checkpoint_dir}"
export JEPA_WM_REVISION="${jepa_revision}"
export HF_HOME="${cache_dir}/huggingface"
export TORCH_HOME="${cache_dir}/torch"
export XDG_CACHE_HOME="${cache_dir}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

parse_named_options() {
  local target_name="$1"
  local allowed_options="$2"
  shift 2
  local -n target="${target_name}"
  local option_name
  (( $# % 2 == 0 )) || die "named options require a value"
  while (( $# )); do
    [[ "$1" == --* ]] || die "expected a named option, received: $1"
    option_name="${1#--}"
    [[ " ${allowed_options} " == *" ${option_name} "* ]] \
      || die "unknown option: $1"
    [[ -z "${target[${option_name}]+x}" ]] || die "duplicate option: $1"
    target["${option_name}"]="$2"
    shift 2
  done
}

task_proposal_window() {
  local task_name="$1"
  require_runtime
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.task_windows "${task_name}"
}

task_proposal_setting() {
  local task_name="$1"
  local setting="$2"
  case "${task_name}:${setting}" in
    grasp:seed) printf '234\n' ;;
    contact-grasp:seed) printf '2600\n' ;;
    insertion:seed) printf '2600\n' ;;
    grasp:inactive_gripper) printf '0\n' ;;
    contact-grasp:inactive_gripper) printf '0\n' ;;
    insertion:inactive_gripper) printf '0.01\n' ;;
    grasp:goal_direction) printf '0\n' ;;
    contact-grasp:goal_direction) printf '0\n' ;;
    insertion:goal_direction) printf '1.0\n' ;;
    grasp:readiness_module) printf 'jepa_wm.grasp_proposal_readiness\n' ;;
    contact-grasp:readiness_module) printf 'jepa_wm.contact_grasp_proposal_readiness\n' ;;
    insertion:readiness_module) printf 'jepa_wm.insertion_proposal_readiness\n' ;;
    grasp:readiness_suffix) printf 'grasp_readiness\n' ;;
    contact-grasp:readiness_suffix) printf 'contact_grasp_readiness\n' ;;
    insertion:readiness_suffix) printf 'insertion_readiness\n' ;;
    *) die "unsupported task proposal setting: ${task_name}:${setting}" ;;
  esac
}

download_file() {
  local url="$1"
  local destination="$2"
  [[ -s "${destination}" ]] && return
  mkdir -p "$(dirname "${destination}")"
  curl --fail --location --retry 3 --continue-at - \
    --output "${destination}.part" "${url}"
  mv "${destination}.part" "${destination}"
}

checkout_repository() {
  local url="$1"
  local revision="$2"
  local destination="$3"
  if [[ ! -d "${destination}/.git" ]]; then
    mkdir -p "$(dirname "${destination}")"
    git clone --filter=blob:none --no-checkout "${url}" "${destination}"
  fi
  if [[ ! -f "${destination}/README.md" ]]; then
    git -C "${destination}" fetch --depth 1 origin "${revision}"
    git -C "${destination}" checkout --detach "${revision}"
    return
  fi
  git -C "${destination}" diff --quiet \
    || die "upstream checkout has local changes: ${destination}"
  git -C "${destination}" diff --cached --quiet \
    || die "upstream checkout has staged changes: ${destination}"
  git -C "${destination}" fetch --depth 1 origin "${revision}"
  git -C "${destination}" checkout --detach "${revision}"
}

ensure_uv() {
  if [[ ! -x "${bootstrap_venv}/bin/uv" ]]; then
    python3 -m venv "${bootstrap_venv}"
  fi
  "${bootstrap_venv}/bin/pip" install --quiet --upgrade "uv==${uv_version}"
}

install_runtime() {
  mkdir -p \
    "${cache_dir}" \
    "${checkpoint_dir}" \
    "${jepa_wm_home}/datasets" \
    "${jepa_wm_home}/logs"
  ensure_uv
  checkout_repository \
    https://github.com/facebookresearch/jepa-wms.git \
    "${jepa_revision}" \
    "${source_dir}"
  checkout_repository \
    https://github.com/facebookresearch/dinov3.git \
    "${dinov3_revision}" \
    "${dinov3_dir}"

  "${bootstrap_venv}/bin/uv" python install "${python_version}"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    mkdir -p "$(dirname "${venv_dir}")"
    "${bootstrap_venv}/bin/uv" venv \
      --python "${python_version}" "${venv_dir}"
  fi
  "${bootstrap_venv}/bin/uv" pip install --python "${venv_dir}/bin/python" \
    "numpy==1.26.4" \
    "torch==2.7.0" \
    "torchvision==0.22.0" \
    "tensordict>=0.9.1" \
    "timm==1.0.19" \
    "torchmetrics>=1.7.4" \
    "einops" \
    "ftfy" \
    "regex" \
    "pyyaml" \
    "ruamel.yaml" \
    "omegaconf" \
    "pandas" \
    "h5py" \
    "decord>=0.6.0" \
    "opencv-python-headless==4.11.0.86" \
    "scipy" \
    "scikit-learn" \
    "submitit" \
    "tqdm" \
    "matplotlib" \
    "lpips>=0.1.4" \
    "termcolor"
  "${bootstrap_venv}/bin/uv" pip install --python "${venv_dir}/bin/python" \
    --no-deps --editable "${source_dir}"

  download_file \
    https://dl.fbaipublicfiles.com/jepa-wms/droid_jepa-wm_noprop.pth.tar \
    "${jepa_checkpoint}"
  if [[ ! -s "${dinov3_checkpoint}" ]]; then
    [[ -n "${DINOV3_CHECKPOINT_URL:-}" ]] || die \
      "DINOv3 weights require Meta approval; set DINOV3_CHECKPOINT_URL in .env"
    download_file "${DINOV3_CHECKPOINT_URL}" "${dinov3_checkpoint}"
  fi
  # JEPA-WMs currently names the native ViT-L checkpoint with a stale hash.
  # Point that expected name and Torch Hub's cache at the approved 8aa4cbdd file.
  mkdir -p "$(dirname "${dinov3_cached_checkpoint}")"
  ln -sfn "${dinov3_checkpoint}" "${dinov3_expected_checkpoint}"
  ln -sfn "${dinov3_checkpoint}" "${dinov3_cached_checkpoint}"
  printf 'JEPA-WM runtime installed at %s\n' "${jepa_wm_home}"
}

require_runtime() {
  [[ -x "${venv_dir}/bin/python" ]] || die "JEPA-WM environment is not installed"
  [[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${jepa_revision}" ]] \
    || die "JEPA-WM source revision does not match bootstrap"
  [[ "$(git -C "${dinov3_dir}" rev-parse HEAD)" == "${dinov3_revision}" ]] \
    || die "DINOv3 source revision does not match bootstrap"
  [[ -s "${jepa_checkpoint}" ]] || die "JEPA-WM checkpoint is missing"
  [[ -s "${dinov3_checkpoint}" ]] || die "DINOv3 checkpoint is missing"
}

smoke_runtime() {
  require_runtime
  "${venv_dir}/bin/python" "${repo_dir}/jepa_wm/smoke.py" \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}"
}

status_runtime() {
  require_runtime
  printf 'ready model=%s revision=%s python=%s\n' \
    "${model_id}" "${jepa_revision}" \
    "$("${venv_dir}/bin/python" --version 2>&1)"
}

evaluate_recording() {
  local recording_name="$1"
  local camera_name="$2"
  local start_index="$3"
  local transition_count="$4"
  local transition_stride="$5"
  local adapter_mode="${6:-base}"
  local minimum_action_norm="${7:-}"
  is_safe_identifier "${recording_name}" \
    || die "invalid recording name"
  is_safe_identifier "${camera_name}" \
    || die "invalid camera name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "transition count" "${transition_count}" || exit 1
  require_positive_integer "transition stride" "${transition_stride}" || exit 1
  if [[ -n "${minimum_action_norm}" ]]; then
    require_nonnegative_number "minimum action norm" "${minimum_action_norm}" \
      || exit 1
  fi
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  require_runtime
  sudo chown -R "${USER}:${USER}" "${recording}"
  cd "${repo_dir}"
  local arguments=(
    -m jepa_wm.evaluate_recording
    --source "${source_dir}"
    --checkpoint "${jepa_checkpoint}"
    --recording "${recording}"
    --camera "${camera_name}"
    --start-index "${start_index}"
    --count "${transition_count}"
    --stride "${transition_stride}"
  )
  if [[ -n "${minimum_action_norm}" ]]; then
    arguments+=(--minimum-action-norm "${minimum_action_norm}")
  fi
  if [[ "${adapter_mode}" == "adapted" ]]; then
    local adapter="${checkpoint_dir}/quantis_isaac_${camera_name}_action_adapter.pth"
    [[ -s "${adapter}" ]] || die "action adapter is not installed for ${camera_name}"
    arguments+=(--adapter "${adapter}")
  elif [[ "${adapter_mode}" != "base" ]]; then
    is_safe_identifier "${adapter_mode}" || die "invalid adapter name"
    local adapter="${checkpoint_dir}/${adapter_mode}.pth"
    [[ -s "${adapter}" ]] || die "action adapter does not exist: ${adapter_mode}"
    arguments+=(--adapter "${adapter}")
  fi
  "${venv_dir}/bin/python" "${arguments[@]}"
}

adapt_recording_set() {
  local recording_list="$1"
  local camera_name="$2"
  local training_steps="$3"
  local adapter_name="${4:-quantis_isaac_${camera_name}_action_adapter}"
  local start_index="${5:-}"
  local rollout_count="${6:-}"
  local rollout_stride="${7:-}"
  local training_batch_size="${8:-2}"
  local candidate_profile="${9:-}"
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  require_positive_integer "training steps" "${training_steps}" || exit 1
  require_positive_integer "training batch size" "${training_batch_size}" || exit 1
  local -a window_arguments=()
  local -a candidate_profile_arguments=()
  if [[ -n "${start_index}${rollout_count}${rollout_stride}" ]]; then
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
    window_arguments=(
      --start-index "${start_index}"
      --count "${rollout_count}"
      --stride "${rollout_stride}"
    )
  fi
  if [[ -n "${candidate_profile}" ]]; then
    candidate_profile_arguments=(--candidate-profile "${candidate_profile}")
  fi
  local -a recording_names
  local -a recording_arguments
  local recording_name
  local recording
  IFS=',' read -r -a recording_names <<<"${recording_list}"
  for recording_name in "${recording_names[@]}"; do
    recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
    [[ -f "${recording}/manifest.json" ]] \
      || die "recording does not exist: ${recording_name}"
    recording_arguments+=(--recording "${recording}")
  done
  require_runtime
  for recording_name in "${recording_names[@]}"; do
    sudo chown -R "${USER}:${USER}" \
      "${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  done
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.adapt_recording \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    "${recording_arguments[@]}" \
    --output "${adapter}" \
    --camera "${camera_name}" \
    --steps "${training_steps}" \
    --batch-size "${training_batch_size}" \
    "${candidate_profile_arguments[@]}" \
    "${window_arguments[@]}"
}

adapt_insertion_world_model() {
  local -A options=()
  parse_named_options options "recordings steps adapter profile" "$@"
  local candidate_profile="${options[profile]:-generic}"
  local window_start window_count window_stride
  local training_batch_size
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window insertion)"
  training_batch_size="$(
    insertion_training_batch_size "${repo_dir}" "${venv_dir}/bin/python"
  )"
  adapt_recording_set \
    "${options[recordings]:-}" wrist \
    "${options[steps]:-$(
      insertion_epoch_steps "${repo_dir}" "${venv_dir}/bin/python"
    )}" \
    "${options[adapter]:-}" \
    "${window_start}" "${window_count}" "${window_stride}" \
    "${training_batch_size}" \
    "${candidate_profile}"
}

evaluate_insertion_world_model() {
  local -A options=()
  parse_named_options options "recording adapter" "$@"
  local window_start window_count window_stride
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window insertion)"
  evaluate_recording \
    "${options[recording]:-}" wrist \
    "${window_start}" "${window_count}" "${window_stride}" \
    "${options[adapter]:-}" 0
}

summarize_insertion_world_model() {
  local -A options=()
  parse_named_options options \
    "recordings adapter experiment base-seed adapter-profile fresh-roster-base64" "$@"
  local held_out_list="${options[recordings]:-}"
  local adapter_name="${options[adapter]:-}"
  local experiment_id="${options[experiment]:-}"
  local base_seed="${options[base-seed]:-}"
  local adapter_profile="${options[adapter-profile]:-generic}"
  local fresh_roster_payload="${options[fresh-roster-base64]:-}"
  (cd "${repo_dir}" && python3 -m jepa_wm.insertion_adapter_profile \
    "${adapter_profile}" artifact-stem >/dev/null)
  require_runtime
  local experiment_root="${checkpoint_dir}/experiments"
  mkdir -p "${experiment_root}"
  cd "${repo_dir}"
  local readiness_experiment
  local -a roster_arguments
  if [[ -n "${fresh_roster_payload}" ]]; then
    local temporary_roster
    temporary_roster="$(mktemp "${experiment_root}/.insertion-fresh.XXXXXX.json")"
    if ! printf '%s' "${fresh_roster_payload}" \
      | base64 --decode >"${temporary_roster}"; then
      rm -f "${temporary_roster}"
      die "fresh evaluation roster encoding is invalid"
    fi
    local roster_evaluation roster_adapter
    if ! roster_evaluation="$("${venv_dir}/bin/python" \
      -m jepa_wm.insertion_corpus show-fresh \
      --roster "${temporary_roster}" --format evaluation-id)"; then
      rm -f "${temporary_roster}"
      die "fresh evaluation roster is invalid"
    fi
    if ! roster_adapter="$("${venv_dir}/bin/python" \
      -m jepa_wm.insertion_corpus show-fresh \
      --roster "${temporary_roster}" --format adapter-name)"; then
      rm -f "${temporary_roster}"
      die "fresh evaluation roster adapter is invalid"
    fi
    is_safe_identifier "${roster_evaluation}" \
      || die "fresh evaluation roster identity is invalid"
    is_safe_identifier "${roster_adapter}" \
      || die "fresh evaluation roster adapter is invalid"
    local fresh_roster="${experiment_root}/${roster_evaluation}_insertion_fresh_evaluation.json"
    mv "${temporary_roster}" "${fresh_roster}"
    readiness_experiment="${roster_evaluation}"
    adapter_name="${roster_adapter}"
    held_out_list="$("${venv_dir}/bin/python" -m jepa_wm.insertion_corpus \
      show-fresh --roster "${fresh_roster}" --format held-out-csv)"
    roster_arguments=(--fresh-evaluation-roster "${fresh_roster}")
  else
    is_safe_identifier_list "${held_out_list}" || die "invalid held-out list"
    is_safe_identifier "${experiment_id}" || die "invalid experiment ID"
    require_nonnegative_integer "base seed" "${base_seed}" || exit 1
    local roster="${experiment_root}/${experiment_id}_insertion_corpus.json"
    "${venv_dir}/bin/python" -m jepa_wm.insertion_corpus create \
      --experiment-id "${experiment_id}" --base-seed "${base_seed}" \
      --output "${roster}"
    readiness_experiment="${experiment_id}"
    roster_arguments=(--roster "${roster}")
  fi
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  [[ -s "${adapter}" ]] || die "action adapter does not exist: ${adapter_name}"
  is_safe_identifier_list "${held_out_list}" || die "invalid held-out list"
  local -a held_out_names reports=()
  local recording_name report
  local window_start window_count window_stride
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window insertion)"
  IFS=',' read -r -a held_out_names <<<"${held_out_list}"
  for recording_name in "${held_out_names[@]}"; do
    printf -v report '%s/jepa_wm/wrist_%s_rollout_eval_%06d_%03d.json' \
      "${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}" \
      "${adapter_name}" "${window_start}" "${window_count}"
    [[ -f "${report}" ]] || die "insertion adapter report does not exist: ${report}"
    reports+=(--evaluation-report "${report}")
  done
  local output="${checkpoint_dir}/experiments/${adapter_name}_insertion_wm_readiness.json"
  if [[ -n "${fresh_roster_payload}" ]]; then
    output="${checkpoint_dir}/experiments/${adapter_name}_${readiness_experiment}_insertion_wm_readiness.json"
  fi
  "${venv_dir}/bin/python" -m jepa_wm.insertion_wm_readiness \
    --experiment-id "${readiness_experiment}" \
    --adapter "${adapter}" \
    "${reports[@]}" \
    "${roster_arguments[@]}" \
    --adapter-profile "${adapter_profile}" \
    --output "${output}"
}

summarize_insertion_planner() {
  local -A options=()
  parse_named_options options "fresh-roster-base64 proposal profile" "$@"
  local fresh_roster_payload="${options[fresh-roster-base64]:-}"
  local proposal_name="${options[proposal]:-}"
  local profile="${options[profile]:-}"
  [[ -n "${fresh_roster_payload}" ]] || die "fresh planner roster is required"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_runtime
  profile="$(insertion_planner_profile_field \
    "${repo_dir}" "${venv_dir}/bin/python" "${profile}" name)" \
    || die "invalid insertion planner profile"
  local experiment_root="${checkpoint_dir}/experiments"
  mkdir -p "${experiment_root}"
  local temporary_roster
  temporary_roster="$(mktemp "${experiment_root}/.insertion-planner.XXXXXX.json")"
  if ! printf '%s' "${fresh_roster_payload}" \
    | base64 --decode >"${temporary_roster}"; then
    rm -f "${temporary_roster}"
    die "fresh planner roster encoding is invalid"
  fi
  cd "${repo_dir}"
  local evaluation_id adapter_name held_out_list
  if ! evaluation_id="$("${venv_dir}/bin/python" \
    -m jepa_wm.insertion_corpus show-fresh \
    --roster "${temporary_roster}" --format evaluation-id)" \
    || ! adapter_name="$("${venv_dir}/bin/python" \
      -m jepa_wm.insertion_corpus show-fresh \
      --roster "${temporary_roster}" --format adapter-name)" \
    || ! held_out_list="$("${venv_dir}/bin/python" \
      -m jepa_wm.insertion_corpus show-fresh \
      --roster "${temporary_roster}" --format held-out-csv)"; then
    rm -f "${temporary_roster}"
    die "fresh planner roster is invalid"
  fi
  is_safe_identifier "${evaluation_id}" || die "invalid planner evaluation ID"
  is_safe_identifier "${adapter_name}" || die "invalid planner adapter name"
  is_safe_identifier_list "${held_out_list}" || die "invalid planner recording list"
  local fresh_roster="${experiment_root}/${evaluation_id}_insertion_fresh_evaluation.json"
  mv "${temporary_roster}" "${fresh_roster}"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${adapter}" ]] || die "planner adapter does not exist: ${adapter_name}"
  [[ -s "${proposal}" ]] || die "planner proposal does not exist: ${proposal_name}"
  local output_suffix
  output_suffix="$(insertion_planner_profile_field \
    "${repo_dir}" "${venv_dir}/bin/python" "${profile}" report-suffix)"
  local output="${experiment_root}/${evaluation_id}_${output_suffix}.json"
  "${venv_dir}/bin/python" -m jepa_wm.insertion_planner_readiness \
    --fresh-roster "${fresh_roster}" \
    --recording-root "${HOME}/docker/isaac-sim/data/quantis/recordings" \
    --adapter "${adapter}" \
    --proposal "${proposal}" \
    --base-checkpoint "${jepa_checkpoint}" \
    --profile "${profile}" \
    --output "${output}"
}

diagnose_insertion_proposal_training() {
  local -A options=()
  parse_named_options options "proposal output" "$@"
  local proposal_name="${options[proposal]:-}"
  local output_name="${options[output]:-${proposal_name}_insertion_training_diagnostic}"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  is_safe_identifier "${output_name}" || die "invalid diagnostic output name"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${proposal}" ]] || die "proposal does not exist: ${proposal_name}"
  require_runtime
  mkdir -p "${checkpoint_dir}/experiments"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.insertion_proposal_training_diagnostic \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --proposal "${proposal}" \
    --recording-root "${HOME}/docker/isaac-sim/data/quantis/recordings" \
    --output "${checkpoint_dir}/experiments/${output_name}.json"
}

benchmark_planner() {
  local -A options=()
  parse_named_options options \
    "recording camera start-index count stride iterations samples elites adapter proposal" \
    "$@"
  local recording_name="${options[recording]:-}"
  local camera_name="${options[camera]:-wrist}"
  local start_index="${options[start-index]:-0}"
  local rollout_count="${options[count]:-8}"
  local rollout_stride="${options[stride]:-1}"
  local planner_iterations="${options[iterations]:-6}"
  local planner_samples="${options[samples]:-300}"
  local planner_elites="${options[elites]:-10}"
  local adapter_name="${options[adapter]:-quantis_isaac_${camera_name}_action_adapter}"
  local proposal_name="${options[proposal]:-}"
  is_safe_identifier "${recording_name}" || die "invalid recording name"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  if [[ -n "${proposal_name}" ]]; then
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  fi
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
  require_positive_integer "planner iterations" "${planner_iterations}" || exit 1
  require_positive_integer "planner samples" "${planner_samples}" || exit 1
  require_positive_integer "planner elites" "${planner_elites}" || exit 1
  (( planner_elites <= planner_samples )) \
    || die "planner elites must not exceed planner samples"
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  [[ -s "${adapter}" ]] || die "action adapter is not installed for ${camera_name}"
  require_runtime
  sudo chown -R "${USER}:${USER}" "${recording}"
  cd "${repo_dir}"
  local -a arguments=(
    -m jepa_wm.benchmark_planner
    --source "${source_dir}"
    --checkpoint "${jepa_checkpoint}"
    --recording "${recording}"
    --adapter "${adapter}"
    --camera "${camera_name}"
    --start-index "${start_index}"
    --count "${rollout_count}"
    --stride "${rollout_stride}"
    --iterations "${planner_iterations}"
    --samples "${planner_samples}"
    --elites "${planner_elites}"
  )
  if [[ -n "${proposal_name}" ]]; then
    local proposal="${checkpoint_dir}/${proposal_name}.pth"
    [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
    arguments+=(--proposal "${proposal}")
  fi
  "${venv_dir}/bin/python" "${arguments[@]}"
}

benchmark_insertion_planner() {
  local -A options=()
  parse_named_options options "recording adapter proposal profile" "$@"
  local recording_name="${options[recording]:-}"
  local adapter_name="${options[adapter]:-}"
  local proposal_name="${options[proposal]:-}"
  local profile="${options[profile]:-}"
  is_safe_identifier "${recording_name}" || die "invalid recording name"
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_runtime
  profile="$(insertion_planner_profile_field \
    "${repo_dir}" "${venv_dir}/bin/python" "${profile}" name)" \
    || die "invalid insertion planner profile"
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  [[ -s "${adapter}" ]] || die "insertion adapter does not exist: ${adapter_name}"
  [[ -s "${proposal}" ]] || die "insertion proposal does not exist: ${proposal_name}"
  sudo chown -R "${USER}:${USER}" "${recording}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.insertion_planner_benchmark \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --recording "${recording}" \
    --adapter "${adapter}" \
    --proposal "${proposal}" \
    --profile "${profile}"
}

train_action_proposal() {
  local -A options=()
  parse_named_options options \
    "recordings camera steps proposal start-index count stride hidden-dimension learning-rate weight-decay seed goal-consistency-weight first-action-weight active-direction-weight goal-direction-weight inactive-gripper-weight first-gripper-weight" "$@"
  local recording_list="${options[recordings]:-}"
  local camera_name="${options[camera]:-wrist}"
  local training_steps="${options[steps]:-2000}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  local start_index="${options[start-index]:-}"
  local rollout_count="${options[count]:-}"
  local rollout_stride="${options[stride]:-}"
  local hidden_dimension="${options[hidden-dimension]:-128}"
  local learning_rate="${options[learning-rate]:-0.001}"
  local weight_decay="${options[weight-decay]:-0.0001}"
  local training_seed="${options[seed]:-234}"
  local goal_consistency_weight="${options[goal-consistency-weight]:-1.0}"
  local first_action_weight="${options[first-action-weight]:-1.0}"
  local active_direction_weight="${options[active-direction-weight]:-0.1}"
  local goal_direction_weight="${options[goal-direction-weight]:-0}"
  local inactive_gripper_weight="${options[inactive-gripper-weight]:-0.01}"
  local first_gripper_weight="${options[first-gripper-weight]:-1.0}"
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_positive_integer "training steps" "${training_steps}" || exit 1
  require_positive_integer "hidden dimension" "${hidden_dimension}" || exit 1
  require_nonnegative_integer "training seed" "${training_seed}" || exit 1
  require_nonnegative_number "learning rate" "${learning_rate}" || exit 1
  require_nonnegative_number "weight decay" "${weight_decay}" || exit 1
  require_nonnegative_number "goal consistency weight" \
    "${goal_consistency_weight}" || exit 1
  require_nonnegative_number "first action weight" \
    "${first_action_weight}" || exit 1
  require_nonnegative_number "active direction weight" \
    "${active_direction_weight}" || exit 1
  require_nonnegative_number "goal direction weight" \
    "${goal_direction_weight}" || exit 1
  require_nonnegative_number "inactive gripper weight" \
    "${inactive_gripper_weight}" || exit 1
  require_nonnegative_number "first gripper weight" \
    "${first_gripper_weight}" || exit 1
  local -a window_arguments=()
  if [[ -n "${start_index}${rollout_count}${rollout_stride}" ]]; then
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
    window_arguments=(
      --start-index "${start_index}"
      --count "${rollout_count}"
      --stride "${rollout_stride}"
    )
  fi
  local -a recording_names
  local -a recording_arguments
  local recording_name
  local recording
  IFS=',' read -r -a recording_names <<<"${recording_list}"
  for recording_name in "${recording_names[@]}"; do
    recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
    [[ -f "${recording}/manifest.json" ]] \
      || die "recording does not exist: ${recording_name}"
    sudo chown -R "${USER}:${USER}" "${recording}"
    recording_arguments+=(--recording "${recording}")
  done
  require_runtime
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.train_proposal \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    "${recording_arguments[@]}" \
    --output "${checkpoint_dir}/${proposal_name}.pth" \
    --camera "${camera_name}" \
    --steps "${training_steps}" \
    --hidden-dimension "${hidden_dimension}" \
    --learning-rate "${learning_rate}" \
    --weight-decay "${weight_decay}" \
    --seed "${training_seed}" \
    --goal-consistency-weight "${goal_consistency_weight}" \
    --first-action-weight "${first_action_weight}" \
    --active-direction-weight "${active_direction_weight}" \
    --goal-direction-weight "${goal_direction_weight}" \
    --inactive-gripper-weight "${inactive_gripper_weight}" \
    --first-gripper-weight "${first_gripper_weight}" \
    "${window_arguments[@]}"
}

train_task_action_proposal() {
  local task_name="$1"
  shift
  local -A options=()
  local window_start window_count window_stride
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window "${task_name}")"
  parse_named_options options \
    "recordings steps proposal hidden-dimension learning-rate weight-decay seed goal-consistency-weight first-action-weight active-direction-weight goal-direction-weight inactive-gripper-weight first-gripper-weight" "$@"
  train_action_proposal \
    --recordings "${options[recordings]:-}" \
    --camera wrist \
    --steps "${options[steps]:-3000}" \
    --proposal "${options[proposal]:-}" \
    --hidden-dimension "${options[hidden-dimension]:-256}" \
    --learning-rate "${options[learning-rate]:-0.001}" \
    --weight-decay "${options[weight-decay]:-0.0001}" \
    --seed "${options[seed]:-$(task_proposal_setting "${task_name}" seed)}" \
    --goal-consistency-weight "${options[goal-consistency-weight]:-1.0}" \
    --first-action-weight "${options[first-action-weight]:-1.0}" \
    --active-direction-weight "${options[active-direction-weight]:-0.1}" \
    --goal-direction-weight "${options[goal-direction-weight]:-$(task_proposal_setting "${task_name}" goal_direction)}" \
    --inactive-gripper-weight "${options[inactive-gripper-weight]:-$(task_proposal_setting "${task_name}" inactive_gripper)}" \
    --first-gripper-weight "${options[first-gripper-weight]:-1.0}" \
    --start-index "${window_start}" \
    --count "${window_count}" \
    --stride "${window_stride}"
}

train_grasp_action_proposal() {
  train_task_action_proposal grasp "$@"
}

train_contact_grasp_action_proposal() {
  train_task_action_proposal contact-grasp "$@"
}

train_insertion_action_proposal() {
  train_task_action_proposal insertion "$@"
}

finetune_insertion_transition() {
  local -A options=()
  parse_named_options options \
    "source-session parent proposal steps learning-rate" "$@"
  local source_session="${options[source-session]:-}"
  local parent_name="${options[parent]:-}"
  local proposal_name="${options[proposal]:-}"
  local training_steps="${options[steps]:-500}"
  local learning_rate="${options[learning-rate]:-0.0001}"
  is_safe_identifier "${source_session}" || die "invalid transition source session"
  is_safe_identifier "${parent_name}" || die "invalid transition parent proposal"
  is_safe_identifier "${proposal_name}" || die "invalid transition proposal"
  require_positive_integer "transition training steps" "${training_steps}" || exit 1
  require_nonnegative_number "transition learning rate" "${learning_rate}" || exit 1
  local parent="${checkpoint_dir}/${parent_name}.pth"
  [[ -s "${parent}" ]] || die "transition parent proposal does not exist: ${parent_name}"
  require_runtime
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.insertion_transition_finetune \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --data-root "${control_frame_root}" \
    --parent "${parent}" \
    --source-session "${source_session}" \
    --output "${checkpoint_dir}/${proposal_name}.pth" \
    --steps "${training_steps}" \
    --learning-rate "${learning_rate}"
}

evaluate_insertion_transition() {
  local -A options=()
  parse_named_options options "source-session proposal output" "$@"
  local source_session="${options[source-session]:-}"
  local proposal_name="${options[proposal]:-}"
  local output_name="${options[output]:-}"
  is_safe_identifier "${source_session}" || die "invalid transition evaluation session"
  is_safe_identifier "${proposal_name}" || die "invalid transition evaluation proposal"
  is_safe_identifier "${output_name}" || die "invalid transition evaluation output"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${proposal}" ]] || die "transition proposal does not exist: ${proposal_name}"
  require_runtime
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.insertion_transition_evaluate \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --data-root "${control_frame_root}" \
    --proposal "${proposal}" \
    --source-session "${source_session}" \
    --output "${checkpoint_dir}/experiments/${output_name}.json"
}

evaluate_action_proposal() {
  local -A options=()
  parse_named_options options \
    "recording camera start-index count stride proposal include-stationary" "$@"
  local recording_name="${options[recording]:-}"
  local camera_name="${options[camera]:-wrist}"
  local start_index="${options[start-index]:-0}"
  local rollout_count="${options[count]:-8}"
  local rollout_stride="${options[stride]:-1}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  local include_stationary="${options[include-stationary]:-false}"
  is_safe_identifier "${recording_name}" || die "invalid recording name"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
  [[ "${include_stationary}" == "true" || "${include_stationary}" == "false" ]] \
    || die "include-stationary must be true or false"
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  require_runtime
  sudo chown -R "${USER}:${USER}" "${recording}"
  local -a stationary_arguments=()
  if [[ "${include_stationary}" == "true" ]]; then
    stationary_arguments+=(--include-stationary)
  fi
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.evaluate_proposal \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --proposal "${proposal}" \
    --recording "${recording}" \
    --camera "${camera_name}" \
    --start-index "${start_index}" \
    --count "${rollout_count}" \
    --stride "${rollout_stride}" \
    "${stationary_arguments[@]}"
}

evaluate_task_action_proposal() {
  local task_name="$1"
  shift
  local -A options=()
  local window_start window_count window_stride
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window "${task_name}")"
  parse_named_options options "recording proposal" "$@"
  evaluate_action_proposal \
    --recording "${options[recording]:-}" \
    --camera wrist \
    --start-index "${window_start}" \
    --count "${window_count}" \
    --stride "${window_stride}" \
    --proposal "${options[proposal]:-}" \
    --include-stationary true
}

evaluate_grasp_action_proposal() {
  evaluate_task_action_proposal grasp "$@"
}

evaluate_contact_grasp_action_proposal() {
  evaluate_task_action_proposal contact-grasp "$@"
}

evaluate_insertion_action_proposal() {
  evaluate_task_action_proposal insertion "$@"
}

proposal_evaluation_report_name() {
  local camera_name="$1"
  local proposal_name="$2"
  local start_index="$3"
  local rollout_count="$4"
  local rollout_stride="$5"
  printf '%s_%s_proposal_eval_%06d_%03d_%03d.json' \
    "${camera_name}" "${proposal_name}" "${start_index}" \
    "${rollout_count}" "${rollout_stride}"
}

run_action_proposal_summary() {
  local recording_list="$1"
  local camera_name="$2"
  local start_index="$3"
  local rollout_count="$4"
  local rollout_stride="$5"
  local proposal_name="$6"
  local readiness_module="$7"
  local output_suffix="$8"
  shift 8
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  local -a recording_names
  local -a arguments=()
  local recording_name
  local report_name
  local report
  IFS=',' read -r -a recording_names <<<"${recording_list}"
  report_name="$(proposal_evaluation_report_name \
    "${camera_name}" "${proposal_name}" "${start_index}" \
    "${rollout_count}" "${rollout_stride}")"
  for recording_name in "${recording_names[@]}"; do
    report="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}/jepa_wm/${report_name}"
    [[ -f "${report}" ]] || die "proposal report does not exist: ${report}"
    arguments+=(--evaluation-report "${report}")
  done
  local output="${checkpoint_dir}/experiments/${proposal_name}_${output_suffix}.json"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m "${readiness_module}" \
    --proposal "${proposal}" \
    "${arguments[@]}" \
    "$@" \
    --output "${output}"
}

summarize_action_proposal() {
  local -A options=()
  parse_named_options options \
    "recordings camera start-index count stride proposal" "$@"
  local recording_list="${options[recordings]:-}"
  local camera_name="${options[camera]:-wrist}"
  local start_index="${options[start-index]:-4}"
  local rollout_count="${options[count]:-62}"
  local rollout_stride="${options[stride]:-1}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  run_action_proposal_summary \
    "${recording_list}" "${camera_name}" "${start_index}" \
    "${rollout_count}" "${rollout_stride}" "${proposal_name}" \
    jepa_wm.proposal_readiness readiness
}

summarize_task_action_proposal() {
  local task_name="$1"
  shift
  local -A options=()
  local window_start window_count window_stride
  read -r window_start window_count window_stride \
    <<<"$(task_proposal_window "${task_name}")"
  parse_named_options options "recordings proposal experiment base-seed" "$@"
  local -a task_arguments=()
  if [[ "${task_name}" == "insertion" ]]; then
    local experiment_id="${options[experiment]:-}"
    local base_seed="${options[base-seed]:-}"
    is_safe_identifier "${experiment_id}" || die "invalid experiment ID"
    require_nonnegative_integer "base seed" "${base_seed}" || exit 1
    local roster="${checkpoint_dir}/experiments/${experiment_id}_insertion_corpus.json"
    mkdir -p "$(dirname "${roster}")"
    cd "${repo_dir}"
    "${venv_dir}/bin/python" -m jepa_wm.insertion_corpus create \
      --experiment-id "${experiment_id}" --base-seed "${base_seed}" \
      --output "${roster}"
    task_arguments=(--roster "${roster}")
  fi
  run_action_proposal_summary \
    "${options[recordings]:-}" wrist \
    "${window_start}" "${window_count}" "${window_stride}" \
    "${options[proposal]:-}" \
    "$(task_proposal_setting "${task_name}" readiness_module)" \
    "$(task_proposal_setting "${task_name}" readiness_suffix)" \
    "${task_arguments[@]}"
}

summarize_grasp_action_proposal() {
  summarize_task_action_proposal grasp "$@"
}

summarize_contact_grasp_action_proposal() {
  summarize_task_action_proposal contact-grasp "$@"
}

summarize_insertion_action_proposal() {
  summarize_task_action_proposal insertion "$@"
}

infer_replayed_control() {
  local -A options=()
  parse_named_options options \
    "recording camera context-index observation-id proposal" "$@"
  local recording_name="${options[recording]:-}"
  local camera_name="${options[camera]:-wrist}"
  local context_index="${options[context-index]:-4}"
  local observation_id="${options[observation-id]:-1}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  is_safe_identifier "${recording_name}" || die "invalid recording name"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_nonnegative_integer "context index" "${context_index}" || exit 1
  require_positive_integer "observation ID" "${observation_id}" || exit 1
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  control_worker_status >/dev/null
  require_runtime
  cd "${repo_dir}"
  local request
  local response
  request="$(mktemp "${control_run_dir}/replay-request.XXXXXX.json")"
  "${venv_dir}/bin/python" -m jepa_wm.control_replay \
    --recording "${recording}" \
    --camera "${camera_name}" \
    --context-index "${context_index}" \
    --observation-id "${observation_id}" \
    --proposal "${proposal}" \
    >"${request}"
  response="$("${venv_dir}/bin/python" -m jepa_wm.control_client \
    --socket "${control_socket}" \
    --request "${request}")"
  rm -f "${request}"
  printf '%s\n' "${response}"
}

infer_control_session() {
  local -A options=()
  parse_named_options options "session" "$@"
  local session_id="${options[session]:-}"
  is_safe_identifier "${session_id}" || die "invalid control session"
  local session="${control_frame_root}/control_sessions/${session_id}"
  local request="${session}/request.json"
  [[ -f "${request}" ]] || die "control session request does not exist: ${session_id}"
  [[ ! -e "${session}/response.json" ]] \
    || die "control session already has a response: ${session_id}"
  control_worker_status >/dev/null
  sudo chown -R "${USER}:${USER}" "${session}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_client \
    --socket "${control_socket}" \
    --request "${request}" \
  | "${venv_dir}/bin/python" -m sim.control_response_cli \
      --data-root "${control_frame_root}" \
      --session "${session_id}"
  sudo chmod -R a+rwX "${session}"
}

infer_shadow_session() {
  local -A options=()
  parse_named_options options "session" "$@"
  local session_id="${options[session]:-}"
  is_safe_identifier "${session_id}" || die "invalid control session"
  local session="${control_frame_root}/control_sessions/${session_id}"
  local request="${session}/request.json"
  local direct_response="${session}/response.json"
  local shadow_response="${session}/shadow.json"
  [[ -f "${request}" ]] || die "control session request does not exist: ${session_id}"
  [[ -f "${direct_response}" ]] \
    || die "control session direct response does not exist: ${session_id}"
  [[ ! -e "${shadow_response}" ]] \
    || die "control session already has shadow evidence: ${session_id}"
  control_worker_status >/dev/null
  local artifacts_name
  artifacts_name="$(<"${control_artifacts_file}")"
  is_safe_identifier "${artifacts_name}" || die "invalid resident artifact state"
  sudo chown -R "${USER}:${USER}" "${session}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_client \
    --socket "${control_socket}" \
    --request "${request}" \
    --state "${session}/state.json" \
    --recording-root "${control_frame_root}" \
    --direct-response "${direct_response}" \
    --artifacts "${checkpoint_dir}/${artifacts_name}.worker.json" \
    --shadow-request-output "${session}/shadow_request.json" \
    --shadow-response-output "${shadow_response}"
  sudo chmod -R a+rwX "${session}"
}

persist_baseline_session() {
  local -A options=()
  parse_named_options options "session policy" "$@"
  local session_id="${options[session]:-}"
  local policy="${options[policy]:-}"
  is_safe_identifier "${session_id}" || die "invalid control session"
  [[ "${policy}" == "zero" || "${policy}" == "scripted" ]] \
    || die "baseline policy must be zero or scripted"
  local session="${control_frame_root}/control_sessions/${session_id}"
  [[ -f "${session}/request.json" ]] \
    || die "control session request does not exist: ${session_id}"
  [[ ! -e "${session}/response.json" ]] \
    || die "control session already has a response: ${session_id}"
  sudo chown -R "${USER}:${USER}" "${session}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_baseline_cli \
    --data-root "${control_frame_root}" \
    --session "${session_id}" \
    --policy "${policy}"
  sudo chmod -R a+rwX "${session}"
}

persist_experimental_candidate() {
  local -A options=()
  parse_named_options options "session source-session" "$@"
  local session_id="${options[session]:-}"
  local source_session_id="${options[source-session]:-}"
  is_safe_identifier "${session_id}" || die "invalid candidate control session"
  is_safe_identifier "${source_session_id}" || die "invalid candidate source session"
  local session="${control_frame_root}/control_sessions/${session_id}"
  local source_session="${control_frame_root}/control_sessions/${source_session_id}"
  [[ -f "${session}/request.json" ]] \
    || die "candidate session request does not exist: ${session_id}"
  [[ ! -e "${session}/response.json" ]] \
    || die "candidate session already has a response: ${session_id}"
  [[ -f "${source_session}/shadow.json" \
    && -f "${source_session}/shadow_safety.json" ]] \
    || die "candidate source has no complete shadow evidence: ${source_session_id}"
  sudo chown -R "${USER}:${USER}" "${session}" "${source_session}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.experimental_candidate_cli \
    --data-root "${control_frame_root}" \
    --session "${session_id}" \
    --source-session "${source_session_id}"
  sudo chmod -R a+rwX "${session}"
}

report_control_rollout() {
  local -A options=()
  parse_named_options options \
    "rollout reference seed proposal policy sessions requested-steps orchestration-failure predecessor-session" \
    "$@"
  local rollout_id="${options[rollout]:-}"
  local reference_name="${options[reference]:-}"
  local exploration_seed="${options[seed]:-}"
  local proposal_name="${options[proposal]:-}"
  local policy="${options[policy]:-direct}"
  local sessions="${options[sessions]:-}"
  local requested_steps="${options[requested-steps]:-}"
  local orchestration_failure="${options[orchestration-failure]:-}"
  local predecessor_session="${options[predecessor-session]:-}"
  is_safe_identifier "${rollout_id}" || die "invalid control rollout"
  is_safe_identifier "${reference_name}" || die "invalid reference recording"
  require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  load_control_policy_descriptor "${policy}" "${proposal_name}" \
    || die "invalid control policy"
  [[ "${proposal_name}" == "${CONTROL_POLICY_PROPOSAL}" ]] \
    || die "proposal does not match control policy"
  is_safe_identifier_list "${sessions}" || die "invalid control session list"
  require_positive_integer "requested steps" "${requested_steps}" || exit 1
  local maximum_rollout_steps
  maximum_rollout_steps="$(contact_grasp_maximum_actions \
    "${repo_dir}" "${venv_dir}/bin/python")"
  (( requested_steps <= maximum_rollout_steps )) \
    || die "control rollout exceeds its task-specific action cap"
  local report_dir="${control_frame_root}/control_rollouts/${rollout_id}"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  local -a error_arguments=()
  local -a predecessor_arguments=()
  if [[ "${CONTROL_POLICY_REQUIRES_CHECKPOINT}" == "true" ]]; then
    [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  fi
  if [[ -n "${orchestration_failure}" ]]; then
    error_arguments=(--orchestration-failure "${orchestration_failure}")
  fi
  if [[ -n "${predecessor_session}" ]]; then
    is_safe_identifier "${predecessor_session}" \
      || die "invalid control rollout predecessor"
    predecessor_arguments=(--predecessor-session "${predecessor_session}")
  fi
  sudo install -d -o "${USER}" -g "${USER}" "${report_dir}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_rollout_cli report \
    --data-root "${control_frame_root}" \
    --rollout-id "${rollout_id}" \
    --reference-recording "${reference_name}" \
    --seed "${exploration_seed}" \
    --proposal "${proposal}" \
    --sessions "${sessions}" \
    --requested-steps "${requested_steps}" \
    "${predecessor_arguments[@]}" \
    "${error_arguments[@]}" \
    --output "${report_dir}/report.json"
}

report_candidate_trial() {
  local -A options=()
  parse_named_options options \
    "experiment baseline-experiment candidate-session source-session" "$@"
  local experiment_id="${options[experiment]:-}"
  local baseline_experiment_id="${options[baseline-experiment]:-}"
  local candidate_session_id="${options[candidate-session]:-}"
  local source_session_id="${options[source-session]:-}"
  for identifier in \
    "${experiment_id}" "${baseline_experiment_id}" \
    "${candidate_session_id}" "${source_session_id}"; do
    is_safe_identifier "${identifier}" || die "invalid candidate trial identifier"
  done
  local report_dir="${control_frame_root}/control_candidates/${experiment_id}"
  sudo install -d -o "${USER}" -g "${USER}" "${report_dir}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.candidate_trial_cli \
    --data-root "${control_frame_root}" \
    --experiment-id "${experiment_id}" \
    --baseline-experiment-id "${baseline_experiment_id}" \
    --candidate-session "${candidate_session_id}" \
    --source-session "${source_session_id}" \
    --output "${report_dir}/report.json"
}

summarize_candidate_trials() {
  local -A options=()
  parse_named_options options "experiments output" "$@"
  local experiments="${options[experiments]:-}"
  local output_name="${options[output]:-}"
  is_safe_identifier_list "${experiments}" \
    || die "invalid candidate readiness experiment list"
  is_safe_identifier "${output_name}" \
    || die "invalid candidate readiness output name"
  local -a experiment_ids
  local -a experiment_arguments=()
  local experiment_id
  IFS=',' read -r -a experiment_ids <<<"${experiments}"
  for experiment_id in "${experiment_ids[@]}"; do
    [[ -f "${control_frame_root}/control_candidates/${experiment_id}/report.json" ]] \
      || die "candidate trial report does not exist: ${experiment_id}"
    experiment_arguments+=(--experiment "${experiment_id}")
  done
  local report_dir="${control_frame_root}/control_candidates/${output_name}"
  sudo install -d -o "${USER}" -g "${USER}" "${report_dir}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.candidate_readiness_cli \
    --data-root "${control_frame_root}" \
    "${experiment_arguments[@]}" \
    --output "${report_dir}/readiness.json"
}

calibrate_control_objective() {
  local -A options=()
  parse_named_options options "sessions output" "$@"
  local sessions="${options[sessions]:-}"
  local output_name="${options[output]:-}"
  is_safe_identifier_list "${sessions}" || die "invalid calibration session list"
  is_safe_identifier "${output_name}" || die "invalid calibration output name"
  local -a session_ids
  local -a session_arguments=()
  local session_id
  IFS=',' read -r -a session_ids <<<"${sessions}"
  (( ${#session_ids[@]} >= 3 )) \
    || die "objective calibration requires at least three sessions"
  for session_id in "${session_ids[@]}"; do
    [[ -f "${control_frame_root}/control_sessions/${session_id}/result.json" ]] \
      || die "candidate session result does not exist: ${session_id}"
    session_arguments+=(--session "${session_id}")
  done
  local output="${checkpoint_dir}/${output_name}.json"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.calibrate_objective \
    --data-root "${control_frame_root}" \
    "${session_arguments[@]}" \
    --output "${output}"
}

configure_control_worker() {
  local -A options=()
  parse_named_options options \
    "name proposal adapter calibration translation-margin rotation-margin gripper-margin planner-seed planner-iterations planner-samples planner-elites" \
    "$@"
  local name="${options[name]:-}"
  local proposal_name="${options[proposal]:-}"
  local adapter_name="${options[adapter]:-}"
  local calibration_name="${options[calibration]:-none}"
  local translation_margin="${options[translation-margin]:-}"
  local rotation_margin="${options[rotation-margin]:-}"
  local gripper_margin="${options[gripper-margin]:-}"
  local planner_seed="${options[planner-seed]:-}"
  local planner_iterations="${options[planner-iterations]:-}"
  local planner_samples="${options[planner-samples]:-}"
  local planner_elites="${options[planner-elites]:-}"
  for identifier in "${name}" "${proposal_name}" "${adapter_name}" "${calibration_name}"; do
    is_safe_identifier "${identifier}" || die "invalid worker artifact identifier"
  done
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  [[ -s "${adapter}" ]] || die "action adapter does not exist: ${adapter_name}"
  local -a calibration_arguments=()
  local -a margin_arguments=()
  local -a planner_arguments=()
  if [[ "${calibration_name}" != "none" ]]; then
    local calibration="${checkpoint_dir}/${calibration_name}.json"
    [[ -s "${calibration}" ]] \
      || die "action-response calibration does not exist: ${calibration_name}"
    calibration_arguments=(--calibration "${calibration}")
  fi
  if cem_settings_requested \
    "${planner_seed}" "${planner_iterations}" "${planner_samples}" "${planner_elites}"; then
    validate_cem_settings \
      "${planner_seed}" "${planner_iterations}" "${planner_samples}" "${planner_elites}" \
      || exit 1
    planner_arguments=(
      --planner-seed "${planner_seed}"
      --planner-iterations "${planner_iterations}"
      --planner-samples "${planner_samples}"
      --planner-elites "${planner_elites}"
    )
  fi
  if [[ -n "${translation_margin}${rotation_margin}${gripper_margin}" ]]; then
    [[ "${calibration_name}" != "none" ]] \
      || die "progress margins require a calibration"
    [[ -n "${translation_margin}" && -n "${rotation_margin}" && -n "${gripper_margin}" ]] \
      || die "all three progress margins must be provided together"
    require_nonnegative_number "translation margin" "${translation_margin}" || exit 1
    require_nonnegative_number "rotation margin" "${rotation_margin}" || exit 1
    require_nonnegative_number "gripper margin" "${gripper_margin}" || exit 1
    margin_arguments=(
      --translation-margin "${translation_margin}"
      --rotation-margin "${rotation_margin}"
      --gripper-margin "${gripper_margin}"
    )
  fi
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.worker_artifacts write \
    --output "${checkpoint_dir}/${name}.worker.json" \
    --proposal "${proposal}" \
    --adapter "${adapter}" \
    "${calibration_arguments[@]}" \
    "${margin_arguments[@]}" \
    "${planner_arguments[@]}"
}

rebase_control_worker_proposal() {
  local -A options=()
  parse_named_options options "source name proposal" "$@"
  local source_name="${options[source]:-}"
  local name="${options[name]:-}"
  local proposal_name="${options[proposal]:-}"
  for identifier in "${source_name}" "${name}" "${proposal_name}"; do
    is_safe_identifier "${identifier}" || die "invalid worker artifact identifier"
  done
  local source_manifest="${checkpoint_dir}/${source_name}.worker.json"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${source_manifest}" ]] || die "source worker manifest does not exist"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.worker_artifacts replace-proposal \
    --source "${source_manifest}" \
    --output "${checkpoint_dir}/${name}.worker.json" \
    --proposal "${proposal}"
}

rollout_session_list() {
  local rollout_id="$1"
  local step_count="$2"
  local sessions=""
  local index
  local suffix
  for (( index = 0; index < step_count; index++ )); do
    printf -v suffix '%02d' "${index}"
    sessions+="${sessions:+,}${rollout_id}-${suffix}"
  done
  printf '%s\n' "${sessions}"
}

report_control_baselines() {
  local -A options=()
  parse_named_options options \
    "experiment reference seed requested-steps direct-rollout direct-sessions zero-rollout scripted-rollout direct-proposal" \
    "$@"
  local experiment_id="${options[experiment]:-}"
  local reference_name="${options[reference]:-}"
  local exploration_seed="${options[seed]:-}"
  local requested_steps="${options[requested-steps]:-}"
  local direct_rollout="${options[direct-rollout]:-}"
  local direct_sessions="${options[direct-sessions]:-}"
  local zero_rollout="${options[zero-rollout]:-}"
  local scripted_rollout="${options[scripted-rollout]:-}"
  local direct_proposal_name="${options[direct-proposal]:-}"
  for identifier in \
    "${experiment_id}" "${reference_name}" "${direct_rollout}" \
    "${zero_rollout}" "${scripted_rollout}" "${direct_proposal_name}"; do
    is_safe_identifier "${identifier}" || die "invalid baseline comparison identifier"
  done
  require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
  require_positive_integer "requested steps" "${requested_steps}" || exit 1
  (( requested_steps <= 8 )) || die "control rollout is capped at eight steps"
  if [[ -z "${direct_sessions}" ]]; then
    direct_sessions="$(rollout_session_list "${direct_rollout}" "${requested_steps}")"
  else
    is_safe_identifier_list "${direct_sessions}" \
      || die "invalid direct baseline session list"
  fi
  local direct_proposal="${checkpoint_dir}/${direct_proposal_name}.pth"
  [[ -s "${direct_proposal}" ]] \
    || die "action proposal does not exist: ${direct_proposal_name}"
  local report_dir="${control_frame_root}/control_baselines/${experiment_id}"
  sudo install -d -o "${USER}" -g "${USER}" "${report_dir}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_baseline_report_cli \
    --data-root "${control_frame_root}" \
    --experiment-id "${experiment_id}" \
    --reference-recording "${reference_name}" \
    --seed "${exploration_seed}" \
    --requested-steps "${requested_steps}" \
    --direct-rollout "${direct_rollout}" \
    --direct-sessions "${direct_sessions}" \
    --direct-proposal "${direct_proposal}" \
    --zero-rollout "${zero_rollout}" \
    --zero-sessions "$(rollout_session_list "${zero_rollout}" "${requested_steps}")" \
    --zero-proposal "${checkpoint_dir}/baseline_zero.pth" \
    --scripted-rollout "${scripted_rollout}" \
    --scripted-sessions "$(rollout_session_list "${scripted_rollout}" "${requested_steps}")" \
    --scripted-proposal "${checkpoint_dir}/baseline_scripted.pth" \
    --output "${report_dir}/report.json"
}

summarize_grasp_control_readiness() {
  local -A options=()
  parse_named_options options "experiment baseline-experiments" "$@"
  local experiment_id="${options[experiment]:-}"
  local baseline_experiments="${options[baseline-experiments]:-}"
  is_safe_identifier "${experiment_id}" \
    || die "invalid grasp control readiness identifier"
  is_safe_identifier_list "${baseline_experiments}" \
    || die "invalid grasp baseline experiment list"
  local output_dir="${control_frame_root}/control_readiness/${experiment_id}"
  sudo install -d -o "${USER}" -g "${USER}" "${output_dir}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.grasp_control_readiness_cli \
    --data-root "${control_frame_root}" \
    --baseline-experiments "${baseline_experiments}" \
    --output "${output_dir}/readiness.json"
}

control_worker_is_running() {
  [[ -S "${control_socket}" ]] \
    && [[ -f "${control_pid_file}" ]] \
    && [[ -f "${control_artifacts_file}" ]] \
    && [[ "$(<"${control_pid_file}")" =~ ^[0-9]+$ ]] \
    && kill -0 "$(<"${control_pid_file}")" 2>/dev/null
}

control_worker_status() {
  control_worker_is_running || die "control worker is not ready"
  local worker_pid
  worker_pid="$(<"${control_pid_file}")"
  printf 'ready pid=%s artifacts=%s socket=%s\n' \
    "${worker_pid}" "$(<"${control_artifacts_file}")" \
    "${control_socket}"
}

stop_control_worker() {
  if [[ ! -f "${control_pid_file}" ]]; then
    rm -f "${control_socket}" "${control_artifacts_file}"
    printf 'control worker is already stopped\n'
    return
  fi
  local worker_pid
  worker_pid="$(<"${control_pid_file}")"
  require_positive_integer "control worker PID" "${worker_pid}" || exit 1
  if kill -0 "${worker_pid}" 2>/dev/null; then
    tr '\0' ' ' <"/proc/${worker_pid}/cmdline" \
      | grep -Fq 'jepa_wm.control_server' \
      || die "refusing to stop an unexpected process: ${worker_pid}"
    kill "${worker_pid}"
    for _ in {1..20}; do
      kill -0 "${worker_pid}" 2>/dev/null || break
      sleep 0.25
    done
  fi
  rm -f "${control_pid_file}" "${control_socket}" "${control_artifacts_file}"
  printf 'control worker stopped\n'
}

start_control_worker() {
  local -A options=()
  parse_named_options options "artifacts" "$@"
  local artifacts_name="${options[artifacts]:-quantis_wrist_control}"
  is_safe_identifier "${artifacts_name}" || die "invalid worker artifact name"
  local artifacts="${checkpoint_dir}/${artifacts_name}.worker.json"
  [[ -s "${artifacts}" ]] \
    || die "control worker artifact manifest does not exist: ${artifacts_name}"
  require_runtime
  mkdir -p "${control_run_dir}" "$(dirname "${control_log}")"
  if [[ -f "${control_pid_file}" ]] \
    && [[ "$(<"${control_pid_file}")" =~ ^[0-9]+$ ]] \
    && kill -0 "$(<"${control_pid_file}")" 2>/dev/null \
    && ! control_worker_is_running; then
    die "control worker is running with incomplete state; stop it before restart"
  fi
  if control_worker_is_running; then
    [[ "$(<"${control_artifacts_file}")" == "${artifacts_name}" ]] \
      || die "control worker is already running with another artifact manifest"
    control_worker_status
    return
  fi
  rm -f "${control_socket}" "${control_pid_file}" "${control_artifacts_file}"
  cd "${repo_dir}"
  nohup "${venv_dir}/bin/python" -m jepa_wm.control_server \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --artifacts "${artifacts}" \
    --socket "${control_socket}" \
    --frame-root "${control_frame_root}" \
    >"${control_log}" 2>&1 &
  local worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${control_pid_file}"
  printf '%s\n' "${artifacts_name}" >"${control_artifacts_file}"
  for _ in {1..60}; do
    if [[ -S "${control_socket}" ]]; then
      control_worker_status
      return
    fi
    kill -0 "${worker_pid}" 2>/dev/null \
      || die "control worker exited during startup; inspect ${control_log}"
    sleep 1
  done
  die "control worker did not become ready; inspect ${control_log}"
}

summarize_experiment() {
  local experiment_id="$1"
  local training_list="$2"
  local held_out_list="$3"
  local camera_name="$4"
  local rollout_count="$5"
  is_safe_identifier "${experiment_id}" || die "invalid experiment name"
  is_safe_identifier_list "${training_list}" || die "invalid training list"
  is_safe_identifier_list "${held_out_list}" || die "invalid held-out list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_runtime

  local -a training_names
  local -a held_out_names
  local -a arguments
  local recording_name
  local report_name
  local report
  IFS=',' read -r -a training_names <<<"${training_list}"
  IFS=',' read -r -a held_out_names <<<"${held_out_list}"
  for recording_name in "${training_names[@]}"; do
    arguments+=(
      --training-recording
      "${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
    )
  done
  printf -v report_name '%s_adapted_rollout_eval_000000_%03d.json' \
    "${camera_name}" "${rollout_count}"
  for recording_name in "${held_out_names[@]}"; do
    report="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}/jepa_wm/${report_name}"
    [[ -f "${report}" ]] || die "held-out report does not exist: ${report}"
    arguments+=(--held-out-report "${report}")
  done
  local output="${checkpoint_dir}/experiments/${experiment_id}.json"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.experiment \
    --experiment-id "${experiment_id}" \
    "${arguments[@]}" \
    --output "${output}"
}

case "${1:-}" in
  install)
    install_runtime
    ;;
  smoke)
    smoke_runtime
    ;;
  status)
    status_runtime
    ;;
  evaluate)
    evaluate_recording \
      "${2:-}" "${3:-wrist}" "${4:-0}" "${5:-8}" "${6:-1}" \
      "${7:-base}" "${8:-}"
    ;;
  adapt)
    adapt_recording_set "${2:-}" "${3:-wrist}" "${4:-100}"
    ;;
  adapt-set)
    adapt_recording_set \
      "${2:-}" "${3:-wrist}" "${4:-500}" "${5:-}" \
      "${6:-}" "${7:-}" "${8:-}"
    ;;
  insertion-wm-adapt)
    adapt_insertion_world_model "${@:2}"
    ;;
  insertion-wm-eval)
    evaluate_insertion_world_model "${@:2}"
    ;;
  action-conditioning-experiment)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" -m jepa_wm.action_conditioning_experiment "${@:2}"
    ;;
  action-routing-experiment)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" -m jepa_wm.action_routing_experiment "${@:2}"
    ;;
  observed-context-routing-experiment)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" \
      -m jepa_wm.observed_context_routing_experiment "${@:2}"
    ;;
  causal-context-routing-probe)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" \
      -m jepa_wm.causal_context_routing_experiment "${@:2}"
    ;;
  physical-state-routing-probe)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" \
      -m jepa_wm.physical_state_routing_experiment "${@:2}"
    ;;
  physical-state-residual-experiment)
    require_runtime
    cd "${repo_dir}"
    "${venv_dir}/bin/python" \
      -m jepa_wm.physical_state_residual_experiment "${@:2}"
    ;;
  plan-benchmark)
    benchmark_planner "${@:2}"
    ;;
  insertion-plan-benchmark)
    benchmark_insertion_planner "${@:2}"
    ;;
  insertion-plan-summarize)
    summarize_insertion_planner "${@:2}"
    ;;
  insertion-proposal-training-diagnostic)
    diagnose_insertion_proposal_training "${@:2}"
    ;;
  proposal-train)
    train_action_proposal "${@:2}"
    ;;
  grasp-proposal-train)
    train_grasp_action_proposal "${@:2}"
    ;;
  contact-grasp-proposal-train)
    train_contact_grasp_action_proposal "${@:2}"
    ;;
  insertion-proposal-train)
    train_insertion_action_proposal "${@:2}"
    ;;
  insertion-transition-finetune)
    finetune_insertion_transition "${@:2}"
    ;;
  insertion-transition-eval)
    evaluate_insertion_transition "${@:2}"
    ;;
  insertion-transition-handoff)
    previous_session_id="${2:-}"
    parent_proposal="${3:-}"
    output_session_id="${4:-}"
    for identifier in \
      "${previous_session_id}" "${parent_proposal}" "${output_session_id}"; do
      is_safe_identifier "${identifier}" \
        || die "invalid insertion transition handoff identifier"
    done
    "${venv_dir}/bin/python" -m jepa_wm.insertion_transition \
      --previous-request "${control_frame_root}/control_sessions/${previous_session_id}/request.json" \
      --previous-response "${control_frame_root}/control_sessions/${previous_session_id}/response.json" \
      --parent "${checkpoint_dir}/${parent_proposal}.pth" \
      --data-root "${control_frame_root}" \
      --previous-session "${previous_session_id}" \
      | base64 | tr -d '\n'
    ;;
  proposal-eval)
    evaluate_action_proposal "${@:2}"
    ;;
  grasp-proposal-eval)
    evaluate_grasp_action_proposal "${@:2}"
    ;;
  contact-grasp-proposal-eval)
    evaluate_contact_grasp_action_proposal "${@:2}"
    ;;
  insertion-proposal-eval)
    evaluate_insertion_action_proposal "${@:2}"
    ;;
  proposal-summarize)
    summarize_action_proposal "${@:2}"
    ;;
  grasp-proposal-summarize)
    summarize_grasp_action_proposal "${@:2}"
    ;;
  contact-grasp-proposal-summarize)
    summarize_contact_grasp_action_proposal "${@:2}"
    ;;
  insertion-proposal-summarize)
    summarize_insertion_action_proposal "${@:2}"
    ;;
  insertion-wm-summarize)
    summarize_insertion_world_model "${@:2}"
    ;;
  control-infer-replay)
    infer_replayed_control "${@:2}"
    ;;
  control-infer-session)
    infer_control_session "${@:2}"
    ;;
  control-shadow-session)
    infer_shadow_session "${@:2}"
    ;;
  control-baseline-session)
    persist_baseline_session "${@:2}"
    ;;
  control-candidate-session)
    persist_experimental_candidate "${@:2}"
    ;;
  control-rollout-report)
    report_control_rollout "${@:2}"
    ;;
  control-baseline-report)
    report_control_baselines "${@:2}"
    ;;
  grasp-control-summarize)
    summarize_grasp_control_readiness "${@:2}"
    ;;
  control-candidate-report)
    report_candidate_trial "${@:2}"
    ;;
  control-candidate-summarize)
    summarize_candidate_trials "${@:2}"
    ;;
  control-objective-calibrate)
    calibrate_control_objective "${@:2}"
    ;;
  control-worker-configure)
    configure_control_worker "${@:2}"
    ;;
  control-worker-rebase-proposal)
    rebase_control_worker_proposal "${@:2}"
    ;;
  control-worker-start)
    start_control_worker "${@:2}"
    ;;
  control-worker-status)
    control_worker_status
    ;;
  control-worker-stop)
    stop_control_worker
    ;;
  summarize)
    summarize_experiment \
      "${2:-}" "${3:-}" "${4:-}" "${5:-wrist}" "${6:-40}"
    ;;
  *)
    die "expected install, smoke, status, evaluate, adapt, adapt-set, action-conditioning-experiment, action-routing-experiment, observed-context-routing-experiment, causal-context-routing-probe, physical-state-routing-probe, physical-state-residual-experiment, plan-benchmark, insertion-plan-benchmark, insertion-plan-summarize, insertion-proposal-training-diagnostic, proposal-train, grasp-proposal-train, contact-grasp-proposal-train, insertion-proposal-train, proposal-eval, grasp-proposal-eval, contact-grasp-proposal-eval, insertion-proposal-eval, proposal-summarize, grasp-proposal-summarize, contact-grasp-proposal-summarize, insertion-proposal-summarize, insertion-wm-summarize, control-worker-configure, control-worker-rebase-proposal, control-worker-start, control-worker-status, control-worker-stop, control-infer-replay, control-infer-session, control-shadow-session, control-baseline-session, control-candidate-session, control-rollout-report, control-baseline-report, grasp-control-summarize, control-candidate-report, control-candidate-summarize, control-objective-calibrate, or summarize"
    ;;
esac
