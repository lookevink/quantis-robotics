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
control_proposal_file="${control_run_dir}/control.proposal"
control_adapter_file="${control_run_dir}/control.adapter"
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
  is_safe_identifier "${recording_name}" \
    || die "invalid recording name"
  is_safe_identifier "${camera_name}" \
    || die "invalid camera name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "transition count" "${transition_count}" || exit 1
  require_positive_integer "transition stride" "${transition_stride}" || exit 1
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
  if [[ "${adapter_mode}" == "adapted" ]]; then
    local adapter="${checkpoint_dir}/quantis_isaac_${camera_name}_action_adapter.pth"
    [[ -s "${adapter}" ]] || die "action adapter is not installed for ${camera_name}"
    arguments+=(--adapter "${adapter}")
  elif [[ "${adapter_mode}" != "base" ]]; then
    die "evaluation mode must be base or adapted"
  fi
  "${venv_dir}/bin/python" "${arguments[@]}"
}

adapt_recording_set() {
  local recording_list="$1"
  local camera_name="$2"
  local training_steps="$3"
  local adapter_name="${4:-quantis_isaac_${camera_name}_action_adapter}"
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  require_positive_integer "training steps" "${training_steps}" || exit 1
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
    --steps "${training_steps}"
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

train_action_proposal() {
  local -A options=()
  parse_named_options options "recordings camera steps proposal" "$@"
  local recording_list="${options[recordings]:-}"
  local camera_name="${options[camera]:-wrist}"
  local training_steps="${options[steps]:-2000}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_positive_integer "training steps" "${training_steps}" || exit 1
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
    --steps "${training_steps}"
}

evaluate_action_proposal() {
  local -A options=()
  parse_named_options options \
    "recording camera start-index count stride proposal" "$@"
  local recording_name="${options[recording]:-}"
  local camera_name="${options[camera]:-wrist}"
  local start_index="${options[start-index]:-0}"
  local rollout_count="${options[count]:-8}"
  local rollout_stride="${options[stride]:-1}"
  local proposal_name="${options[proposal]:-quantis_isaac_${camera_name}_action_proposal}"
  is_safe_identifier "${recording_name}" || die "invalid recording name"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
  local recording="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -f "${recording}/manifest.json" ]] \
    || die "recording does not exist: ${recording_name}"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  require_runtime
  sudo chown -R "${USER}:${USER}" "${recording}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.evaluate_proposal \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --proposal "${proposal}" \
    --recording "${recording}" \
    --camera "${camera_name}" \
    --start-index "${start_index}" \
    --count "${rollout_count}" \
    --stride "${rollout_stride}"
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
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  require_nonnegative_integer "start index" "${start_index}" || exit 1
  require_positive_integer "rollout count" "${rollout_count}" || exit 1
  require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  local -a recording_names
  local -a arguments
  local recording_name
  local report_name
  local report
  IFS=',' read -r -a recording_names <<<"${recording_list}"
  printf -v report_name '%s_%s_proposal_eval_%06d_%03d_%03d.json' \
    "${camera_name}" "${proposal_name}" "${start_index}" \
    "${rollout_count}" "${rollout_stride}"
  for recording_name in "${recording_names[@]}"; do
    report="${HOME}/docker/isaac-sim/data/quantis/recordings/${recording_name}/jepa_wm/${report_name}"
    [[ -f "${report}" ]] || die "proposal report does not exist: ${report}"
    arguments+=(--evaluation-report "${report}")
  done
  local output="${checkpoint_dir}/experiments/${proposal_name}_readiness.json"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.proposal_readiness \
    --proposal "${proposal}" \
    "${arguments[@]}" \
    --output "${output}"
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
  local adapter_name
  [[ -f "${request}" ]] || die "control session request does not exist: ${session_id}"
  [[ -f "${direct_response}" ]] \
    || die "control session direct response does not exist: ${session_id}"
  [[ ! -e "${shadow_response}" ]] \
    || die "control session already has shadow evidence: ${session_id}"
  control_worker_status >/dev/null
  adapter_name="$(<"${control_adapter_file}")"
  is_safe_identifier "${adapter_name}" || die "invalid resident adapter state"
  sudo chown -R "${USER}:${USER}" "${session}"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.control_client \
    --socket "${control_socket}" \
    --request "${request}" \
    --direct-response "${direct_response}" \
    --adapter "${checkpoint_dir}/${adapter_name}.pth" \
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

report_control_rollout() {
  local -A options=()
  parse_named_options options \
    "rollout reference seed proposal sessions requested-steps orchestration-failure" \
    "$@"
  local rollout_id="${options[rollout]:-}"
  local reference_name="${options[reference]:-}"
  local exploration_seed="${options[seed]:-}"
  local proposal_name="${options[proposal]:-}"
  local sessions="${options[sessions]:-}"
  local requested_steps="${options[requested-steps]:-}"
  local orchestration_failure="${options[orchestration-failure]:-}"
  is_safe_identifier "${rollout_id}" || die "invalid control rollout"
  is_safe_identifier "${reference_name}" || die "invalid reference recording"
  require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  is_safe_identifier_list "${sessions}" || die "invalid control session list"
  require_positive_integer "requested steps" "${requested_steps}" || exit 1
  (( requested_steps <= 8 )) || die "control rollout is capped at eight steps"
  local report_dir="${control_frame_root}/control_rollouts/${rollout_id}"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  local -a error_arguments=()
  if [[ "${proposal_name}" != "baseline_zero" \
    && "${proposal_name}" != "baseline_scripted" ]]; then
    [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  fi
  if [[ -n "${orchestration_failure}" ]]; then
    error_arguments=(--orchestration-failure "${orchestration_failure}")
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
    "${error_arguments[@]}" \
    --output "${report_dir}/report.json"
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
    "experiment reference seed requested-steps direct-rollout zero-rollout scripted-rollout direct-proposal" \
    "$@"
  local experiment_id="${options[experiment]:-}"
  local reference_name="${options[reference]:-}"
  local exploration_seed="${options[seed]:-}"
  local requested_steps="${options[requested-steps]:-}"
  local direct_rollout="${options[direct-rollout]:-}"
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
    --direct-sessions "$(rollout_session_list "${direct_rollout}" "${requested_steps}")" \
    --direct-proposal "${direct_proposal}" \
    --zero-rollout "${zero_rollout}" \
    --zero-sessions "$(rollout_session_list "${zero_rollout}" "${requested_steps}")" \
    --zero-proposal "${checkpoint_dir}/baseline_zero.pth" \
    --scripted-rollout "${scripted_rollout}" \
    --scripted-sessions "$(rollout_session_list "${scripted_rollout}" "${requested_steps}")" \
    --scripted-proposal "${checkpoint_dir}/baseline_scripted.pth" \
    --output "${report_dir}/report.json"
}

control_worker_is_running() {
  [[ -S "${control_socket}" ]] \
    && [[ -f "${control_pid_file}" ]] \
    && [[ -f "${control_proposal_file}" ]] \
    && [[ -f "${control_adapter_file}" ]] \
    && [[ "$(<"${control_pid_file}")" =~ ^[0-9]+$ ]] \
    && kill -0 "$(<"${control_pid_file}")" 2>/dev/null
}

control_worker_status() {
  control_worker_is_running || die "control worker is not ready"
  local worker_pid
  worker_pid="$(<"${control_pid_file}")"
  printf 'ready pid=%s proposal=%s adapter=%s socket=%s\n' \
    "${worker_pid}" "$(<"${control_proposal_file}")" \
    "$(<"${control_adapter_file}")" "${control_socket}"
}

stop_control_worker() {
  if [[ ! -f "${control_pid_file}" ]]; then
    rm -f "${control_socket}" "${control_proposal_file}" "${control_adapter_file}"
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
  rm -f \
    "${control_pid_file}" "${control_socket}" \
    "${control_proposal_file}" "${control_adapter_file}"
  printf 'control worker stopped\n'
}

start_control_worker() {
  local -A options=()
  parse_named_options options "proposal adapter" "$@"
  local proposal_name="${options[proposal]:-quantis_isaac_wrist_action_proposal}"
  local adapter_name="${options[adapter]:-quantis_isaac_wrist_action_adapter}"
  is_safe_identifier "${proposal_name}" || die "invalid proposal name"
  is_safe_identifier "${adapter_name}" || die "invalid adapter name"
  local proposal="${checkpoint_dir}/${proposal_name}.pth"
  local adapter="${checkpoint_dir}/${adapter_name}.pth"
  [[ -s "${proposal}" ]] || die "action proposal does not exist: ${proposal_name}"
  [[ -s "${adapter}" ]] || die "action adapter does not exist: ${adapter_name}"
  require_runtime
  mkdir -p "${control_run_dir}" "$(dirname "${control_log}")"
  if [[ -f "${control_pid_file}" ]] \
    && [[ "$(<"${control_pid_file}")" =~ ^[0-9]+$ ]] \
    && kill -0 "$(<"${control_pid_file}")" 2>/dev/null \
    && ! control_worker_is_running; then
    die "control worker is running with incomplete state; stop it before restart"
  fi
  if control_worker_is_running; then
    [[ "$(<"${control_proposal_file}")" == "${proposal_name}" ]] \
      || die "control worker is already running with another proposal"
    [[ "$(<"${control_adapter_file}")" == "${adapter_name}" ]] \
      || die "control worker is already running with another adapter"
    control_worker_status
    return
  fi
  rm -f \
    "${control_socket}" "${control_pid_file}" \
    "${control_proposal_file}" "${control_adapter_file}"
  cd "${repo_dir}"
  nohup "${venv_dir}/bin/python" -m jepa_wm.control_server \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    --proposal "${proposal}" \
    --adapter "${adapter}" \
    --socket "${control_socket}" \
    --frame-root "${control_frame_root}" \
    >"${control_log}" 2>&1 &
  local worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${control_pid_file}"
  printf '%s\n' "${proposal_name}" >"${control_proposal_file}"
  printf '%s\n' "${adapter_name}" >"${control_adapter_file}"
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
      "${2:-}" "${3:-wrist}" "${4:-0}" "${5:-8}" "${6:-1}" "${7:-base}"
    ;;
  adapt)
    adapt_recording_set "${2:-}" "${3:-wrist}" "${4:-100}"
    ;;
  adapt-set)
    adapt_recording_set "${2:-}" "${3:-wrist}" "${4:-500}" "${5:-}"
    ;;
  plan-benchmark)
    benchmark_planner "${@:2}"
    ;;
  proposal-train)
    train_action_proposal "${@:2}"
    ;;
  proposal-eval)
    evaluate_action_proposal "${@:2}"
    ;;
  proposal-summarize)
    summarize_action_proposal "${@:2}"
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
  control-rollout-report)
    report_control_rollout "${@:2}"
    ;;
  control-baseline-report)
    report_control_baselines "${@:2}"
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
    die "expected install, smoke, status, evaluate, adapt, adapt-set, plan-benchmark, proposal-train, proposal-eval, proposal-summarize, control-worker-start, control-worker-status, control-worker-stop, control-infer-replay, control-infer-session, control-shadow-session, control-baseline-session, control-rollout-report, control-baseline-report, or summarize"
    ;;
esac
