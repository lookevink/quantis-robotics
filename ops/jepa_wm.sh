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
  is_safe_identifier_list "${recording_list}" || die "invalid recording list"
  is_safe_identifier "${camera_name}" || die "invalid camera name"
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
  local adapter="${checkpoint_dir}/quantis_isaac_${camera_name}_action_adapter.pth"
  cd "${repo_dir}"
  "${venv_dir}/bin/python" -m jepa_wm.adapt_recording \
    --source "${source_dir}" \
    --checkpoint "${jepa_checkpoint}" \
    "${recording_arguments[@]}" \
    --output "${adapter}" \
    --camera "${camera_name}" \
    --steps "${training_steps}"
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
    adapt_recording_set "${2:-}" "${3:-wrist}" "${4:-500}"
    ;;
  summarize)
    summarize_experiment \
      "${2:-}" "${3:-}" "${4:-}" "${5:-wrist}" "${6:-40}"
    ;;
  *)
    die "expected install, smoke, status, evaluate, adapt, adapt-set, or summarize"
    ;;
esac
