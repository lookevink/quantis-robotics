#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=isaac_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/isaac_common.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: nvidia-smi is unavailable; use an RTX-capable EC2 GPU image\n' >&2
  exit 1
fi

required_packages=(ca-certificates curl gnupg python3-venv unzip)
if ! command -v docker >/dev/null 2>&1; then
  required_packages+=(docker.io)
fi
missing_packages=()
for package in "${required_packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
    missing_packages+=("${package}")
  fi
done
if (( ${#missing_packages[@]} )); then
  sudo apt-get update
  sudo apt-get install -y "${missing_packages[@]}"
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl --fail --silent --show-error --location \
    https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
fi

if ! sudo docker info >/dev/null 2>&1; then
  sudo systemctl enable --now docker
fi
sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
sudo usermod -aG docker "${USER}"
sudo systemctl restart docker

if ! sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
  printf 'error: Docker cannot access the GPU. Install/configure NVIDIA Container Toolkit, then rerun.\n' >&2
  exit 1
fi

signal_port="${ISAAC_SIGNAL_PORT:-49100}"
stream_port="${ISAAC_STREAM_PORT:-47998}"
if command -v ufw >/dev/null 2>&1 && sudo ufw status | grep -q '^Status: active'; then
  sudo ufw allow "${signal_port}/tcp" comment 'Quantis Isaac Sim WebRTC signaling'
  sudo ufw allow "${stream_port}/udp" comment 'Quantis Isaac Sim WebRTC media'
fi

mkdir -p \
  "${isaac_home}/cache/kit" \
  "${isaac_home}/cache/ov" \
  "${isaac_home}/cache/pip" \
  "${isaac_home}/cache/glcache" \
  "${isaac_home}/cache/computecache" \
  "${isaac_home}/data" \
  "${isaac_home}/logs" \
  "${isaac_home}/documents" \
  "${asset_home}/archives" \
  "${asset_home}/datacenter" \
  "${asset_home}/datasets" \
  "${asset_home}/cable" \
  "${HOME}/quantis-robotics/data/episodes"

sudo chown -R "${USER}:${USER}" "${isaac_home}" "${asset_home}" "${HOME}/quantis-robotics/data"

datacenter_url="https://d4i3qtqj3r0z5.cloudfront.net/Datacenter_NVD%4010012.zip"
datacenter_zip="${asset_home}/archives/Datacenter_NVD@10012.zip"
datacenter_marker="${asset_home}/datacenter/.extracted"
if [[ ! -f "${datacenter_marker}" ]]; then
  curl --fail --location --retry 3 --continue-at - \
    --output "${datacenter_zip}.part" "${datacenter_url}"
  mv "${datacenter_zip}.part" "${datacenter_zip}"
  unzip -q -o "${datacenter_zip}" -d "${asset_home}/datacenter"
  touch "${datacenter_marker}"
fi

if [[ "${DOWNLOAD_PHYSICALAI_DATASET:-1}" == "1" ]]; then
  hf_venv="${asset_home}/.venv-huggingface"
  if [[ ! -x "${hf_venv}/bin/hf" ]]; then
    python3 -m venv "${hf_venv}"
    "${hf_venv}/bin/pip" install --quiet --upgrade huggingface_hub
  fi
  "${hf_venv}/bin/hf" download \
    nvidia/PhysicalAI-Robotics-Manipulation-SingleArm \
    --repo-type dataset \
    --local-dir "${asset_home}/datasets/PhysicalAI-Robotics-Manipulation-SingleArm"
fi

sudo docker pull "${isaac_image}"
sudo docker run --rm --entrypoint bash --gpus all --network=host \
  -e ACCEPT_EULA=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "${isaac_mounts[@]}" \
  -v "${HOME}/quantis-robotics:/workspace:rw" \
  "${isaac_image}" \
  -lc './python.sh /workspace/sim/runtime_smoke.py'

printf 'Remote bootstrap complete. If group membership changed, reconnect SSH before starting Isaac Sim.\n'
