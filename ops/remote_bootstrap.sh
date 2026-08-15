#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=isaac_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/isaac_common.sh"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'error: nvidia-smi is unavailable; use a Lambda GPU image\n' >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y docker.io
fi

# Lambda's base A10 image includes compute libraries but may omit the matching
# Vulkan and NVENC/NVDEC packages required by Isaac Sim and WebRTC.
required_driver_packages=(
  libnvidia-gl-570-server
  libnvidia-encode-570-server
  libnvidia-decode-570-server
)
missing_driver_packages=()
for package in "${required_driver_packages[@]}"; do
  if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
    missing_driver_packages+=("${package}")
  fi
done
if (( ${#missing_driver_packages[@]} )); then
  sudo apt-get update
  sudo apt-get install -y "${missing_driver_packages[@]}"
fi

kernel_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
userspace_driver="$(dpkg-query -W -f='${Version}' libnvidia-gl-570-server | cut -d- -f1)"
if [[ "${kernel_driver}" != "${userspace_driver}" ]]; then
  printf 'NVIDIA graphics libraries advanced from %s to %s; reboot the instance, then rerun remote-bootstrap.\n' \
    "${kernel_driver}" "${userspace_driver}" >&2
  exit 2
fi

if ! sudo docker info >/dev/null 2>&1; then
  sudo systemctl enable --now docker
fi
sudo usermod -aG docker "${USER}"
sudo systemctl restart docker

if ! sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
  printf 'error: Docker cannot access the GPU. Install/configure NVIDIA Container Toolkit, then rerun.\n' >&2
  exit 1
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
  "${HOME}/quantis-robotics/data/episodes"

sudo chown -R "${USER}:${USER}" "${isaac_home}" "${HOME}/quantis-robotics/data"

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
