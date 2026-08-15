#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
isaac_home="${HOME}/docker/isaac-sim"
version="${ISAAC_SIM_VERSION:-5.0.0}"
image="nvcr.io/nvidia/isaac-sim:${version}"
container_name="quantis-isaac-sim"

docker_args=(
  --name "${container_name}"
  --gpus all
  --network=host
  -e ACCEPT_EULA=Y
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e PYTHONDONTWRITEBYTECODE=1
  -e ISAACSIM_SIGNAL_PORT=49100
  -e ISAACSIM_STREAM_PORT=47998
  -v "${isaac_home}/cache/kit:/isaac-sim/kit/cache:rw"
  -v "${isaac_home}/cache/ov:/root/.cache/ov:rw"
  -v "${isaac_home}/cache/pip:/root/.cache/pip:rw"
  -v "${isaac_home}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
  -v "${isaac_home}/cache/computecache:/root/.nv/ComputeCache:rw"
  -v "${isaac_home}/logs:/root/.nvidia-omniverse/logs:rw"
  -v "${isaac_home}/data:/root/.local/share/ov/data:rw"
  -v "${isaac_home}/documents:/root/Documents:rw"
  -v "${repo_dir}:/workspace:rw"
)

case "${1:-help}" in
  start)
    public_ip="$(curl --fail --silent --show-error https://api.ipify.org)"
    sudo docker rm -f "${container_name}" >/dev/null 2>&1 || true
    sudo docker run -d "${docker_args[@]}" \
      -e "ISAACSIM_HOST=${public_ip}" \
      --entrypoint bash \
      "${image}" \
      -lc "./runheadless.sh -v --/exts/omni.kit.livestream.app/primaryStream/publicIp=${public_ip} --/exts/omni.kit.livestream.app/primaryStream/signalPort=49100 --/exts/omni.kit.livestream.app/primaryStream/streamPort=47998"
    printf 'Isaac Sim starting at %s (TCP 49100, UDP 47998)\n' "${public_ip}"
    ;;
  stop)
    sudo docker stop "${container_name}"
    ;;
  logs)
    sudo docker logs -f "${container_name}"
    ;;
  capture-smoke)
    sudo docker run --rm "${docker_args[@]}" \
      --entrypoint bash \
      "${image}" \
      -lc './python.sh /workspace/sim/capture_smoke.py --output /workspace/data/episodes'
    sudo chown -R "${USER}:${USER}" "${repo_dir}/data"
    ;;
  help|*)
    printf 'usage: %s {start|stop|logs|capture-smoke}\n' "$0"
    ;;
esac
