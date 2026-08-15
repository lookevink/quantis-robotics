#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=isaac_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/isaac_common.sh"
container_name="quantis-isaac-sim"
signal_port="${ISAAC_SIGNAL_PORT:-49100}"
stream_port="${ISAAC_STREAM_PORT:-47998}"

docker_args=(
  --gpus all
  --network=host
  -e ACCEPT_EULA=Y
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e PYTHONDONTWRITEBYTECODE=1
  -e "ISAACSIM_SIGNAL_PORT=${signal_port}"
  -e "ISAACSIM_STREAM_PORT=${stream_port}"
  "${isaac_mounts[@]}"
  -v "${repo_dir}:/workspace:rw"
)

case "${1:-help}" in
  start)
    public_ip="$(curl --fail --silent --show-error https://api.ipify.org)"
    sudo docker rm -f "${container_name}" >/dev/null 2>&1 || true
    sudo docker run -d --name "${container_name}" "${docker_args[@]}" \
      -e "ISAACSIM_HOST=${public_ip}" \
      --entrypoint bash \
      "${isaac_image}" \
      -lc "./runheadless.sh -v --/exts/omni.kit.livestream.app/primaryStream/publicIp=${public_ip} --/exts/omni.kit.livestream.app/primaryStream/signalPort=${signal_port} --/exts/omni.kit.livestream.app/primaryStream/streamPort=${stream_port}"
    printf 'Isaac Sim starting at %s (TCP %s, UDP %s)\n' "${public_ip}" "${signal_port}" "${stream_port}"
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
      "${isaac_image}" \
      -lc './python.sh /workspace/sim/capture_smoke.py --output /workspace/data/episodes'
    sudo chown -R "${USER}:${USER}" "${repo_dir}/data"
    ;;
  help|*)
    printf 'usage: %s {start|stop|logs|capture-smoke}\n' "$0"
    ;;
esac
