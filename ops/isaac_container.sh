#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=isaac_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/isaac_common.sh"
container_name="quantis-isaac-sim"
signal_port="${ISAAC_SIGNAL_PORT:-49100}"
stream_port="${ISAAC_STREAM_PORT:-47998}"

verify_checkpoint_directory_readable() {
  [[ -d "${jepa_wm_checkpoint_dir}" ]] || {
    printf 'error: checkpoint directory does not exist: %s\n' \
      "${jepa_wm_checkpoint_dir}" >&2
    return 1
  }
  if ! sudo docker run --rm \
    "${isaac_checkpoint_access_args[@]}" \
    --entrypoint bash \
    "${isaac_image}" \
    -lc 'test -r "$1" && test -x "$1"' _ "${jepa_wm_checkpoint_dir}"; then
    printf 'error: checkpoint directory is unreadable to the Isaac runtime user\n' >&2
    return 1
  fi
}

docker_args=(
  "${isaac_checkpoint_access_args[@]}"
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
    verify_checkpoint_directory_readable
    public_ip="$(curl --fail --silent --show-error https://api.ipify.org)"
    sudo docker rm -f "${container_name}" >/dev/null 2>&1 || true
    sudo docker run -d --name "${container_name}" "${docker_args[@]}" \
      -e "ISAACSIM_HOST=${public_ip}" \
      --entrypoint bash \
      "${isaac_image}" \
      -lc "./runheadless.sh --/exts/omni.kit.livestream.app/primaryStream/publicIp=${public_ip} --/exts/omni.kit.livestream.app/primaryStream/signalPort=${signal_port} --/exts/omni.kit.livestream.app/primaryStream/streamPort=${stream_port}"
    printf 'Isaac Sim starting at %s (TCP %s, UDP %s)\n' "${public_ip}" "${signal_port}" "${stream_port}"
    ;;
  stop)
    sudo docker stop "${container_name}"
    ;;
  logs)
    sudo docker logs -f "${container_name}"
    ;;
  status)
    sudo docker inspect --format \
      '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' \
      "${container_name}"
    ;;
  checkpoint-readable)
    proposal_name="${2:-}"
    [[ "${proposal_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
      printf 'error: invalid checkpoint proposal name\n' >&2
      exit 1
    }
    checkpoint_path="${jepa_wm_checkpoint_dir}/${proposal_name}.pth"
    for artifact_path in "${checkpoint_path}" "${checkpoint_path}.json"; do
      sudo docker exec "${container_name}" test -r "${artifact_path}" || {
        printf 'error: checkpoint artifact is unreadable to Isaac: %s\n' \
          "${artifact_path}" >&2
        exit 1
      }
    done
    ;;
  capture-smoke)
    sudo docker run --rm "${docker_args[@]}" \
      --entrypoint bash \
      "${isaac_image}" \
      -lc './python.sh /workspace/sim/capture_smoke.py --output /workspace/data/episodes'
    sudo chown -R "${USER}:${USER}" "${repo_dir}/data"
    ;;
  help|*)
    printf 'usage: %s {start|stop|status|checkpoint-readable PROPOSAL|logs|capture-smoke}\n' "$0"
    ;;
esac
