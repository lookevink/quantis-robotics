#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=isaac_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/isaac_common.sh"
# shellcheck source=shell_helpers.sh
source "$(dirname "${BASH_SOURCE[0]}")/shell_helpers.sh"

recording_id="${1:-}"
if ! is_safe_identifier "${recording_id}"; then
  printf 'error: invalid recording ID: %s\n' "${recording_id}" >&2
  exit 1
fi

recording_dir="${isaac_home}/data/quantis/recordings/${recording_id}"
manifest_path="${recording_dir}/manifest.json"
if [[ ! -f "${manifest_path}" ]]; then
  printf 'error: recording manifest does not exist: %s\n' "${manifest_path}" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf 'error: ffmpeg is unavailable; run ./ops/aws.sh remote-bootstrap\n' >&2
  exit 1
fi

fps="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["fps"])' "${manifest_path}")"
if [[ ! "${fps}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'error: invalid recording FPS in %s: %s\n' "${manifest_path}" "${fps}" >&2
  exit 1
fi

mapfile -t camera_rows < <(
  python3 -c \
    'import json, sys; manifest=json.load(open(sys.argv[1])); [print(f"{camera}\t{manifest['\''videos'\''][camera]}") for camera in manifest["cameras"]]' \
    "${manifest_path}"
)
if (( ${#camera_rows[@]} == 0 )); then
  printf 'error: recording manifest has no cameras: %s\n' "${manifest_path}" >&2
  exit 1
fi

video_names=()
for camera_row in "${camera_rows[@]}"; do
  IFS=$'\t' read -r camera video_name <<<"${camera_row}"
  if [[ ! "${camera}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || [[ ! "${video_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$ ]]; then
    printf 'error: unsafe camera entry in %s: %s\n' "${manifest_path}" "${camera_row}" >&2
    exit 1
  fi
  frame_pattern="${recording_dir}/${camera}/frame_%06d.png"
  video_path="${recording_dir}/${video_name}"
  if [[ ! -f "${recording_dir}/${camera}/frame_000000.png" ]]; then
    printf 'error: %s has no camera frames\n' "${camera}" >&2
    exit 1
  fi
  sudo ffmpeg -hide_banner -loglevel error -y \
    -framerate "${fps}" \
    -i "${frame_pattern}" \
    -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
    "${video_path}"
  sudo chown "${USER}:${USER}" "${video_path}"
  [[ -s "${video_path}" ]] || {
    printf 'error: ffmpeg produced an empty video: %s\n' "${video_path}" >&2
    exit 1
  }
  video_names+=("${video_name}")
done
sudo chown -R "${USER}:${USER}" "${recording_dir}"

printf 'Recording encoded successfully.\n'
printf 'Host output directory: %s\n' "${recording_dir}"
printf 'Isaac container directory: %s\n' "/isaac-sim/.local/share/ov/data/quantis/recordings/${recording_id}"
printf 'Videos: %s\n' "${video_names[*]}"
