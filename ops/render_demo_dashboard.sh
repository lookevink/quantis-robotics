#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/isaac_common.sh
source "${repo_dir}/ops/isaac_common.sh"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
# shellcheck source=ops/jepa_common.sh
source "${repo_dir}/ops/jepa_common.sh"

recording_id="${1:-}"
primary_camera="${2:-wrist}"
jepa_camera="${3:-wrist}"
venv_dir="${HOME}/.venvs/quantis-jepa"

is_safe_identifier "${recording_id}" || {
  printf 'error: invalid recording ID: %s\n' "${recording_id}" >&2
  exit 1
}
is_safe_identifier "${primary_camera}" || {
  printf 'error: invalid primary camera: %s\n' "${primary_camera}" >&2
  exit 1
}
is_safe_identifier "${jepa_camera}" || {
  printf 'error: invalid JEPA camera: %s\n' "${jepa_camera}" >&2
  exit 1
}

recording_dir="${isaac_home}/data/quantis/recordings/${recording_id}"
manifest_path="${recording_dir}/manifest.json"
[[ -f "${manifest_path}" ]] || {
  printf 'error: recording manifest does not exist: %s\n' "${manifest_path}" >&2
  exit 1
}
[[ -f "${recording_dir}/${primary_camera}/frame_000000.png" ]] || {
  printf 'error: primary camera frames do not exist: %s\n' "${primary_camera}" >&2
  exit 1
}

ensure_jepa_environment "${repo_dir}" "${venv_dir}"
cd "${repo_dir}"
"${venv_dir}/bin/python" -m jepa.dashboard \
  "${recording_dir}" --jepa-camera "${jepa_camera}"

layout_path="${recording_dir}/dashboard/layout.json"
read -r fps frames primary_width primary_height primary_y output_width output_height < <(
  python3 -c \
    'import json,sys; m=json.load(open(sys.argv[1])); l=json.load(open(sys.argv[2])); print(m["fps"], m["frames"], *l["primary_size"], l["primary_y"], *l["output_size"])' \
    "${manifest_path}" "${layout_path}"
)
[[ "${fps}" =~ ^[1-9][0-9]*$ && "${frames}" =~ ^[1-9][0-9]*$ ]] || {
  printf 'error: invalid FPS or frame count in %s\n' "${manifest_path}" >&2
  exit 1
}
for dimension in \
  "${primary_width}" "${primary_height}" "${primary_y}" \
  "${output_width}" "${output_height}"; do
  [[ "${dimension}" =~ ^[0-9]+$ ]] || {
    printf 'error: invalid dashboard layout in %s\n' "${layout_path}" >&2
    exit 1
  }
done

output_path="${recording_dir}/dashboard.mp4"
filter="[0:v]scale=${primary_width}:${primary_height},pad=${primary_width}:${output_height}:0:${primary_y}:color=0x080d15[left];[left][1:v]hstack=inputs=2[out]"
ffmpeg -hide_banner -loglevel error -y \
  -framerate "${fps}" -i "${recording_dir}/${primary_camera}/frame_%06d.png" \
  -framerate "${fps}" -i "${recording_dir}/dashboard/panel/frame_%06d.png" \
  -filter_complex "${filter}" \
  -map '[out]' -frames:v "${frames}" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
  "${output_path}"

[[ -s "${output_path}" ]] || {
  printf 'error: FFmpeg produced an empty dashboard video\n' >&2
  exit 1
}

printf 'Dashboard video rendered successfully.\n'
printf 'Host output: %s\n' "${output_path}"
printf 'Resolution: %sx%s\n' "${output_width}" "${output_height}"
printf 'Primary camera: %s\n' "${primary_camera}"
