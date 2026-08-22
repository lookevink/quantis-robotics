#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
# shellcheck source=ops/jepa_common.sh
source "${repo_dir}/ops/jepa_common.sh"
venv_dir="${HOME}/.venvs/quantis-jepa"
source_name="${1:-latest}"
camera_name="${2:-wrist}"
recordings_dir="${HOME}/docker/isaac-sim/data/quantis/recordings"
episodes_dir="${repo_dir}/data/episodes"

is_safe_identifier "${source_name}" || {
  printf 'error: invalid JEPA source name: %s\n' "${source_name}" >&2
  exit 1
}
is_safe_identifier "${camera_name}" || {
  printf 'error: invalid JEPA camera name: %s\n' "${camera_name}" >&2
  exit 1
}

ensure_jepa_environment "${repo_dir}" "${venv_dir}"

if [[ "${source_name}" == "latest" ]]; then
  source_dir=""
  if [[ -d "${recordings_dir}" ]]; then
    source_dir="$(find "${recordings_dir}" -mindepth 1 -maxdepth 1 -type d \
      -exec test -f '{}/manifest.json' \; -print | sort | tail -1)"
  fi
  if [[ -z "${source_dir}" ]]; then
    source_dir="$(find "${episodes_dir}" -mindepth 1 -maxdepth 1 -type d \
      2>/dev/null | sort | tail -1)"
  fi
elif [[ -d "${recordings_dir}/${source_name}" ]]; then
  source_dir="${recordings_dir}/${source_name}"
else
  source_dir="${episodes_dir}/${source_name}"
fi

[[ -n "${source_dir}" && -d "${source_dir}" ]] || {
  printf 'error: JEPA source directory does not exist for: %s\n' "${source_name}" >&2
  exit 1
}

if [[ "${source_dir}" == "${recordings_dir}/"* ]]; then
  sudo chown -R "${USER}:${USER}" "${source_dir}"
fi

cd "${repo_dir}"
exec "${venv_dir}/bin/python" -m jepa.embed_episode \
  "${source_dir}" --camera "${camera_name}"
