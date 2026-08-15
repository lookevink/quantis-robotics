#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
venv_dir="${HOME}/.venvs/quantis-jepa"
episode_name="${1:-latest}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python3 -m venv --system-site-packages "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --disable-pip-version-check \
  -r "${repo_dir}/jepa/requirements.txt"

if [[ "${episode_name}" == "latest" ]]; then
  episode_dir="$(find "${repo_dir}/data/episodes" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
else
  [[ "${episode_name}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'error: invalid episode name: %s\n' "${episode_name}" >&2
    exit 1
  }
  episode_dir="${repo_dir}/data/episodes/${episode_name}"
fi

[[ -d "${episode_dir}" ]] || {
  printf 'error: episode directory does not exist: %s\n' "${episode_dir}" >&2
  exit 1
}

exec "${venv_dir}/bin/python" "${repo_dir}/jepa/embed_episode.py" "${episode_dir}"
