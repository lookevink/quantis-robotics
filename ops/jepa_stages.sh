#!/usr/bin/env bash
set -euo pipefail

repo_dir="${HOME}/quantis-robotics"
# shellcheck source=ops/shell_helpers.sh
source "${repo_dir}/ops/shell_helpers.sh"
# shellcheck source=ops/jepa_common.sh
source "${repo_dir}/ops/jepa_common.sh"
venv_dir="${HOME}/.venvs/quantis-jepa"
recordings_dir="${HOME}/docker/isaac-sim/data/quantis/recordings"
action="${1:-}"

recording_path() {
  local name="$1"
  is_safe_identifier "${name}" || {
    printf 'error: invalid recording name: %s\n' "${name}" >&2
    return 1
  }
  local path
  if [[ "${name}" == "latest" ]]; then
    path=""
    if [[ -d "${recordings_dir}" ]]; then
      path="$(find "${recordings_dir}" -mindepth 1 -maxdepth 1 -type d \
        -exec test -f '{}/manifest.json' \; -print | sort | tail -1)"
    fi
  else
    path="${recordings_dir}/${name}"
  fi
  [[ -d "${path}" ]] || {
    printf 'error: recording does not exist: %s\n' "${path}" >&2
    return 1
  }
  printf '%s\n' "${path}"
}

ensure_jepa_environment "${repo_dir}" "${venv_dir}"
cd "${repo_dir}"

case "${action}" in
  embed)
    recording="$(recording_path "${2:-}")"
    camera="${3:-wrist}"
    is_safe_identifier "${camera}" || {
      printf 'error: invalid camera name: %s\n' "${camera}" >&2
      exit 1
    }
    sudo chown -R "${USER}:${USER}" "${recording}"
    exec "${venv_dir}/bin/python" -m jepa.embed_stages \
      "${recording}" --camera "${camera}"
    ;;
  report)
    reference="$(recording_path "${2:-}")"
    query="$(recording_path "${3:-}")"
    camera="${4:-wrist}"
    is_safe_identifier "${camera}" || {
      printf 'error: invalid camera name: %s\n' "${camera}" >&2
      exit 1
    }
    sudo chown -R "${USER}:${USER}" "${reference}" "${query}"
    exec "${venv_dir}/bin/python" -m jepa.report_stages \
      --reference "${reference}" --query "${query}" --camera "${camera}"
    ;;
  *)
    printf 'error: expected embed or report\n' >&2
    exit 1
    ;;
esac
