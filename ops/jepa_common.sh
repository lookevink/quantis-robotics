#!/usr/bin/env bash

ensure_jepa_environment() {
  local repo_dir="$1"
  local venv_dir="$2"
  if [[ ! -x "${venv_dir}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${venv_dir}"
  fi
  "${venv_dir}/bin/python" -m pip install --quiet --disable-pip-version-check \
    -r "${repo_dir}/jepa/requirements.txt"
}
