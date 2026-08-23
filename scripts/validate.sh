#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${QUANTIS_TEST_PYTHON:-${repo_root}/.runtime/test-venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi

for script in ops/*.sh scripts/*.sh; do
  bash -n "${script}"
done

"${python_bin}" -m compileall -q sim jepa jepa_wm tests
"${python_bin}" -m unittest discover -s tests -v

printf 'Validation passed.\n'
