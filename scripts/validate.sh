#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

for script in ops/*.sh scripts/*.sh; do
  bash -n "${script}"
done

python3 -m compileall -q sim jepa tests
python3 -m unittest discover -s tests -v

printf 'Validation passed.\n'
