#!/usr/bin/env bash

is_safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
}

is_safe_identifier_list() {
  local remainder="$1"
  local identifier
  [[ -n "${remainder}" ]] || return 1
  while [[ "${remainder}" == *,* ]]; do
    identifier="${remainder%%,*}"
    is_safe_identifier "${identifier}" || return 1
    remainder="${remainder#*,}"
  done
  is_safe_identifier "${remainder}"
}

require_nonnegative_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || {
    printf 'error: %s must be non-negative\n' "${name}" >&2
    return 1
  }
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'error: %s must be positive\n' "${name}" >&2
    return 1
  }
}

isaac_demo_code() {
  local expression="$1"
  printf \
    "import sys,json,importlib; sys.path.insert(0,'/workspace') if '/workspace' not in sys.path else None; importlib.invalidate_caches(); import sim.runtime_loader as loader; importlib.reload(loader); demo=loader.reload_demo_runtime(); print(json.dumps(%s,indent=2))" \
    "${expression}"
}

print_checked_isaac_response() {
  local response="$1"
  printf '%s\n' "${response}"
  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"' <<<"${response}"; then
    return 1
  fi
}
