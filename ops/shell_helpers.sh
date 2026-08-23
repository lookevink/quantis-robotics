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
