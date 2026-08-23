#!/usr/bin/env bash

is_safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
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
