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

isaac_loaded_demo_code() {
  local expression="$1"
  printf \
    "import json; import sim.isaac_demo as demo; print(json.dumps(%s,indent=2))" \
    "${expression}"
}

print_checked_isaac_response() {
  local response="$1"
  printf '%s\n' "${response}"
  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"' <<<"${response}"; then
    return 1
  fi
}

isaac_server_call() {
  local expression="$1"
  local timeout_seconds="$2"
  local reload_runtime="${3:-false}"
  local code
  local response
  if [[ "${reload_runtime}" == "true" ]]; then
    code="$(isaac_demo_code "${expression}")"
  else
    code="$(isaac_loaded_demo_code "${expression}")"
  fi
  response="$(printf '%s\n' "${code}" \
    | timeout "${timeout_seconds}" nc -N 127.0.0.1 8226)"
  print_checked_isaac_response "${response}"
}

capture_shadow_control_evidence() {
  local repository="$1"
  local session_id="$2"
  if ! bash "${repository}/ops/jepa_wm.sh" \
    control-shadow-session --session "${session_id}"; then
    printf 'warning: shadow planning failed for control session %s\n' \
      "${session_id}" >&2
    return 0
  fi
  if ! isaac_server_call \
    "await demo.evaluate_shadow_candidate('${session_id}')" 180; then
    printf 'warning: shadow safety evaluation failed for control session %s\n' \
      "${session_id}" >&2
  fi
}
