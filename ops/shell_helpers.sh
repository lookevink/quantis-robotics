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

load_control_policy_descriptor() {
  local policy="$1"
  local direct_proposal="${2:-direct-proposal}"
  case "${policy}" in
    direct)
      CONTROL_POLICY_PROPOSAL="${direct_proposal}"
      CONTROL_POLICY_REQUIRES_CHECKPOINT=true
      CONTROL_POLICY_RESPONDER=direct
      ;;
    zero|scripted)
      CONTROL_POLICY_PROPOSAL="baseline_${policy}"
      CONTROL_POLICY_REQUIRES_CHECKPOINT=false
      CONTROL_POLICY_RESPONDER=baseline
      ;;
    reset_trial_candidate)
      CONTROL_POLICY_PROPOSAL=experimental_shadow_candidate
      CONTROL_POLICY_REQUIRES_CHECKPOINT=false
      CONTROL_POLICY_RESPONDER=candidate
      ;;
    *)
      printf 'error: unsupported control policy: %s\n' "${policy}" >&2
      return 1
      ;;
  esac
}

validate_control_policy() {
  load_control_policy_descriptor "$1" "${2:-direct-proposal}"
}

control_proposal_for_policy() {
  load_control_policy_descriptor "$1" "${2:-direct-proposal}" || return 1
  printf '%s\n' "${CONTROL_POLICY_PROPOSAL}"
}

control_proposal_from_identity() {
  local policy="$1"
  local identity="$2"
  local checkpoint_root="$3"
  local python_bin="$4"
  local proposal_name="${identity}"
  if [[ "${policy}" == "direct" ]]; then
    proposal_name="$("${python_bin}" -m jepa_wm.worker_artifacts proposal-name \
      --manifest "${checkpoint_root}/${identity}.worker.json")" || return 1
  fi
  is_safe_identifier "${proposal_name}" || {
    printf 'error: control identity resolves to an invalid proposal\n' >&2
    return 1
  }
  printf '%s\n' "${proposal_name}"
}

respond_to_control_session() {
  local repository="$1"
  local session_id="$2"
  local policy="$3"
  local source_session_id="${4:-}"
  load_control_policy_descriptor "${policy}" || return 1
  case "${CONTROL_POLICY_RESPONDER}" in
    direct)
      bash "${repository}/ops/jepa_wm.sh" \
        control-infer-session --session "${session_id}"
      ;;
    baseline)
      bash "${repository}/ops/jepa_wm.sh" control-baseline-session \
        --session "${session_id}" --policy "${policy}"
      ;;
    candidate)
      is_safe_identifier "${source_session_id}" || {
        printf 'error: candidate policy requires a source session\n' >&2
        return 1
      }
      bash "${repository}/ops/jepa_wm.sh" control-candidate-session \
        --session "${session_id}" --source-session "${source_session_id}"
      ;;
  esac
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
