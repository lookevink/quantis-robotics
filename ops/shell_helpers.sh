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

require_nonnegative_number() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]] || {
    printf 'error: %s must be a non-negative number\n' "${name}" >&2
    return 1
  }
}

# The fixed insertion corpus has 12 TRAIN recordings with 88 rollouts each.
insertion_epoch_steps() {
  printf '1056\n'
}

insertion_planner_profile_field() {
  local repository="$1"
  local python_bin="$2"
  local profile="$3"
  local field="$4"
  (
    cd "${repository}"
    local -a arguments=(-m jepa_wm.insertion_planner_profile "${field}")
    [[ -z "${profile}" ]] || arguments+=(--profile "${profile}")
    "${python_bin}" "${arguments[@]}"
  )
}

control_resolution_profile_field() {
  local repository="$1"
  local python_bin="$2"
  local field="$3"
  local load="${4:-}"
  (
    cd "${repository}"
    local -a arguments=(-m jepa_wm.control_resolution_profile "${field}")
    [[ -z "${load}" ]] || arguments+=(--load "${load}")
    "${python_bin}" "${arguments[@]}"
  )
}

insertion_rollout_profile_field() {
  local repository="$1"
  local python_bin="$2"
  local profile="$3"
  local field="$4"
  (
    cd "${repository}"
    "${python_bin}" -m jepa_wm.insertion_rollout "${profile}" "${field}"
  )
}

cem_settings_requested() {
  [[ -n "$1$2$3$4" ]]
}

validate_cem_settings() {
  local seed="$1"
  local iterations="$2"
  local samples="$3"
  local elites="$4"
  [[ -n "${seed}" && -n "${iterations}" && -n "${samples}" && -n "${elites}" ]] \
    || {
      printf 'error: all four planner settings must be provided together\n' >&2
      return 1
    }
  require_nonnegative_integer "planner seed" "${seed}" || return 1
  require_positive_integer "planner iterations" "${iterations}" || return 1
  require_positive_integer "planner samples" "${samples}" || return 1
  require_positive_integer "planner elites" "${elites}" || return 1
  (( elites > 1 && elites <= samples )) || {
    printf 'error: planner elites must be between two and the sample count\n' >&2
    return 1
  }
}

load_control_policy_descriptor() {
  local policy="$1"
  local direct_proposal="${2:-direct-proposal}"
  case "${policy}" in
    direct|calibration_collection|insertion_safety_evaluation|insertion_reset_trial|insertion_followup_trial)
      CONTROL_POLICY_PROPOSAL="${direct_proposal}"
      CONTROL_POLICY_REQUIRES_CHECKPOINT=true
      if [[ "${policy}" == "insertion_reset_trial" || "${policy}" == "insertion_followup_trial" ]]; then
        CONTROL_POLICY_RESPONDER=insertion_trial
      else
        CONTROL_POLICY_RESPONDER=direct
      fi
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
  local proposal_name
  load_control_policy_descriptor "${policy}" "${identity}" || return 1
  proposal_name="${CONTROL_POLICY_PROPOSAL}"
  if [[ "${CONTROL_POLICY_REQUIRES_CHECKPOINT}" == "true" ]]; then
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
      isaac_server_call \
        "demo.persist_baseline_response('${session_id}','${policy}')" 120
      ;;
    candidate)
      is_safe_identifier "${source_session_id}" || {
        printf 'error: candidate policy requires a source session\n' >&2
        return 1
      }
      bash "${repository}/ops/jepa_wm.sh" control-candidate-session \
        --session "${session_id}" --source-session "${source_session_id}"
      ;;
    insertion_trial)
      is_safe_identifier "${source_session_id}" || {
        printf 'error: insertion trial policy requires a source session\n' >&2
        return 1
      }
      isaac_server_call \
        "demo.persist_insertion_trial_response('${session_id}','${source_session_id}')" 120
      ;;
  esac
}

capture_and_respond_control_session() {
  local repository="$1"
  local session_id="$2"
  local reference_name="$3"
  local exploration_seed="$4"
  local control_identity="$5"
  local policy="$6"
  local context_index="$7"
  local checkpoint_root="$8"
  local python_bin="$9"
  local source_session_id="${10:-}"
  local insertion_rollout_maximum_steps="${11:-}"
  local insertion_rollout_argument="None"
  local proposal_name
  for identifier in "${session_id}" "${reference_name}" "${control_identity}"; do
    is_safe_identifier "${identifier}" || {
      printf 'error: invalid control capture identifier\n' >&2
      return 1
    }
  done
  require_nonnegative_integer "exploration seed" "${exploration_seed}" || return 1
  require_positive_integer "context index" "${context_index}" || return 1
  if [[ -n "${insertion_rollout_maximum_steps}" ]]; then
    require_positive_integer \
      "insertion rollout maximum steps" \
      "${insertion_rollout_maximum_steps}" || return 1
    insertion_rollout_argument="${insertion_rollout_maximum_steps}"
  fi
  validate_control_policy "${policy}" || return 1
  cd "${repository}"
  proposal_name="$(control_proposal_from_identity \
    "${policy}" "${control_identity}" "${checkpoint_root}" "${python_bin}")" \
    || return 1
  isaac_server_call \
    "await demo.capture_control_observation('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}','${policy}',${context_index},${insertion_rollout_argument})" \
    900 true
  respond_to_control_session \
    "${repository}" "${session_id}" "${policy}" "${source_session_id}"
}

finalize_reset_trial_control_session() {
  local command_status=$?
  local report_status=0
  local -a report_arguments=(
    --rollout "${RESET_TRIAL_SESSION_ID}"
    --reference "${RESET_TRIAL_REFERENCE}"
    --seed "${RESET_TRIAL_SEED}"
    --proposal "${RESET_TRIAL_PROPOSAL}"
    --policy "${RESET_TRIAL_POLICY}"
    --sessions "${RESET_TRIAL_SESSION_ID}"
    --requested-steps 1
  )
  trap - EXIT
  if (( command_status != 0 )); then
    report_arguments+=(
      --orchestration-failure "${RESET_TRIAL_PHASE}:exit_${command_status}"
    )
  fi
  set +e
  bash "${RESET_TRIAL_REPOSITORY}/ops/jepa_wm.sh" control-rollout-report \
    "${report_arguments[@]}"
  report_status=$?
  set -e
  if (( command_status == 0 && report_status != 0 )); then
    command_status=${report_status}
  fi
  exit "${command_status}"
}

run_reset_trial_control_session() {
  RESET_TRIAL_REPOSITORY="$1"
  RESET_TRIAL_SESSION_ID="$2"
  RESET_TRIAL_REFERENCE="$3"
  RESET_TRIAL_SEED="$4"
  RESET_TRIAL_PROPOSAL="$5"
  RESET_TRIAL_POLICY="$6"
  local source_session_id="$7"
  local context_index="$8"
  local capture_timeout="$9"
  local prepare_function="${10}"
  local persist_function="${11}"
  local insertion_rollout_maximum_steps="${12:-}"
  local insertion_rollout_argument="None"
  if [[ -n "${insertion_rollout_maximum_steps}" ]]; then
    require_positive_integer \
      "insertion rollout maximum steps" \
      "${insertion_rollout_maximum_steps}" || return 1
    insertion_rollout_argument="${insertion_rollout_maximum_steps}"
  fi
  RESET_TRIAL_PHASE="reset_trial_source_preflight"
  trap finalize_reset_trial_control_session EXIT
  isaac_server_call \
    "demo.${prepare_function}('${source_session_id}')" 180 true
  RESET_TRIAL_PHASE="reset_trial_capture"
  isaac_server_call \
    "await demo.capture_control_observation('${RESET_TRIAL_SESSION_ID}','${RESET_TRIAL_REFERENCE}',${RESET_TRIAL_SEED},'${RESET_TRIAL_PROPOSAL}','${RESET_TRIAL_POLICY}',${context_index},${insertion_rollout_argument})" \
    "${capture_timeout}"
  RESET_TRIAL_PHASE="reset_trial_binding"
  isaac_server_call \
    "demo.${persist_function}('${RESET_TRIAL_SESSION_ID}','${source_session_id}')" \
    180
  RESET_TRIAL_PHASE="reset_trial_apply"
  isaac_server_call \
    "await demo.apply_control_response('${RESET_TRIAL_SESSION_ID}')" 180
  RESET_TRIAL_PHASE="complete"
  finalize_reset_trial_control_session
}

run_insertion_followup_trial() {
  local repository="$1"
  local safety_session_id="$2"
  local execution_session_id="$3"
  local previous_session_id="$4"
  local reference_name="$5"
  local exploration_seed="$6"
  local proposal_name="$7"
  local session_roster="${8:-${previous_session_id},${execution_session_id}}"
  local requested_steps="${9:-2}"
  local phase="followup_capture_01"
  local command_status=0
  local report_status=0
  local -a report_arguments=(
    --rollout "${execution_session_id}"
    --reference "${reference_name}"
    --seed "${exploration_seed}"
    --proposal "${proposal_name}"
    --policy insertion_followup_trial
    --sessions "${session_roster}"
    --requested-steps "${requested_steps}"
  )
  require_positive_integer "requested rollout steps" "${requested_steps}" \
    || return 1

  isaac_server_call \
    "await demo.capture_followup_observation('${safety_session_id}','${previous_session_id}','${proposal_name}')" \
    180 true || command_status=$?
  if (( command_status == 0 )); then
    phase="followup_inference_01"
    respond_to_control_session \
      "${repository}" "${safety_session_id}" insertion_safety_evaluation \
      || command_status=$?
  fi
  if (( command_status == 0 )); then
    phase="followup_safety_01"
    isaac_server_call \
      "await demo.evaluate_direct_insertion_candidate('${safety_session_id}')" \
      180 || command_status=$?
  fi
  if (( command_status == 0 )); then
    phase="followup_source_preflight_01"
    isaac_server_call \
      "demo.prepare_insertion_trial_source('${safety_session_id}')" \
      180 || command_status=$?
  fi
  if (( command_status == 0 )); then
    phase="followup_binding_01"
    isaac_server_call \
      "demo.persist_insertion_followup_response('${execution_session_id}','${safety_session_id}')" \
      180 || command_status=$?
  fi
  if (( command_status == 0 )); then
    phase="followup_apply_01"
    isaac_server_call \
      "await demo.apply_control_response('${execution_session_id}')" \
      180 || command_status=$?
  fi
  if (( command_status != 0 )); then
    report_arguments+=(
      --orchestration-failure "${phase}:exit_${command_status}"
    )
  fi
  set +e
  bash "${repository}/ops/jepa_wm.sh" control-rollout-report \
    "${report_arguments[@]}"
  report_status=$?
  set -e
  if (( command_status == 0 && report_status != 0 )); then
    command_status=${report_status}
  fi
  return "${command_status}"
}

isaac_demo_code() {
  local expression="$1"
  printf \
    "import sys,json,runpy; sys.path.insert(0,'/workspace') if '/workspace' not in sys.path else None; runtime=runpy.run_path('/workspace/sim/runtime_loader.py'); demo=runtime['reload_demo_runtime'](); print(json.dumps(%s,indent=2))" \
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
