#!/usr/bin/env bash

isaac_control_capture_timeout_seconds=900
isaac_insertion_trial_apply_timeout_seconds=600

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

resolve_insertion_context() {
  local supplied="$1"
  local repository="$2"
  local python_bin="$3"
  if [[ -n "${supplied}" ]]; then
    printf '%s\n' "${supplied}"
    return
  fi
  (
    cd "${repository}"
    "${python_bin}" -m jepa_wm.insertion_layout initial-command-context
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

contact_grasp_maximum_actions() {
  local repository="$1"
  local python_bin="$2"
  (
    cd "${repository}"
    "${python_bin}" -c \
      'from jepa_wm.grasp_task import MAXIMUM_CONTACT_GRASP_ACTIONS; print(MAXIMUM_CONTACT_GRASP_ACTIONS)'
  )
}

contact_grasp_initial_context() {
  local repository="$1"
  local python_bin="$2"
  (
    cd "${repository}"
    "${python_bin}" -m jepa_wm.task_windows contact-grasp start-index
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

validate_demo_run_spec() {
  local source_revision="$1"
  local python_bin="$2"
  local spec_path="$3"
  local spec_fingerprint="$4"
  local recording_root="$5"
  local stage_asset="$6"
  local grasp_identity="$7"
  local grasp_manifest="$8"
  local insertion_identity="$9"
  local insertion_manifest="${10}"
  local reference_recording="${11}"
  local exploration_seed="${12}"
  local run_id="${13}"
  local binding_output="${14}"
  local grasp_actions="${15}"
  local insertion_actions="${16}"
  local container_image_digest
  [[ "${source_revision}" =~ ^[0-9a-f]{40}$ ]] || {
    printf 'error: invalid deployed source revision\n' >&2
    return 1
  }
  container_image_digest="$(sudo docker inspect --format '{{.Image}}' quantis-isaac-sim)" \
    || return 1
  "${python_bin}" -m sim.demo_run_cli verify \
    --spec "${spec_path}" \
    --fingerprint "${spec_fingerprint}" \
    --recording-root "${recording_root}" \
    --source-revision "${source_revision}" \
    --container-image-digest "${container_image_digest}" \
    --run-id "${run_id}" \
    --binding-output "${binding_output}" \
    --grasp-actions "${grasp_actions}" \
    --insertion-actions "${insertion_actions}" \
    --reference-recording "${reference_recording}" \
    --exploration-seed "${exploration_seed}" \
    --artifact "stage_asset=${stage_asset}" \
    --worker "grasp=${grasp_identity}=${grasp_manifest}" \
    --worker "insertion=${insertion_identity}=${insertion_manifest}"
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
  local context_purpose="${12:-standard}"
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
  [[ "${context_purpose}" == "standard" || "${context_purpose}" == "contact_grasp" ]] || {
    printf 'error: invalid control context purpose\n' >&2
    return 1
  }
  validate_control_policy "${policy}" || return 1
  cd "${repository}"
  proposal_name="$(control_proposal_from_identity \
    "${policy}" "${control_identity}" "${checkpoint_root}" "${python_bin}")" \
    || return 1
  start_and_wait_control_capture \
    "${session_id}" "${reference_name}" "${exploration_seed}" \
    "${proposal_name}" "${policy}" "${context_index}" \
    "${insertion_rollout_argument}" "${context_purpose}" \
    "${isaac_control_capture_timeout_seconds}" true
  respond_to_control_session \
    "${repository}" "${session_id}" "${policy}" "${source_session_id}"
}

start_and_wait_control_capture() {
  local session_id="$1"
  local reference_name="$2"
  local exploration_seed="$3"
  local proposal_name="$4"
  local policy="$5"
  local context_index="$6"
  local insertion_rollout_argument="$7"
  local context_purpose="$8"
  local timeout_seconds="$9"
  local reload_runtime="${10:-false}"
  local job_id="control-${session_id}"
  isaac_server_call \
    "demo.start_control_capture('${session_id}','${reference_name}',${exploration_seed},'${proposal_name}','${policy}',${context_index},${insertion_rollout_argument},'${context_purpose}')" \
    60 "${reload_runtime}"
  wait_control_capture_job "${job_id}" "${timeout_seconds}"
}

wait_control_capture_job() {
  local job_id="$1"
  local timeout_seconds="$2"
  local job_file="${HOME}/docker/isaac-sim/data/quantis/recording_jobs/${job_id}.json"
  local deadline=$((SECONDS + timeout_seconds))
  local status=""
  local next_notice
  is_safe_identifier "${job_id}" || {
    printf 'error: invalid control capture job ID: %s\n' "${job_id}" >&2
    return 1
  }
  require_positive_integer "control capture timeout" "${timeout_seconds}" \
    || return 1
  while (( SECONDS < deadline )); do
    status="$(control_capture_job_status "${job_file}")" || return 1
    if [[ "${status}" == "complete" || "${status}" == "error" ]]; then
      cat "${job_file}"
      [[ "${status}" == "complete" ]]
      return
    fi
    sleep 1
  done
  printf 'error: control capture job timed out: %s\n' "${job_id}" >&2
  while ! isaac_server_call "demo.cancel_recording_job('${job_id}')" 30; do
    status="$(control_capture_job_status "${job_file}")" || return 1
    if [[ "${status}" == "complete" || "${status}" == "error" ]]; then
      cat "${job_file}" >&2
      return 124
    fi
    printf 'waiting to deliver cancellation for control capture job: %s\n' \
      "${job_id}" >&2
    sleep 30
  done
  next_notice=$((SECONDS + 30))
  while true; do
    status="$(control_capture_job_status "${job_file}")" || return 1
    if [[ "${status}" == "complete" || "${status}" == "error" ]]; then
      cat "${job_file}" >&2
      return 124
    fi
    if (( SECONDS >= next_notice )); then
      printf 'waiting for cancelled control capture job to terminalize: %s\n' \
        "${job_id}" >&2
      next_notice=$((SECONDS + 30))
    fi
    sleep 1
  done
}

control_capture_job_status() {
  local job_file="$1"
  if [[ ! -f "${job_file}" ]]; then
    return 0
  fi
  python3 -c \
    'import json,sys; status=json.load(open(sys.argv[1])).get("status"); print(status if isinstance(status,str) else "")' \
    "${job_file}"
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
  local apply_timeout_seconds="${13:-180}"
  local insertion_rollout_argument="None"
  require_positive_integer "control apply timeout" "${apply_timeout_seconds}" \
    || return 1
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
  start_and_wait_control_capture \
    "${RESET_TRIAL_SESSION_ID}" "${RESET_TRIAL_REFERENCE}" \
    "${RESET_TRIAL_SEED}" "${RESET_TRIAL_PROPOSAL}" \
    "${RESET_TRIAL_POLICY}" "${context_index}" \
    "${insertion_rollout_argument}" standard "${capture_timeout}" false
  RESET_TRIAL_PHASE="reset_trial_binding"
  isaac_server_call \
    "demo.${persist_function}('${RESET_TRIAL_SESSION_ID}','${source_session_id}')" \
    180
  RESET_TRIAL_PHASE="reset_trial_apply"
  isaac_server_call \
    "await demo.apply_control_response('${RESET_TRIAL_SESSION_ID}')" \
    "${apply_timeout_seconds}"
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
  local predecessor_session_id="${10:-}"
  local proposal_handoff="${11:-false}"
  local runtime_owner_session="${12:-}"
  local next_maximum_steps="${13:-}"
  local reload_capture="true"
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
  if [[ -n "${predecessor_session_id}" ]]; then
    is_safe_identifier "${predecessor_session_id}" || return 1
    report_arguments+=(--predecessor-session "${predecessor_session_id}")
  fi
  require_positive_integer "requested rollout steps" "${requested_steps}" \
    || return 1
  if [[ -n "${next_maximum_steps}" ]]; then
    require_positive_integer "next insertion rollout maximum" \
      "${next_maximum_steps}" || return 1
  fi
  if [[ "${proposal_handoff}" != "true" && "${proposal_handoff}" != "false" ]]; then
    printf 'error: invalid insertion proposal handoff mode\n' >&2
    return 1
  fi
  if [[ -n "${runtime_owner_session}" ]]; then
    is_safe_identifier "${runtime_owner_session}" || return 1
    local restore_maximum_argument=""
    if [[ -n "${next_maximum_steps}" ]]; then
      restore_maximum_argument=",${next_maximum_steps}"
    fi
    isaac_server_call \
      "demo.restore_insertion_retry('${previous_session_id}','${runtime_owner_session}'${restore_maximum_argument})" \
      180 true || return 1
    reload_capture="false"
  fi
  if [[ "${proposal_handoff}" == "true" ]]; then
    local encoded_handoff
    encoded_handoff="$(
      bash "${repository}/ops/jepa_wm.sh" insertion-transition-handoff \
        "${previous_session_id}" "${proposal_name}" "${safety_session_id}"
    )" || return 1
    isaac_server_call \
      "demo.persist_insertion_proposal_handoff('${previous_session_id}','${safety_session_id}','${encoded_handoff}')" \
      180 "${reload_capture}" || return 1
    reload_capture="false"
  fi

  local capture_maximum_argument=""
  if [[ -n "${next_maximum_steps}" ]]; then
    capture_maximum_argument=",${next_maximum_steps}"
  fi
  isaac_server_call \
    "await demo.capture_followup_observation('${safety_session_id}','${previous_session_id}','${proposal_name}'${capture_maximum_argument})" \
    180 "${reload_capture}" || command_status=$?
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
      "${isaac_insertion_trial_apply_timeout_seconds}" || command_status=$?
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

require_control_rollout_applied() {
  local python_bin="$1"
  local report="$2"
  "${python_bin}" -c \
    'import json,sys; payload=json.load(open(sys.argv[1])); raise SystemExit(0 if payload.get("all_steps_applied") is True else "control rollout did not apply every requested step")' \
    "${report}"
}

require_control_rollout_reach_and_grasp() {
  local python_bin="$1"
  local report="$2"
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); g=p.get("reach_and_grasp") or {}; ok=(g.get("passed") is True and p.get("orchestration_failure") is None and p.get("applied_steps")==p.get("complete_steps")); raise SystemExit(0 if ok else "control rollout did not establish a retained grasp")' \
    "${report}"
}

control_rollout_terminal_session() {
  local python_bin="$1"
  local report="$2"
  "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); steps=p.get("steps") or []; assert steps, "control rollout has no terminal step"; print(steps[-1]["session"])' \
    "${report}"
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

control_rollout_shadow_session_roster() {
  local context_purpose="$1"
  local sessions="$2"
  local first_session
  local final_session
  if [[ "${context_purpose}" != "contact_grasp" || "${sessions}" != *,* ]]; then
    printf '%s\n' "${sessions}"
    return
  fi
  first_session="${sessions%%,*}"
  final_session="${sessions##*,}"
  printf '%s,%s\n' "${first_session}" "${final_session}"
}
