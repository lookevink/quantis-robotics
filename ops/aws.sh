#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${ENV_FILE:-${repo_root}/.env}"
# shellcheck source=ops/shell_helpers.sh
source "${repo_root}/ops/shell_helpers.sh"

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

expected_account="686410906008"
profile="quantis"
region="${AWS_REGION:-us-east-1}"
instance_id="${AWS_INSTANCE_ID:-}"
private_key="${AWS_SSH_PRIVATE_KEY:-${HOME}/.ssh/github_signing_ed25519}"
ssh_user="${AWS_SSH_USER:-ubuntu}"
security_group_id="${AWS_SECURITY_GROUP_ID:-}"
signal_port="${ISAAC_SIGNAL_PORT:-49100}"
stream_port="${ISAAC_STREAM_PORT:-47998}"
ssh_transport_options=(
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=4
)
ssh_options=("${ssh_transport_options[@]}")

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

aws_cli() {
  command aws --profile "${profile}" --region "${region}" "$@"
}

verify_identity() {
  command -v aws >/dev/null 2>&1 || die "AWS CLI is not installed"
  local account
  account="$(aws_cli sts get-caller-identity --query Account --output text)"
  [[ "${account}" == "${expected_account}" ]] \
    || die "AWS profile ${profile} resolved to account ${account}, expected ${expected_account}"
}

require_instance() {
  [[ "${instance_id}" =~ ^i-[0-9a-f]+$ ]] \
    || die "set AWS_INSTANCE_ID to the EC2 GPU instance ID in ${env_file}"
}

instance_value() {
  local query="$1"
  local value
  value="$(aws_cli ec2 describe-instances \
    --instance-ids "${instance_id}" \
    --query "Reservations[0].Instances[0].${query}" \
    --output text)"
  [[ "${value}" != "None" && -n "${value}" ]] \
    || die "instance ${instance_id} has no ${query}"
  printf '%s\n' "${value}"
}

instance_state() {
  instance_value 'State.Name'
}

instance_ip() {
  instance_value 'PublicIpAddress'
}

ensure_running() {
  local state
  state="$(instance_state)"

  case "${state}" in
    running)
      printf 'Instance %s is already running.\n' "${instance_id}" >&2
      ;;
    pending)
      printf 'Instance %s is starting.\n' "${instance_id}" >&2
      ;;
    stopping)
      printf 'Waiting for instance %s to stop before restarting it.\n' "${instance_id}" >&2
      aws_cli ec2 wait instance-stopped --instance-ids "${instance_id}"
      aws_cli ec2 start-instances --instance-ids "${instance_id}" >/dev/null
      ;;
    stopped)
      printf 'Starting instance %s.\n' "${instance_id}" >&2
      aws_cli ec2 start-instances --instance-ids "${instance_id}" >/dev/null
      ;;
    shutting-down|terminated)
      die "instance ${instance_id} is ${state} and cannot be restarted"
      ;;
    *)
      die "instance ${instance_id} is in unsupported state ${state}"
      ;;
  esac

  aws_cli ec2 wait instance-running --instance-ids "${instance_id}"
  aws_cli ec2 wait instance-status-ok --instance-ids "${instance_id}"
  printf 'Instance %s is ready at %s.\n' "${instance_id}" "$(instance_ip)" >&2
}

require_private_key() {
  [[ -f "${private_key}" ]] || die "SSH private key does not exist: ${private_key}"
  ssh_options=(
    -i "${private_key}"
    "${ssh_transport_options[@]}"
  )
}

resolve_security_group() {
  if [[ -n "${security_group_id}" ]]; then
    printf '%s\n' "${security_group_id}"
    return
  fi
  instance_value 'SecurityGroups[0].GroupId'
}

configure_firewall() {
  local source_cidr="${WEBRTC_SOURCE_CIDR:-}"
  local group_id
  local rule_ids

  if [[ -z "${source_cidr}" ]]; then
    source_cidr="$(curl --fail --silent --show-error https://api.ipify.org)/32"
  fi
  [[ "${source_cidr}" =~ ^[^/]+/[0-9]+$ ]] || die "invalid source CIDR: ${source_cidr}"
  group_id="$(resolve_security_group)"

  rule_ids="$(aws_cli ec2 describe-security-group-rules \
    --filters "Name=group-id,Values=${group_id}" \
    --query "SecurityGroupRules[?Description=='Quantis SSH' || Description=='Quantis Isaac Sim WebRTC signaling' || Description=='Quantis Isaac Sim WebRTC media'].SecurityGroupRuleId" \
    --output text)"
  if [[ -n "${rule_ids}" && "${rule_ids}" != "None" ]]; then
    # shellcheck disable=SC2086
    aws_cli ec2 revoke-security-group-ingress --group-id "${group_id}" \
      --security-group-rule-ids ${rule_ids} >/dev/null
  fi

  aws_cli ec2 authorize-security-group-ingress --group-id "${group_id}" --ip-permissions \
    "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${source_cidr},Description='Quantis SSH'}]" \
    "IpProtocol=tcp,FromPort=${signal_port},ToPort=${signal_port},IpRanges=[{CidrIp=${source_cidr},Description='Quantis Isaac Sim WebRTC signaling'}]" \
    "IpProtocol=udp,FromPort=${stream_port},ToPort=${stream_port},IpRanges=[{CidrIp=${source_cidr},Description='Quantis Isaac Sim WebRTC media'}]" \
    >/dev/null
  printf 'Security group %s allows SSH and Isaac WebRTC from %s.\n' "${group_id}" "${source_cidr}" >&2
}

remote() {
  require_private_key
  ssh "${ssh_options[@]}" "${ssh_user}@$(instance_ip)" "$@"
}

remote_with_config() {
  local remote_command="env"
  local name
  local assignment
  for name in \
    ISAAC_SIM_VERSION \
    ISAAC_SIGNAL_PORT \
    ISAAC_STREAM_PORT \
    DOWNLOAD_PHYSICALAI_DATASET \
    HF_DOWNLOAD_MAX_WORKERS \
    QUANTIS_ASSET_HOME \
    DINOV3_CHECKPOINT_URL; do
    if declare -p "${name}" >/dev/null 2>&1; then
      printf -v assignment '%q' "${name}=${!name}"
      remote_command+=" ${assignment}"
    fi
  done
  remote "${remote_command} $1"
}

sync_repo() {
  require_private_key
  local option
  local quoted_option
  local remote_shell="ssh"
  for option in "${ssh_options[@]}"; do
    printf -v quoted_option '%q' "${option}"
    remote_shell+=" ${quoted_option}"
  done
  rsync -az --delete \
    --exclude .git --exclude .env --exclude .runtime --exclude .agents \
    --exclude data --exclude outputs \
    -e "${remote_shell}" \
    "${repo_root}/" "${ssh_user}@$(instance_ip):~/quantis-robotics/"
}

prepare_remote_host() {
  ensure_running
  configure_firewall
  sync_repo
}

bootstrap() {
  prepare_remote_host
  remote_with_config 'bash ~/quantis-robotics/ops/remote_bootstrap.sh'
}

up() {
  prepare_remote_host
  remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh start'
}

stop_instance() {
  local state
  state="$(instance_state)"
  case "${state}" in
    stopped)
      printf 'Instance %s is already stopped.\n' "${instance_id}"
      ;;
    stopping)
      aws_cli ec2 wait instance-stopped --instance-ids "${instance_id}"
      ;;
    running|pending)
      aws_cli ec2 stop-instances --instance-ids "${instance_id}" >/dev/null
      aws_cli ec2 wait instance-stopped --instance-ids "${instance_id}"
      printf 'Instance %s is stopped; EBS storage still incurs charges.\n' "${instance_id}"
      ;;
    *)
      die "instance ${instance_id} is ${state} and cannot be stopped"
      ;;
  esac
}

isaac_python() {
  local code="$1"
  local timeout_seconds="${2:-60}"
  local quoted_code
  local remote_command
  local response
  printf -v quoted_code '%q' "${code}"
  printf -v remote_command \
    "printf '%%s\\n' %s | timeout %q nc -q 3 127.0.0.1 8226" \
    "${quoted_code}" "${timeout_seconds}"
  response="$(remote "${remote_command}")"
  print_checked_isaac_response "${response}"
}

demo_python() {
  local expression="$1"
  local timeout_seconds="${2:-60}"
  sync_repo
  isaac_python "$(isaac_demo_code "${expression}")" "${timeout_seconds}"
}

wait_recording_job() {
  local recording_id="$1"
  local timeout_seconds="${DEMO_RECORDING_TIMEOUT_SECONDS:-2400}"
  local deadline=$((SECONDS + timeout_seconds))
  local response
  local status
  if remote "DEMO_RECORDING_TIMEOUT_SECONDS='${timeout_seconds}' bash ~/quantis-robotics/ops/wait_demo_recording.sh '${recording_id}'"; then
    return
  fi
  printf 'Recording wait disconnected; reconnecting to job %s.\n' \
    "${recording_id}" >&2
  while (( SECONDS < deadline )); do
    response="$(remote "job=~/docker/isaac-sim/data/quantis/recording_jobs/'${recording_id}'.json; test ! -f \"\${job}\" || cat \"\${job}\"")"
    if [[ -n "${response}" ]]; then
      printf '%s\n' "${response}"
      status="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' \
        <<<"${response}")"
      [[ "${status}" == "complete" ]]
      return
    fi
    sleep 5
  done
  die "recording job timed out: ${recording_id}"
}

record_exploration_job() {
  local recording_id="$1"
  local exploration_seed="$2"
  local dataset_split="$3"
  demo_python "demo.start_exploration_recording('${recording_id}',${exploration_seed},'${dataset_split}')"
  wait_recording_job "${recording_id}"
}

command="${1:-help}"
if [[ "${command}" == "help" ]]; then
  cat <<'EOF'
Usage: ./ops/aws.sh COMMAND

Commands:
  ensure-running | status | ip | firewall-webrtc | backup-state
  bootstrap                      Start, secure, sync, and bootstrap the host
  up                             Start, secure, sync, and start Isaac Sim
  down                           Stop the EC2 instance
  ssh | sync | remote-bootstrap
  isaac-start | isaac-stop | isaac-status | isaac-logs
  demo-reset | demo-preflight | demo-run | demo-capture | demo-record
  demo-record-actions             Capture a short 4 FPS JEPA-WM trajectory
  demo-record-exploration RECORDING SEED train|held_out
  demo-dashboard REFERENCE [primary-camera] [jepa-camera]
  capture-smoke | jepa-embed [source-name] [camera]
  jepa-stage-embed [recording-name] [camera]
  jepa-stage-report REFERENCE QUERY [camera]
  jepa-wm-install | jepa-wm-smoke | jepa-wm-status
  jepa-wm-eval RECORDING [camera] [start-index] [count] [stride]
  jepa-wm-adapt RECORDING [camera] [steps]
  jepa-wm-adapt-set RECORDING[,RECORDING...] [camera] [steps] [adapter-name]
  jepa-wm-plan-benchmark RECORDING [camera] [start] [count] [stride] [iterations] [samples] [elites] [adapter] [proposal]
  jepa-wm-proposal-train RECORDING[,RECORDING...] [camera] [steps] [proposal]
  jepa-wm-proposal-eval RECORDING [camera] [start] [count] [stride] [proposal]
  jepa-wm-proposal-summarize RECORDING[,RECORDING...] [camera] [start] [count] [stride] [proposal]
  jepa-wm-control-infer-replay RECORDING [camera] [context-index] [proposal]
  jepa-wm-control-worker-start [proposal] [adapter] | jepa-wm-control-worker-status | jepa-wm-control-worker-stop
  jepa-wm-control-step REFERENCE_RECORDING SEED [proposal] [adapter]
  jepa-wm-control-rollout REFERENCE_RECORDING SEED STEPS [proposal] [adapter]
  jepa-wm-control-baseline REFERENCE_RECORDING SEED STEPS zero|scripted
  jepa-wm-control-baselines EXPERIMENT DIRECT ZERO SCRIPTED REFERENCE SEED STEPS [direct-proposal]
  jepa-wm-control-rollout-report ROLLOUT SESSION[,SESSION...] REQUESTED_STEPS REFERENCE SEED [proposal]
  jepa-wm-control-apply SESSION
  jepa-wm-summarize EXPERIMENT TRAINING_CSV HELD_OUT_CSV [camera] [count]
  jepa-wm-milestone [train-count] [held-out-count] [steps] [base-seed]
  jepa-wm-eval-adapted RECORDING [camera] [start-index] [count] [stride]
EOF
  exit 0
fi

verify_identity
require_instance

case "${command}" in
  ensure-running)
    ensure_running
    ;;
  stop|down)
    stop_instance
    ;;
  status)
    aws_cli ec2 describe-instances --instance-ids "${instance_id}" \
      --query 'Reservations[0].Instances[0].{Id:InstanceId,Name:Tags[?Key==`Name`]|[0].Value,State:State.Name,Type:InstanceType,PublicIp:PublicIpAddress,PrivateIp:PrivateIpAddress,AZ:Placement.AvailabilityZone}'
    ;;
  ip)
    instance_ip
    ;;
  firewall-webrtc)
    configure_firewall
    ;;
  ssh)
    require_private_key
    exec ssh "${ssh_options[@]}" "${ssh_user}@$(instance_ip)"
    ;;
  sync)
    sync_repo
    ;;
  remote-bootstrap)
    remote_with_config 'bash ~/quantis-robotics/ops/remote_bootstrap.sh'
    ;;
  bootstrap)
    bootstrap
    ;;
  up)
    up
    ;;
  isaac-start)
    remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh start'
    ;;
  isaac-stop)
    remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh stop'
    ;;
  isaac-logs)
    require_private_key
    remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh logs'
    ;;
  isaac-status)
    remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh status'
    ;;
  backup-state)
    sync_repo
    remote_with_config 'bash ~/quantis-robotics/ops/backup_state.sh'
    ;;
  demo-preflight)
    demo_python 'demo.preflight_report()'
    ;;
  demo-reset)
    demo_python 'await demo.reset_demo()' 120
    ;;
  demo-run)
    demo_python 'await demo.run_demo()' 300
    ;;
  demo-capture)
    demo_python 'await demo.capture_cameras()' 180
    ;;
  demo-record)
    recording_id="demo-$(date -u +%Y%m%dT%H%M%SZ)"
    demo_python "demo.start_recording('${recording_id}')"
    wait_recording_job "${recording_id}"
    remote "bash ~/quantis-robotics/ops/encode_demo_recording.sh '${recording_id}'"
    ;;
  demo-record-actions)
    recording_id="trajectory-$(date -u +%Y%m%dT%H%M%SZ)"
    demo_python "demo.start_action_recording('${recording_id}')"
    wait_recording_job "${recording_id}"
    remote "bash ~/quantis-robotics/ops/encode_demo_recording.sh '${recording_id}'"
    printf 'Recording ID: %s\n' "${recording_id}"
    ;;
  demo-record-exploration)
    recording_id="${2:-}"
    exploration_seed="${3:-}"
    dataset_split="${4:-}"
    is_safe_identifier "${recording_id}" || die "invalid recording name"
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    [[ "${dataset_split}" == "train" || "${dataset_split}" == "held_out" ]] \
      || die "dataset split must be train or held_out"
    record_exploration_job \
      "${recording_id}" "${exploration_seed}" "${dataset_split}"
    ;;
  demo-dashboard)
    reference_name="${2:-}"
    primary_camera="${3:-wrist}"
    jepa_camera="${4:-wrist}"
    is_safe_identifier "${reference_name}" || die "invalid reference recording name"
    is_safe_identifier "${primary_camera}" || die "invalid primary camera name"
    is_safe_identifier "${jepa_camera}" || die "invalid JEPA camera name"
    recording_id="demo-$(date -u +%Y%m%dT%H%M%SZ)"
    demo_python "demo.start_recording('${recording_id}')"
    DEMO_RECORDING_TIMEOUT_SECONDS=2400 wait_recording_job "${recording_id}"
    remote "bash ~/quantis-robotics/ops/encode_demo_recording.sh '${recording_id}'"
    remote "bash ~/quantis-robotics/ops/jepa_stages.sh report '${reference_name}' '${recording_id}' '${jepa_camera}'"
    remote "bash ~/quantis-robotics/ops/render_demo_dashboard.sh '${recording_id}' '${primary_camera}' '${jepa_camera}'"
    printf 'Recording ID: %s\n' "${recording_id}"
    ;;
  capture-smoke)
    remote_with_config 'bash ~/quantis-robotics/ops/isaac_container.sh capture-smoke'
    ;;
  jepa-embed)
    source_name="${2:-latest}"
    camera_name="${3:-wrist}"
    is_safe_identifier "${source_name}" || die "invalid JEPA source name"
    is_safe_identifier "${camera_name}" || die "invalid JEPA camera name"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_embed.sh '${source_name}' '${camera_name}'"
    ;;
  jepa-stage-embed)
    recording_name="${2:-latest}"
    camera_name="${3:-wrist}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid JEPA camera name"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_stages.sh embed '${recording_name}' '${camera_name}'"
    ;;
  jepa-stage-report)
    reference_name="${2:-}"
    query_name="${3:-}"
    camera_name="${4:-wrist}"
    is_safe_identifier "${reference_name}" || die "invalid reference recording name"
    is_safe_identifier "${query_name}" || die "invalid query recording name"
    is_safe_identifier "${camera_name}" || die "invalid JEPA camera name"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_stages.sh report '${reference_name}' '${query_name}' '${camera_name}'"
    ;;
  jepa-wm-install)
    sync_repo
    remote_with_config "bash ~/quantis-robotics/ops/jepa_wm.sh install"
    ;;
  jepa-wm-smoke)
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh smoke"
    ;;
  jepa-wm-status)
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh status"
    ;;
  jepa-wm-eval|jepa-wm-eval-adapted)
    recording_name="${2:-}"
    camera_name="${3:-wrist}"
    start_index="${4:-0}"
    transition_count="${5:-8}"
    transition_stride="${6:-1}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "transition count" "${transition_count}" || exit 1
    require_positive_integer "transition stride" "${transition_stride}" || exit 1
    evaluation_mode="base"
    [[ "${command}" == "jepa-wm-eval-adapted" ]] && evaluation_mode="adapted"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh evaluate '${recording_name}' '${camera_name}' '${start_index}' '${transition_count}' '${transition_stride}' '${evaluation_mode}'"
    ;;
  jepa-wm-adapt)
    recording_name="${2:-}"
    camera_name="${3:-wrist}"
    training_steps="${4:-100}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    require_positive_integer "training steps" "${training_steps}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh adapt '${recording_name}' '${camera_name}' '${training_steps}'"
    ;;
  jepa-wm-adapt-set)
    recording_names="${2:-}"
    camera_name="${3:-wrist}"
    training_steps="${4:-500}"
    adapter_name="${5:-quantis_isaac_${camera_name}_action_adapter}"
    is_safe_identifier_list "${recording_names}" \
      || die "invalid training recording list"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${adapter_name}" || die "invalid adapter name"
    require_positive_integer "training steps" "${training_steps}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh adapt-set '${recording_names}' '${camera_name}' '${training_steps}' '${adapter_name}'"
    ;;
  jepa-wm-plan-benchmark)
    recording_name="${2:-}"
    camera_name="${3:-wrist}"
    start_index="${4:-0}"
    rollout_count="${5:-8}"
    rollout_stride="${6:-1}"
    planner_iterations="${7:-6}"
    planner_samples="${8:-300}"
    planner_elites="${9:-10}"
    adapter_name="${10:-quantis_isaac_${camera_name}_action_adapter}"
    proposal_name="${11:-}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${adapter_name}" || die "invalid adapter name"
    if [[ -n "${proposal_name}" ]]; then
      is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    fi
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
    require_positive_integer "planner iterations" "${planner_iterations}" || exit 1
    require_positive_integer "planner samples" "${planner_samples}" || exit 1
    require_positive_integer "planner elites" "${planner_elites}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh plan-benchmark --recording '${recording_name}' --camera '${camera_name}' --start-index '${start_index}' --count '${rollout_count}' --stride '${rollout_stride}' --iterations '${planner_iterations}' --samples '${planner_samples}' --elites '${planner_elites}' --adapter '${adapter_name}' --proposal '${proposal_name}'"
    ;;
  jepa-wm-proposal-train)
    recording_names="${2:-}"
    camera_name="${3:-wrist}"
    training_steps="${4:-2000}"
    proposal_name="${5:-quantis_isaac_${camera_name}_action_proposal}"
    is_safe_identifier_list "${recording_names}" \
      || die "invalid training recording list"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    require_positive_integer "training steps" "${training_steps}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh proposal-train --recordings '${recording_names}' --camera '${camera_name}' --steps '${training_steps}' --proposal '${proposal_name}'"
    ;;
  jepa-wm-proposal-eval)
    recording_name="${2:-}"
    camera_name="${3:-wrist}"
    start_index="${4:-0}"
    rollout_count="${5:-8}"
    rollout_stride="${6:-1}"
    proposal_name="${7:-quantis_isaac_${camera_name}_action_proposal}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh proposal-eval --recording '${recording_name}' --camera '${camera_name}' --start-index '${start_index}' --count '${rollout_count}' --stride '${rollout_stride}' --proposal '${proposal_name}'"
    ;;
  jepa-wm-proposal-summarize)
    recording_names="${2:-}"
    camera_name="${3:-wrist}"
    start_index="${4:-4}"
    rollout_count="${5:-62}"
    rollout_stride="${6:-1}"
    proposal_name="${7:-quantis_isaac_${camera_name}_action_proposal}"
    is_safe_identifier_list "${recording_names}" \
      || die "invalid held-out recording list"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    require_nonnegative_integer "start index" "${start_index}" || exit 1
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    require_positive_integer "rollout stride" "${rollout_stride}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh proposal-summarize --recordings '${recording_names}' --camera '${camera_name}' --start-index '${start_index}' --count '${rollout_count}' --stride '${rollout_stride}' --proposal '${proposal_name}'"
    ;;
  jepa-wm-control-infer-replay)
    recording_name="${2:-}"
    camera_name="${3:-wrist}"
    context_index="${4:-4}"
    proposal_name="${5:-quantis_isaac_${camera_name}_action_proposal}"
    is_safe_identifier "${recording_name}" || die "invalid recording name"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    require_nonnegative_integer "context index" "${context_index}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-infer-replay --recording '${recording_name}' --camera '${camera_name}' --context-index '${context_index}' --observation-id '1' --proposal '${proposal_name}'"
    ;;
  jepa-wm-control-worker-start)
    proposal_name="${2:-quantis_isaac_wrist_action_proposal}"
    adapter_name="${3:-quantis_isaac_wrist_action_adapter}"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    is_safe_identifier "${adapter_name}" || die "invalid adapter name"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-worker-start --proposal '${proposal_name}' --adapter '${adapter_name}'"
    ;;
  jepa-wm-control-worker-status)
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-worker-status"
    ;;
  jepa-wm-control-worker-stop)
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-worker-stop"
    ;;
  jepa-wm-control-step)
    reference_name="${2:-}"
    exploration_seed="${3:-}"
    proposal_name="${4:-quantis_isaac_wrist_action_proposal}"
    adapter_name="${5:-quantis_isaac_wrist_action_adapter}"
    is_safe_identifier "${reference_name}" || die "invalid reference recording"
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    is_safe_identifier "${adapter_name}" || die "invalid adapter name"
    session_id="step-$(date -u +%Y%m%dT%H%M%SZ)-${exploration_seed}"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-worker-start --proposal '${proposal_name}' --adapter '${adapter_name}'"
    remote "bash ~/quantis-robotics/ops/run_control_step.sh '${session_id}' '${reference_name}' '${exploration_seed}' '${proposal_name}'"
    printf 'Control session: %s\n' "${session_id}"
    ;;
  jepa-wm-control-rollout)
    reference_name="${2:-}"
    exploration_seed="${3:-}"
    step_count="${4:-3}"
    proposal_name="${5:-quantis_isaac_wrist_action_proposal}"
    adapter_name="${6:-quantis_isaac_wrist_action_adapter}"
    is_safe_identifier "${reference_name}" || die "invalid reference recording"
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    require_positive_integer "step count" "${step_count}" || exit 1
    (( step_count <= 8 )) || die "control rollout is capped at eight steps"
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    is_safe_identifier "${adapter_name}" || die "invalid adapter name"
    rollout_id="rollout-$(date -u +%Y%m%dT%H%M%SZ)-${exploration_seed}"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-worker-start --proposal '${proposal_name}' --adapter '${adapter_name}'"
    remote "bash ~/quantis-robotics/ops/run_control_rollout.sh '${rollout_id}' '${reference_name}' '${exploration_seed}' '${step_count}' '${proposal_name}'"
    printf 'Control rollout: %s\n' "${rollout_id}"
    ;;
  jepa-wm-control-baseline)
    reference_name="${2:-}"
    exploration_seed="${3:-}"
    step_count="${4:-3}"
    policy="${5:-}"
    is_safe_identifier "${reference_name}" || die "invalid reference recording"
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    require_positive_integer "step count" "${step_count}" || exit 1
    (( step_count <= 8 )) || die "control rollout is capped at eight steps"
    [[ "${policy}" == "zero" || "${policy}" == "scripted" ]] \
      || die "baseline policy must be zero or scripted"
    proposal_name="baseline_${policy}"
    rollout_id="${policy}-$(date -u +%Y%m%dT%H%M%SZ)-${exploration_seed}"
    sync_repo
    remote "bash ~/quantis-robotics/ops/run_control_rollout.sh '${rollout_id}' '${reference_name}' '${exploration_seed}' '${step_count}' '${proposal_name}' '${policy}'"
    printf 'Control baseline: %s\n' "${rollout_id}"
    ;;
  jepa-wm-control-baselines)
    experiment_id="${2:-}"
    direct_rollout="${3:-}"
    zero_rollout="${4:-}"
    scripted_rollout="${5:-}"
    reference_name="${6:-}"
    exploration_seed="${7:-}"
    step_count="${8:-3}"
    proposal_name="${9:-quantis_isaac_wrist_action_proposal}"
    for identifier in \
      "${experiment_id}" "${direct_rollout}" "${zero_rollout}" \
      "${scripted_rollout}" "${reference_name}" "${proposal_name}"; do
      is_safe_identifier "${identifier}" || die "invalid baseline comparison identifier"
    done
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    require_positive_integer "step count" "${step_count}" || exit 1
    (( step_count <= 8 )) || die "control rollout is capped at eight steps"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-baseline-report --experiment '${experiment_id}' --reference '${reference_name}' --seed '${exploration_seed}' --requested-steps '${step_count}' --direct-rollout '${direct_rollout}' --zero-rollout '${zero_rollout}' --scripted-rollout '${scripted_rollout}' --direct-proposal '${proposal_name}'"
    ;;
  jepa-wm-control-rollout-report)
    rollout_id="${2:-}"
    sessions="${3:-}"
    requested_steps="${4:-}"
    reference_name="${5:-}"
    exploration_seed="${6:-}"
    proposal_name="${7:-quantis_isaac_wrist_action_proposal}"
    is_safe_identifier "${rollout_id}" || die "invalid control rollout"
    is_safe_identifier_list "${sessions}" || die "invalid control session list"
    require_positive_integer "requested steps" "${requested_steps}" || exit 1
    (( requested_steps <= 8 )) || die "control rollout is capped at eight steps"
    is_safe_identifier "${reference_name}" || die "invalid reference recording"
    require_nonnegative_integer "exploration seed" "${exploration_seed}" || exit 1
    is_safe_identifier "${proposal_name}" || die "invalid proposal name"
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh control-rollout-report --rollout '${rollout_id}' --reference '${reference_name}' --seed '${exploration_seed}' --proposal '${proposal_name}' --sessions '${sessions}' --requested-steps '${requested_steps}'"
    ;;
  jepa-wm-control-apply)
    session_id="${2:-}"
    is_safe_identifier "${session_id}" || die "invalid control session"
    demo_python "await demo.apply_control_response('${session_id}')" 180
    ;;
  jepa-wm-summarize)
    experiment_id="${2:-}"
    training_recordings="${3:-}"
    held_out_recordings="${4:-}"
    camera_name="${5:-wrist}"
    rollout_count="${6:-40}"
    is_safe_identifier "${experiment_id}" || die "invalid experiment name"
    is_safe_identifier_list "${training_recordings}" \
      || die "invalid training recording list"
    is_safe_identifier_list "${held_out_recordings}" \
      || die "invalid held-out recording list"
    is_safe_identifier "${camera_name}" || die "invalid camera name"
    require_positive_integer "rollout count" "${rollout_count}" || exit 1
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh summarize '${experiment_id}' '${training_recordings}' '${held_out_recordings}' '${camera_name}' '${rollout_count}'"
    ;;
  jepa-wm-milestone)
    exec bash "${repo_root}/ops/jepa_wm_milestone.sh" \
      "${2:-4}" "${3:-2}" "${4:-500}" "${5:-1200}"
    ;;
  *)
    die "unknown command: ${command} (run $0 help)"
    ;;
esac
