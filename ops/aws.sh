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
ssh_options=(-o StrictHostKeyChecking=accept-new)

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
  ssh_options=(-i "${private_key}" -o StrictHostKeyChecking=accept-new)
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
  local remote_shell
  printf -v remote_shell 'ssh -i %q -o StrictHostKeyChecking=accept-new' "${private_key}"
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
  printf '%s\n' "${response}"
  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"' <<<"${response}"; then
    return 1
  fi
}

demo_python() {
  local expression="$1"
  local timeout_seconds="${2:-60}"
  sync_repo
  isaac_python "import sys,json,importlib; sys.path.insert(0,'/workspace') if '/workspace' not in sys.path else None; importlib.invalidate_caches(); import jepa.contract as contract; importlib.reload(contract); import sim.recording as recording; importlib.reload(recording); import sim.recording_jobs as recording_jobs, sim.isaac_demo_scene as scene, sim.isaac_demo_camera as camera, sim.isaac_demo_kinematics as kinematics, sim.isaac_demo as demo; importlib.reload(recording_jobs); importlib.reload(scene); importlib.reload(camera); importlib.reload(kinematics); importlib.reload(demo); print(json.dumps(${expression},indent=2))" "${timeout_seconds}"
}

command="${1:-help}"
if [[ "${command}" == "help" ]]; then
  cat <<'EOF'
Usage: ./ops/aws.sh COMMAND

Commands:
  ensure-running | status | ip | firewall-webrtc
  bootstrap                      Start, secure, sync, and bootstrap the host
  up                             Start, secure, sync, and start Isaac Sim
  down                           Stop the EC2 instance
  ssh | sync | remote-bootstrap
  isaac-start | isaac-stop | isaac-status | isaac-logs
  demo-reset | demo-preflight | demo-run | demo-capture | demo-record
  demo-record-actions             Capture a short 4 FPS JEPA-WM trajectory
  demo-dashboard REFERENCE [primary-camera] [jepa-camera]
  capture-smoke | jepa-embed [source-name] [camera]
  jepa-stage-embed [recording-name] [camera]
  jepa-stage-report REFERENCE QUERY [camera]
  jepa-wm-install | jepa-wm-smoke | jepa-wm-status
  jepa-wm-eval RECORDING [camera] [start-index] [count] [stride]
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
    remote "bash ~/quantis-robotics/ops/wait_demo_recording.sh '${recording_id}'"
    remote "bash ~/quantis-robotics/ops/encode_demo_recording.sh '${recording_id}'"
    ;;
  demo-record-actions)
    recording_id="trajectory-$(date -u +%Y%m%dT%H%M%SZ)"
    demo_python "demo.start_action_recording('${recording_id}')"
    remote "bash ~/quantis-robotics/ops/wait_demo_recording.sh '${recording_id}'"
    remote "bash ~/quantis-robotics/ops/encode_demo_recording.sh '${recording_id}'"
    printf 'Recording ID: %s\n' "${recording_id}"
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
    remote "DEMO_RECORDING_TIMEOUT_SECONDS=2400 bash ~/quantis-robotics/ops/wait_demo_recording.sh '${recording_id}'"
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
  jepa-wm-eval)
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
    sync_repo
    remote "bash ~/quantis-robotics/ops/jepa_wm.sh evaluate '${recording_name}' '${camera_name}' '${start_index}' '${transition_count}' '${transition_stride}'"
    ;;
  *)
    die "unknown command: ${command} (run $0 help)"
    ;;
esac
