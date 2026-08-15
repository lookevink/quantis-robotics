#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${ENV_FILE:-${repo_root}/.env}"

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

api_base="https://cloud.lambda.ai/api/v1"
instance_type="${LAMBDA_INSTANCE_TYPE:-gpu_1x_a10}"
region="${LAMBDA_REGION:-us-west-1}"
key_name="${LAMBDA_SSH_KEY_NAME:-quantis-robotics-mac}"
instance_name="${LAMBDA_INSTANCE_NAME:-quantis-isaac-sim}"
private_key="${LAMBDA_SSH_PRIVATE_KEY:-${HOME}/.ssh/github_signing_ed25519}"
signal_port="${ISAAC_SIGNAL_PORT:-49100}"
stream_port="${ISAAC_STREAM_PORT:-47998}"
ssh_options=(-o StrictHostKeyChecking=accept-new)

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_api_key() {
  [[ -n "${LAMBDA_API_KEY:-}" ]] || die "LAMBDA_API_KEY is missing from ${env_file}"
}

api() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  require_api_key

  if [[ -n "${body}" ]]; then
    curl --fail --silent --show-error \
      --request "${method}" \
      --url "${api_base}${path}" \
      --header 'accept: application/json' \
      --header 'content-type: application/json' \
      --header "Authorization: Bearer ${LAMBDA_API_KEY}" \
      --data-binary "${body}"
  else
    curl --fail --silent --show-error \
      --request "${method}" \
      --url "${api_base}${path}" \
      --header 'accept: application/json' \
      --header "Authorization: Bearer ${LAMBDA_API_KEY}"
  fi
}

resolve_instance() {
  if [[ -n "${LAMBDA_INSTANCE_ID:-}" ]]; then
    printf '%s\n' "${LAMBDA_INSTANCE_ID}"
    return
  fi

  api GET /instances | jq -er --arg name "${instance_name}" \
    '.data | map(select(.name == $name and .status != "terminated")) | first | .id' \
    || die "no active instance named ${instance_name}; set LAMBDA_INSTANCE_ID"
}

instance_json() {
  local id
  id="$(resolve_instance)"
  api GET "/instances/${id}"
}

instance_ip() {
  instance_json | jq -er '.data.ip // empty' || die "instance has no public IP yet"
}

require_private_key() {
  [[ -n "${private_key}" ]] || die "set LAMBDA_SSH_PRIVATE_KEY in .env"
  [[ -f "${private_key}" ]] || die "SSH private key does not exist: ${private_key}"
  ssh_options=(-i "${private_key}" -o StrictHostKeyChecking=accept-new)
}

command="${1:-help}"
case "${command}" in
  capacity)
    api GET /instance-types | jq --arg type "${instance_type}" '.data[$type]'
    ;;
  register-key)
    public_key_path="${2:-}"
    [[ -f "${public_key_path}" ]] || die "usage: $0 register-key /path/to/key.pub"
    public_key="$(<"${public_key_path}")"
    body="$(jq -cn --arg name "${key_name}" --arg public_key "${public_key}" '{name:$name,public_key:$public_key}')"
    api POST /ssh-keys "${body}" | jq '.data | {id,name,public_key}'
    ;;
  launch)
    body="$(jq -cn \
      --arg region "${region}" \
      --arg type "${instance_type}" \
      --arg key "${key_name}" \
      --arg name "${instance_name}" \
      '{region_name:$region,instance_type_name:$type,ssh_key_names:[$key],file_system_names:[],quantity:1,name:$name}')"
    api POST /instance-operations/launch "${body}" | jq '.data'
    ;;
  list)
    api GET /instances | jq '.data | map({id,name,status,ip,region:.region.name,instance_type:.instance_type.name})'
    ;;
  status)
    instance_json | jq '.data | {id,name,status,ip,region:.region.name,instance_type:.instance_type.name}'
    ;;
  ip)
    instance_ip
    ;;
  firewall-webrtc)
    source_cidr="${WEBRTC_SOURCE_CIDR:-$(curl --fail --silent --show-error https://api.ipify.org)/32}"
    current_rules="$(api GET /firewall-rulesets/global)"
    body="$(printf '%s' "${current_rules}" | jq -c --arg source "${source_cidr}" --argjson signal "${signal_port}" --argjson stream "${stream_port}" '
      .data.rules
      | (if any(.protocol == "tcp" and .port_range == [$signal,$signal] and .source_network == $source)
         then . else . + [{protocol:"tcp",port_range:[$signal,$signal],source_network:$source,description:"Quantis Isaac Sim WebRTC signaling"}] end)
      | (if any(.protocol == "udp" and .port_range == [$stream,$stream] and .source_network == $source)
         then . else . + [{protocol:"udp",port_range:[$stream,$stream],source_network:$source,description:"Quantis Isaac Sim WebRTC media"}] end)
      | {rules:.}'
    )"
    api PATCH /firewall-rulesets/global "${body}" | jq --arg source "${source_cidr}" --argjson signal "${signal_port}" --argjson stream "${stream_port}" \
      '.data | {id,source:$source,webrtc_rules:[.rules[] | select(.source_network == $source and (.port_range == [$signal,$signal] or .port_range == [$stream,$stream]))]}'
    ;;
  ssh)
    require_private_key
    exec ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)"
    ;;
  sync)
    require_private_key
    rsync -az --delete \
      --exclude .git --exclude .env --exclude .runtime --exclude data --exclude outputs \
      -e "ssh -i ${private_key} -o StrictHostKeyChecking=accept-new" \
      "${repo_root}/" "ubuntu@$(instance_ip):~/quantis-robotics/"
    ;;
  remote-bootstrap)
    require_private_key
    ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" 'bash ~/quantis-robotics/ops/remote_bootstrap.sh'
    ;;
  isaac-start)
    require_private_key
    ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" 'bash ~/quantis-robotics/ops/isaac_container.sh start'
    ;;
  isaac-stop)
    require_private_key
    ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" 'bash ~/quantis-robotics/ops/isaac_container.sh stop'
    ;;
  isaac-logs)
    require_private_key
    exec ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" 'bash ~/quantis-robotics/ops/isaac_container.sh logs'
    ;;
  capture-smoke)
    require_private_key
    ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" 'bash ~/quantis-robotics/ops/isaac_container.sh capture-smoke'
    ;;
  jepa-embed)
    require_private_key
    episode_name="${2:-latest}"
    [[ "${episode_name}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid episode name"
    ssh "${ssh_options[@]}" "ubuntu@$(instance_ip)" \
      "bash ~/quantis-robotics/ops/jepa_embed.sh '${episode_name}'"
    ;;
  terminate)
    [[ "${2:-}" == "--yes" ]] || die "termination is irreversible; rerun: $0 terminate --yes"
    id="$(resolve_instance)"
    body="$(jq -cn --arg id "${id}" '{instance_ids:[$id]}')"
    api POST /instance-operations/terminate "${body}" | jq '.data.terminated_instances | map({id,name,status})'
    ;;
  help|*)
    cat <<'EOF'
Usage: ./ops/lambda.sh COMMAND

Commands:
  capacity
  register-key /path/to/key.pub
  launch
  list | status | ip | firewall-webrtc
  ssh | sync | remote-bootstrap
  isaac-start | isaac-stop | isaac-logs
  capture-smoke | jepa-embed [episode-name]
  terminate --yes
EOF
    ;;
esac
