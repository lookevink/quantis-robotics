#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${repo_root}/ops/cloudwatch-agent.json"
agent_ctl="/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ -x "${agent_ctl}" ]] \
  || die "CloudWatch agent is not installed at ${agent_ctl}"
[[ -f "${config_file}" ]] || die "missing agent config: ${config_file}"
command -v nvidia-smi >/dev/null 2>&1 \
  || die "nvidia-smi is unavailable; GPU metrics cannot be collected"

case "${1:-status}" in
  enable)
    sudo "${agent_ctl}" -a fetch-config -m ec2 -s -c "file:${config_file}"
    ;;
  status)
    sudo "${agent_ctl}" -a status
    ;;
  *)
    die "usage: $0 enable|status"
    ;;
esac
