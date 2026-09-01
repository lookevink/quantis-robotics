#!/usr/bin/env bash
set -euo pipefail

isaac_data_root="${ISAAC_DATA_ROOT:-${HOME}/docker/isaac-sim/data}"
checkpoint_dir="${JEPA_WM_CHECKPOINT_DIR:-${HOME}/docker/jepa-wm/checkpoints}"
asset_home="${QUANTIS_ASSET_HOME:-/mnt/quantis-assets}"
backup_root="${asset_home}/quantis-state"
findmnt_command="${FINDMNT_COMMAND:-findmnt}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

sync_tree() {
  local source="$1"
  local destination="$2"
  local remaining
  [[ -d "${source}" ]] || die "backup source does not exist: ${source}"
  if ! mkdir -p "${destination}" 2>/dev/null \
    || [[ ! -w "${destination}" ]]; then
    command -v sudo >/dev/null 2>&1 \
      || die "backup destination is not writable and sudo is unavailable: ${destination}"
    sudo install -d -o "$(id -u)" -g "$(id -g)" "${destination}"
  fi
  rsync -a "${source}/" "${destination}/"
  remaining="$(rsync -anic "${source}/" "${destination}/")"
  [[ -z "${remaining}" ]] \
    || die "checksum verification failed for ${source}"
}

ensure_writable_backup_root() {
  mkdir -p "${backup_root}" 2>/dev/null || true
  if [[ ! -w "${backup_root}" ]]; then
    command -v sudo >/dev/null 2>&1 \
      || die "backup root is not writable and sudo is unavailable: ${backup_root}"
    sudo chown "$(id -u):$(id -g)" "${backup_root}"
  fi
}

require_dedicated_backup_filesystem() {
  local asset_mount_target
  local asset_device_id
  local source_root
  local source_device_id
  local resolved_asset_home
  local resolved_mount_target

  [[ -d "${asset_home}" ]] \
    || die "asset mount directory does not exist: ${asset_home}"
  command -v "${findmnt_command}" >/dev/null 2>&1 \
    || die "findmnt is required to verify the backup volume"

  asset_mount_target="$("${findmnt_command}" -n -T "${asset_home}" -o TARGET)" \
    || die "cannot resolve the backup mount for ${asset_home}"
  resolved_asset_home="$(cd "${asset_home}" && pwd -P)"
  resolved_mount_target="$(cd "${asset_mount_target}" && pwd -P)"
  [[ "${resolved_mount_target}" == "${resolved_asset_home}" ]] \
    || die "backup destination is not a dedicated mount point: ${asset_home}"

  asset_device_id="$("${findmnt_command}" -n -T "${asset_home}" -o MAJ:MIN)" \
    || die "cannot resolve the backup filesystem identity for ${asset_home}"
  for source_root in "${isaac_data_root}" "${checkpoint_dir}"; do
    [[ -d "${source_root}" ]] || die "backup source does not exist: ${source_root}"
    source_device_id="$("${findmnt_command}" -n -T "${source_root}" -o MAJ:MIN)" \
      || die "cannot resolve the source filesystem identity for ${source_root}"
    [[ "${source_device_id}" != "${asset_device_id}" ]] \
      || die "backup destination shares a filesystem with ${source_root}"
  done
}

command -v rsync >/dev/null 2>&1 || die "rsync is not installed"
require_dedicated_backup_filesystem
ensure_writable_backup_root
sync_tree "${isaac_data_root}/quantis/scenes" "${backup_root}/isaac/scenes"
sync_tree "${isaac_data_root}/quantis/recordings" "${backup_root}/isaac/recordings"
for state_tree in \
  control_sessions control_rollouts control_baselines control_readiness \
  control_candidates unknown_start_reset_claims; do
  if [[ -d "${isaac_data_root}/quantis/${state_tree}" ]]; then
    sync_tree \
      "${isaac_data_root}/quantis/${state_tree}" \
      "${backup_root}/isaac/${state_tree}"
  fi
done
sync_tree "${checkpoint_dir}" "${backup_root}/jepa-wm/checkpoints"
date -u +%Y-%m-%dT%H:%M:%SZ >"${backup_root}/LAST_BACKUP_UTC"

printf 'Quantis state backup verified at %s\n' "${backup_root}"
du -sh "${backup_root}"
