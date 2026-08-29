#!/usr/bin/env bash

isaac_home="${HOME}/docker/isaac-sim"
jepa_wm_checkpoint_dir="${JEPA_WM_CHECKPOINT_DIR:-${HOME}/docker/jepa-wm/checkpoints}"
asset_home="${QUANTIS_ASSET_HOME:-${HOME}/quantis-assets}"
isaac_version="${ISAAC_SIM_VERSION:-6.0.1}"
isaac_image="nvcr.io/nvidia/isaac-sim:${isaac_version}"
isaac_runtime_user="1234:1234"
if [[ -d "${jepa_wm_checkpoint_dir}" ]]; then
  if ! isaac_checkpoint_group_id="$(stat -c '%g' "${jepa_wm_checkpoint_dir}" 2>/dev/null)"; then
    isaac_checkpoint_group_id="$(stat -f '%g' "${jepa_wm_checkpoint_dir}")"
  fi
else
  isaac_checkpoint_group_id="$(id -g)"
fi

isaac_checkpoint_access_args=(
  --user "${isaac_runtime_user}"
  --group-add "${isaac_checkpoint_group_id}"
  -v "${jepa_wm_checkpoint_dir}:${jepa_wm_checkpoint_dir}:ro"
)

isaac_mounts=(
  -v "${isaac_home}/cache/main:/isaac-sim/.cache:rw"
  -v "${isaac_home}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw"
  -v "${isaac_home}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
  -v "${isaac_home}/config:/isaac-sim/.nvidia-omniverse/config:rw"
  -v "${isaac_home}/data:/isaac-sim/.local/share/ov/data:rw"
  -v "${isaac_home}/pkg:/isaac-sim/.local/share/ov/pkg:rw"
  -v "${asset_home}:/assets:ro"
)
