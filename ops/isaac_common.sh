#!/usr/bin/env bash

isaac_home="${HOME}/docker/isaac-sim"
asset_home="${QUANTIS_ASSET_HOME:-${HOME}/quantis-assets}"
isaac_version="${ISAAC_SIM_VERSION:-6.0.1}"
isaac_image="nvcr.io/nvidia/isaac-sim:${isaac_version}"

isaac_mounts=(
  -v "${isaac_home}/cache/main:/isaac-sim/.cache:rw"
  -v "${isaac_home}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw"
  -v "${isaac_home}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
  -v "${isaac_home}/config:/isaac-sim/.nvidia-omniverse/config:rw"
  -v "${isaac_home}/data:/isaac-sim/.local/share/ov/data:rw"
  -v "${isaac_home}/pkg:/isaac-sim/.local/share/ov/pkg:rw"
  -v "${asset_home}:/assets:ro"
)
