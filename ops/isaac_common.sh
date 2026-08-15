#!/usr/bin/env bash

isaac_home="${HOME}/docker/isaac-sim"
isaac_version="${ISAAC_SIM_VERSION:-5.0.0}"
isaac_image="nvcr.io/nvidia/isaac-sim:${isaac_version}"

isaac_mounts=(
  -v "${isaac_home}/cache/kit:/isaac-sim/kit/cache:rw"
  -v "${isaac_home}/cache/ov:/root/.cache/ov:rw"
  -v "${isaac_home}/cache/pip:/root/.cache/pip:rw"
  -v "${isaac_home}/cache/glcache:/root/.cache/nvidia/GLCache:rw"
  -v "${isaac_home}/cache/computecache:/root/.nv/ComputeCache:rw"
  -v "${isaac_home}/logs:/root/.nvidia-omniverse/logs:rw"
  -v "${isaac_home}/data:/root/.local/share/ov/data:rw"
  -v "${isaac_home}/documents:/root/Documents:rw"
)
