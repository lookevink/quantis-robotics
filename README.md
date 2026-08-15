# Quantis robotics bootstrap

Simulation-first JEPA experiments for a narrow data-center manipulation demo.

The first embodiment is the **Franka Panda** because:

- Isaac Sim ships a supported Franka USD asset.
- NVIDIA's PhysicalAI single-arm dataset uses Franka with RGB, state, and 7D/8D actions.
- Meta's V-JEPA 2-AC robot experiments use Franka/DROID trajectories.

The NVIDIA Physical AI Hugging Face collection is mostly training datasets, not a catalog of drop-in robot USD assets. This project uses Isaac Sim's built-in Franka asset and treats [`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm) as a compatible reference dataset.

The Lambda A10 image initially exposed only CUDA libraries. Bootstrap installs the matching Lambda graphics/Vulkan driver package and requires a reboot when that advances the driver; this instance now uses `570.195.03`. Isaac Sim 6.0 requires a newer 595-series driver, so the VM is deliberately pinned to Isaac Sim `5.0.0`. Do not change the image tag without rerunning the rendered-frame smoke test.

## Current architecture

```text
Isaac Sim 5.0.0
  ├─ Franka + rack/module scene
  ├─ RGB capture (Replicator)
  ├─ action + robot state (JSONL)
  └─ scripted / motion-planning controller
              │
              ▼
episode dataset: frames + actions + state + manifest
              │
              ▼
frozen V-JEPA 2 encoder → goal-progress/stage model
              │
              ▼
high-level subgoal → Isaac motion controller
```

For the first demo, JEPA estimates visual state and progress while Isaac executes motion. Action-conditioned world-model planning is a second integration seam after capture and closed-loop control are working.

## 1. Lambda instance

Prerequisites:

- `.env` containing `LAMBDA_API_KEY`.
- An existing local SSH keypair.
- `curl` and `jq` locally.

```bash
./ops/lambda.sh capacity
./ops/lambda.sh register-key ~/.ssh/id_ed25519.pub
./ops/lambda.sh launch
./ops/lambda.sh status
./ops/lambda.sh ssh
```

`launch` is billable. Lambda does not suspend instances; terminate them when finished:

```bash
./ops/lambda.sh terminate --yes
```

## 2. Bootstrap the remote host

Copy this repository to the active instance, then bootstrap and start Isaac Sim:

```bash
./ops/lambda.sh sync
./ops/lambda.sh remote-bootstrap
./ops/lambda.sh firewall-webrtc
./ops/lambda.sh isaac-start
./ops/lambda.sh isaac-logs
```

If bootstrap installs a newer matched NVIDIA graphics package, it exits with a reboot instruction. Reboot the instance and rerun `remote-bootstrap`. Wait for this final streaming log line:

```text
Streaming server started.
```

## 3. Stream the UI

Isaac Sim uses TCP `49100` for WebRTC signaling and UDP `47998` for media. `firewall-webrtc` adds both to Lambda's global rules, restricted to your current public IP. Override it with `WEBRTC_SOURCE_CIDR` if needed. The container uses host networking; Docker bridge port publishing is not sufficient for Isaac WebRTC.

Install the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/manual_livestream_clients.html) on the local workstation and connect to the Lambda instance's public IP printed by `./ops/lambda.sh ip`.

Streaming is only remote display/control. It is **not** the training-data capture path.

## 4. Capture training data

Run the standalone capture smoke test inside the container:

```bash
./ops/lambda.sh capture-smoke
```

It creates an episode under `/workspace/data/episodes` containing:

- `rgb/`: ordered PNG camera frames;
- `steps.jsonl`: timestamped action and state records;
- `episode.json`: schema, task, robot, cameras, and outcome metadata.

For JEPA control data, every frame must be synchronized with the action that caused the transition and the robot state. The WebRTC video is intentionally not recorded or used as the dataset because it is compressed, latency-shifted, and lacks synchronized actions.

## 5. Load JEPA

The fastest meaningful path is offline embedding and goal-progress scoring:

```bash
./ops/lambda.sh jepa-embed
# Or select an episode explicitly:
./ops/lambda.sh jepa-embed <episode-id>
```

This uses the official `facebook/vjepa2-vitl-fpc64-256` checkpoint. For online control, keep the model in a separate process and exchange the latest frame window plus subgoal with the simulator. See [`docs/control-loop.md`](docs/control-loop.md).

## Validation

Local checks do not require Isaac Sim:

```bash
./scripts/validate.sh
```

## Primary documentation

- [Lambda Cloud API](https://docs.lambda.ai/public-cloud/cloud-api/)
- [Isaac Sim container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Isaac Sim livestream clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
- [Isaac Sim Replicator workflows](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_sdg_workflows.html)
- [Official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
