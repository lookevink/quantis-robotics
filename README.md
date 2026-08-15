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
frozen V-JEPA 2 encoder → goal-progress/stage model (next milestone)
              │
              ▼
high-level subgoal → Isaac motion controller
```

The current bootstrap proves simulation, capture, and JEPA inference. Closing the loop from JEPA output back into Isaac control is the next milestone. Action-conditioned world-model planning comes after that simpler stage/subgoal loop works.

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

Isaac Sim uses TCP `49100` for WebRTC signaling and UDP `47998` for media. `firewall-webrtc` adds both to Lambda's **account-wide global ruleset**, restricted to your current public IP. Lambda cannot attach a per-instance ruleset after an instance has launched. Override the source with `WEBRTC_SOURCE_CIDR` if needed. The container uses host networking; Docker bridge port publishing is not sufficient for Isaac WebRTC.

Install the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/manual_livestream_clients.html) on the local workstation and connect to the Lambda instance's public IP printed by `./ops/lambda.sh ip`.

Streaming is only remote display/control. It is **not** the training-data capture path.

## 4. Capture training data

Run the standalone capture smoke test inside the container:

```bash
./ops/lambda.sh capture-smoke
```

It creates a pipeline-test episode under `/workspace/data/episodes` containing:

- `rgb/`: ordered PNG camera frames;
- `steps.jsonl`: timestamped action and state records;
- `episode.json`: schema, task, robot, cameras, and outcome metadata.

The smoke test moves a module proxy while Franka is visible; its synthetic action/state labels are **not robot training data**. For JEPA control data, replace that motion with commands applied to Franka and record its joints/end effector at every step. The WebRTC video is intentionally not recorded or used as the dataset because it is compressed, latency-shifted, and lacks synchronized actions.

## 5. Load JEPA

The fastest meaningful path is offline embedding and goal-progress scoring:

```bash
./ops/lambda.sh jepa-embed
# Or select an episode explicitly:
./ops/lambda.sh jepa-embed <episode-id>
```

This uses the official `facebook/vjepa2-vitl-fpc64-256` checkpoint and proves GPU inference on simulated observations. It does **not** yet drive Franka. For online control, keep the model in a separate process and exchange the latest frame window plus subgoal with the simulator. See [`docs/control-loop.md`](docs/control-loop.md) for the planned interface and safety boundary.

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
