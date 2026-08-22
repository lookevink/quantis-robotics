# Quantis robotics bootstrap

Simulation-first JEPA experiments for a narrow data-center manipulation demo.

The first embodiment is the **Franka Panda** because:

- Isaac Sim ships a supported Franka USD asset.
- NVIDIA's PhysicalAI single-arm dataset uses Franka with RGB, state, and 7D/8D actions.
- Meta's V-JEPA 2-AC robot experiments use Franka/DROID trajectories.

The NVIDIA Physical AI Hugging Face collection is training data, not a catalog of drop-in robot USD assets. This project uses Isaac Sim's built-in Franka asset and can optionally download [`nvidia/PhysicalAI-Robotics-Manipulation-SingleArm`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-SingleArm) as a compatible reference dataset.

The simulator is pinned to the rendered-frame-tested Isaac Sim `6.0.1` container. AWS's current DLAMI driver is compatible with Isaac 6; Isaac 5.0 crashes on its newer CUDA 13/595 driver stack. Do not change the image tag without rerunning the compatibility, runtime, and capture smoke tests.

## Current architecture

```text
Isaac Sim 6.0.1
  ├─ Franka + module-proxy scene
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

## 1. AWS EC2 instance

Prerequisites:

- AWS CLI profile `quantis`, authenticated to account `686410906008`.
- An existing EBS-backed Ubuntu EC2 GPU instance with an RTX-capable GPU and a public IP.
- An SSH keypair authorized by the instance.
- `aws`, `curl`, and `rsync` locally.

Use a `g6.2xlarge` (L4, 24 GB VRAM) or `g5.2xlarge` (A10G, 24 GB VRAM). Both provide the 8 vCPUs and 32 GB RAM that meet Isaac Sim's minimum host requirements. A practical split is a 200 GB root volume for the DLAMI and container images plus a separate 250 GB gp3 data volume mounted at `/mnt/quantis-assets`. Use AWS's **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)**; resolve the current AMI instead of hard-coding an ID:

```bash
aws --profile quantis --region us-east-1 ssm get-parameter \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id \
  --query Parameter.Value --output text
```

Copy `.env.example` to `.env` and set `AWS_INSTANCE_ID`, `AWS_SSH_PRIVATE_KEY`, and the instance's region. The ID is deliberately required so this script never starts or stops a similarly named instance by accident.

Check and start that instance when necessary:

```bash
./ops/aws.sh ensure-running
./ops/aws.sh status
./ops/aws.sh ssh
```

## 2. Bootstrap the remote host

The idempotent bootstrap starts the instance, restricts SSH and WebRTC ingress to your current public IP, syncs this repository, installs Docker and the NVIDIA container runtime, downloads the assets, pulls Isaac Sim, and runs a rendered-frame smoke test:

```bash
./ops/aws.sh bootstrap
./ops/aws.sh up
./ops/aws.sh isaac-status
./ops/aws.sh isaac-logs
```

Bootstrap stores persistent content under `QUANTIS_ASSET_HOME` (configured as `/mnt/quantis-assets` for a separate EBS volume), which is mounted read-only inside the container at `/assets`:

- `/assets/datacenter`: the complete NVIDIA data-center asset pack;
- `/assets/datacenter/usd-assets.txt`: an inventory of the pack's USD stages and components;
- `/assets/datasets/PhysicalAI-Robotics-Manipulation-SingleArm`: the optional 15.3 GB LeRobot-format reference dataset;
- the Franka Panda robot itself comes from Isaac Sim's built-in asset catalog at `Robots/FrankaRobotics/FrankaPanda/franka.usd`.

### Cable/cord asset gap

Neither downloaded pack supplies the task-ready power cord and matching receptacle needed for the plug-in demo. `/assets/cable` is reserved for that custom asset. For the first JEPA-controlled demo, use a vendor CAD/mesh pair for the connector and socket, convert them to USD, and author accurate collision geometry and insertion tolerances. Keep the connector rigid and represent the trailing cable as either a short articulated chain or a visually updated curve at first. Full flexible-cable contact is a separate physics milestone and would make policy debugging substantially harder.

Bootstrap provisions and mounts the assets; it does not yet compose the data-center pack, Franka, connector, and socket into one task stage. That scene and its insertion controller are the next implementation milestone.

The PhysicalAI reference dataset is opt-in because it is training data rather than a simulator asset. Set `DOWNLOAD_PHYSICALAI_DATASET=1` in local `.env` when it is needed; authenticate Hugging Face on the remote host first to avoid anonymous API limits. For the collection's 136,000+ small files, `HF_DOWNLOAD_MAX_WORKERS` controls download concurrency and defaults to `32`. The AWS wrapper forwards these and the Isaac version/port settings to the remote scripts. The stream is ready when the status command reports:

```text
running healthy
```

For normal sessions after the first bootstrap:

```bash
./ops/aws.sh up
# Work, capture, or embed...
./ops/aws.sh down
```

`down` stops the EC2 host and waits until it is fully stopped. Compute billing then stops, while EBS volumes and snapshots continue to incur storage charges. The EBS data and instance ID survive. Unless the instance has an Elastic IP, its public IP changes on the next start; `aws.sh` discovers the new address automatically and refreshes the security-group rules.

## 3. Stream the UI

Isaac Sim uses TCP `49100` for WebRTC signaling and UDP `47998` for media. `firewall-webrtc` replaces only the Quantis-managed rules in the instance's EC2 security group and restricts SSH and both streaming ports to your current public IP. Override the source with `WEBRTC_SOURCE_CIDR` if needed. The stream has no authentication or encryption, so do not open it to `0.0.0.0/0`. The container uses host networking; Docker bridge port publishing is not sufficient for Isaac WebRTC.

Install the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html) on the local workstation and connect to the EC2 public IP printed by `./ops/aws.sh ip`.

Streaming is only remote display/control. It is **not** the training-data capture path.

## 4. Run the deterministic plug-in demo

With `isaacsim.code_editor.python_server` enabled in the live Isaac session, the AWS wrapper can preflight and execute the arm sequence through the server's loopback-only port:

```bash
./ops/aws.sh demo-reset
./ops/aws.sh demo-preflight
./ops/aws.sh demo-run
./ops/aws.sh demo-capture
```

The ordered sequence is `ready → pre-grasp → grasp → pre-insertion → insert → release`. Preflight solves all six poses before physics advances. The executor interpolates Isaac articulation positions, keeps the placeholder plug kinematic, carries it with the hand after grasp, and pauses on the final pose. It exports the result beside the reusable scene as `datacenter_demo_sequence_result.usda`; `demo-reset` reopens the clean starting stage.

This is deliberately a deterministic coordinate/constraint demo. Plug collision is disabled while attached, and the final seating position is enforced geometrically. It does **not** yet model grasp force, insertion force, deformable cable dynamics, or collision-aware path planning. Those belong in the later force/contact-control milestone.

`demo-capture` renders 640×480 RGB verification frames from `/World/ShotCam` and the arm-mounted `/World/Franka_R/panda_hand/WristCamera` into Isaac's persistent data directory at `/isaac-sim/.local/share/ov/data/quantis/captures`.

## 5. Capture training data

Run the standalone capture smoke test inside the container:

```bash
./ops/aws.sh capture-smoke
```

It creates a pipeline-test episode under `/workspace/data/episodes` containing:

- `rgb/`: ordered PNG camera frames;
- `steps.jsonl`: timestamped action and state records;
- `episode.json`: schema, task, robot, cameras, and outcome metadata.

The smoke test moves a module proxy while Franka is visible; its synthetic action/state labels are **not robot training data**. For JEPA control data, replace that motion with commands applied to Franka and record its joints/end effector at every step. The WebRTC video is intentionally not recorded or used as the dataset because it is compressed, latency-shifted, and lacks synchronized actions.

## 6. Load JEPA

The fastest meaningful path is offline embedding and goal-progress scoring:

```bash
./ops/aws.sh jepa-embed
# Or select an episode explicitly:
./ops/aws.sh jepa-embed <episode-id>
```

This uses the official `facebook/vjepa2-vitl-fpc64-256` checkpoint and proves GPU inference on simulated observations. It does **not** yet drive Franka. For online control, keep the model in a separate process and exchange the latest frame window plus subgoal with the simulator. See [`docs/control-loop.md`](docs/control-loop.md) for the planned interface and safety boundary.

## Validation

Local checks do not require Isaac Sim:

```bash
./scripts/validate.sh
```

## Primary documentation

- [AWS EC2 stop/start behavior and billing](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/how-ec2-instance-stop-start-works.html)
- [AWS Deep Learning Base GPU AMI](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-24-04.html)
- [Isaac Sim AWS deployment](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_advanced_cloud_setup_aws.html)
- [Isaac Sim container installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html)
- [Isaac Sim livestream clients](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
- [Isaac Sim Replicator workflows](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_sdg_workflows.html)
- [Official V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
