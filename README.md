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
recording dataset: per-camera frames + actions + state + manifest
              │
              ▼
frozen V-JEPA 2 encoder → goal-progress/stage model
              │
              ▼
DINOv3 + JEPA-WM + candidate 7D actions → predicted future latents
              │
              ▼
planner → safety gate → Isaac motion controller (next milestone)
```

The bootstrap proves the simulation, capture, stage-recognition, and base
JEPA-WM runtimes. The separate offline workflow additionally proves native
three-action rollouts and persistent lightweight action adaptation. It does not
yet let JEPA command the arm. The held-out action gate still fails, so the next
milestone is to collect diverse Isaac trajectories and adapt the
action-conditioned predictor before candidate-action planning is enabled.

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

The idempotent bootstrap starts the instance, restricts SSH and WebRTC ingress to your current public IP, syncs this repository, installs Docker and the NVIDIA container runtime, downloads the assets, pulls Isaac Sim, provisions JEPA-WM, and verifies both runtimes:

```bash
./ops/aws.sh bootstrap
./ops/aws.sh up
./ops/aws.sh isaac-status
./ops/aws.sh isaac-logs
```

JEPA-WM uses the public DROID checkpoint and Meta's gated DINOv3 ViT-L/16
LVD-1689M encoder weights. After Meta approves access, put the signed checkpoint
URL in `.env` with quotes so its `&` characters are not interpreted by the
shell:

```bash
DINOV3_CHECKPOINT_URL='https://dinov3.llamameta.net/...'
```

The signed URL is forwarded only to the remote installer; it is not synced into
the repository. It is required on the first bootstrap only. Sources, virtual
environment, caches, and checkpoints persist at
`/home/ubuntu/docker/jepa-wm`, so later bootstraps and EC2 stop/start cycles do
not download the models again.

Bootstrap stores persistent content under `QUANTIS_ASSET_HOME` (configured as `/mnt/quantis-assets` for a separate EBS volume), which is mounted read-only inside the container at `/assets`:

- `/assets/datacenter`: the complete NVIDIA data-center asset pack;
- `/assets/datacenter/usd-assets.txt`: an inventory of the pack's USD stages and components;
- `/assets/datasets/PhysicalAI-Robotics-Manipulation-SingleArm`: the optional 15.3 GB LeRobot-format reference dataset;
- the Franka Panda robot itself comes from Isaac Sim's built-in asset catalog at `Robots/FrankaRobotics/FrankaPanda/franka.usd`.

### Cable/cord asset gap

Neither downloaded pack supplies the task-ready power cord and matching receptacle needed for the plug-in demo. `/assets/cable` is reserved for that custom asset. For the first JEPA-controlled demo, use a vendor CAD/mesh pair for the connector and socket, convert them to USD, and author accurate collision geometry and insertion tolerances. Keep the connector rigid and represent the trailing cable as either a short articulated chain or a visually updated curve at first. Full flexible-cable contact is a separate physics milestone and would make policy debugging substantially harder.

Bootstrap provisions and mounts the source assets. The implemented demo composes
the Franka, rack/module proxy, rigid connector, and socket into the reusable
`datacenter_demo.usda` stage. This remains a geometry-driven placeholder task:
the connector has no deformable trailing cable and is not a production asset.

The PhysicalAI reference dataset is opt-in because it is training data rather than a simulator asset. Set `DOWNLOAD_PHYSICALAI_DATASET=1` in local `.env` when it is needed; authenticate Hugging Face on the remote host first to avoid anonymous API limits. For the collection's 136,000+ small files, `HF_DOWNLOAD_MAX_WORKERS` controls download concurrency and defaults to `32`. The AWS wrapper forwards these and the Isaac version/port settings to the remote scripts. The stream is ready when the status command reports:

```text
running healthy
```

For normal sessions after the first bootstrap:

```bash
./ops/aws.sh up
# Work, capture, or embed...
./ops/aws.sh backup-state
./ops/aws.sh down
```

`down` stops the EC2 host and waits until it is fully stopped. Compute billing then stops, while EBS volumes and snapshots continue to incur storage charges. The EBS data and instance ID survive. Unless the instance has an Elastic IP, its public IP changes on the next start; `aws.sh` discovers the new address automatically and refreshes the security-group rules.

### Persistent state and recovery copy

Scenes, recordings, reports, virtual environments, and model checkpoints live
on the instance's 200 GB root EBS volume. They survive container restarts and
normal EC2 stop/start cycles. On the current instance, however, the root volume
`vol-0df988a8afe2217b0` has `DeleteOnTermination=true`.

The asset volume `vol-023dd41d53c058eac` is mounted at
`/mnt/quantis-assets`, has `DeleteOnTermination=false`, and therefore also holds
a recovery copy of mutable project state. Refresh and checksum-verify it before
stopping work:

```bash
./ops/aws.sh backup-state
```

The command copies without deleting older backup files:

| Live state | Recovery copy |
| --- | --- |
| `/home/ubuntu/docker/isaac-sim/data/quantis/scenes` | `/mnt/quantis-assets/quantis-state/isaac/scenes` |
| `/home/ubuntu/docker/isaac-sim/data/quantis/recordings` | `/mnt/quantis-assets/quantis-state/isaac/recordings` |
| `/home/ubuntu/docker/jepa-wm/checkpoints` | `/mnt/quantis-assets/quantis-state/jepa-wm/checkpoints` |

The backup command fails closed unless `/mnt/quantis-assets` is the exact mount
point of a filesystem distinct from both live source filesystems. This prevents
an unmounted asset-volume directory from being mistaken for a recovery copy on
the deletable root disk.

`LAST_BACKUP_UTC` records the most recent verified copy time. The snapshot
verified at `2026-08-23T04:42:24Z` is 8,189,717,453 bytes and includes the
reusable/result stages, all 15 synchronized recording directories,
base/adapted evaluation reports, DINOv3 and JEPA-WM checkpoints, and the Isaac
action adapter. Source code is preserved separately in this Git repository.
Bootstrap does not restore this backup automatically; after replacing or
terminating the instance, copy these three trees back to their corresponding
live paths before starting Isaac or JEPA-WM.

### GPU capacity after a stop

Stopping an On-Demand GPU instance releases its physical GPU capacity even though its EBS volumes and instance configuration persist. A later `./ops/aws.sh up` can therefore fail with `InsufficientInstanceCapacity` when the instance's type is temporarily unavailable in its Availability Zone. This is not an IAM, profile, quota, or data-loss error.

Keep the instance stopped and retry the same `./ops/aws.sh up` command every 15 minutes. The wrapper always verifies the `quantis` profile against AWS account `686410906008`, starts only the configured instance ID, discovers its new public IP, refreshes the managed firewall rules, and starts Isaac Sim. Stop retrying when `./ops/aws.sh isaac-status` reports `running healthy`.

Do not automatically change the instance type, Availability Zone, volumes, or security rules as part of the retry. If capacity remains unavailable, explicitly choose between temporarily changing the stopped instance to the compatible `g5.2xlarge`, migrating a replacement into another Availability Zone, or purchasing an On-Demand Capacity Reservation. A Capacity Reservation avoids this restart gap but incurs charges while the capacity is held.

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
./ops/aws.sh demo-record
```

The ordered sequence is `ready → pre-grasp → grasp → pre-insertion → insert → release`. Preflight solves all six poses before physics advances. The executor interpolates Isaac articulation positions, keeps the placeholder plug kinematic, carries it with the hand after grasp, and pauses on the final pose. It exports the result beside the reusable scene as `datacenter_demo_sequence_result.usda`; `demo-reset` reopens the clean starting stage.

This is deliberately a deterministic coordinate/constraint demo. Plug collision is disabled while attached, and the final seating position is enforced geometrically. It does **not** yet model grasp force, insertion force, deformable cable dynamics, or collision-aware path planning. Those belong in the later force/contact-control milestone.

`demo-capture` renders 1920×1080 RGB verification frames from `/World/ShotCam` and the arm-mounted `/World/Franka_R/panda_hand/WristCamera` into Isaac's persistent data directory at `/isaac-sim/.local/share/ov/data/quantis/captures`.

`demo-record` resets and runs the complete sequence while recording both cameras
at 1920×1080 and 12 FPS. Long captures run as a simulator background job; the AWS
wrapper polls its atomic result file and encodes only after successful completion,
so an idle Python-server socket cannot strand the MP4 step. It then encodes
`presentation.mp4` and `wrist.mp4` on the EC2 host. Each timestamped directory
also keeps the lossless PNG streams, stage labels, `steps.jsonl` robot/plug state,
and `manifest.json` under:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/recordings/demo-<UTC timestamp>
```

The same directory is visible inside Isaac Sim at `/isaac-sim/.local/share/ov/data/quantis/recordings/demo-<UTC timestamp>`. This is on the persistent EBS-backed Isaac data mount, so recordings survive container and instance restarts.

Create the 2560×1440 presentation dashboard with a full-resolution wrist-camera
view and a synchronized metrics panel by naming a separate JEPA reference run:

```bash
./ops/aws.sh demo-dashboard demo-<reference UTC timestamp> wrist wrist
```

The metrics panel displays the recorded task phase, offline JEPA stage
classification, cosine similarity and margin, joint positions, gripper width,
plug position, and attachment state. It explicitly labels the controller as
scripted position control; the demo does not report torque or claim JEPA control.
The final `dashboard.mp4` is written beside the source camera videos and lossless
frame streams.

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

Stage recordings contain at least 64 real synchronized observations for each of
`approaching_cable`, `cable_grasped`, `aligned_with_socket`, and `plug_seated`.
The simulator holds each landmark and renders new frames until the count is met;
it never copies PNGs to fill a window. Embed one whole-run wrist point of view
from the latest recording with:

```bash
./ops/aws.sh jepa-embed latest wrist
# Or select a recording and camera explicitly:
./ops/aws.sh jepa-embed demo-<UTC timestamp> presentation
```

`latest` prefers the newest timestamped demo recording and falls back to the
legacy capture-smoke episodes. The command uses the official
`facebook/vjepa2-vitl-fpc64-256` checkpoint, automatically selects CUDA, Apple
Metal, or CPU, and stores the normalized embedding beside the persistent source
as `<camera>_vjepa2_embedding.npy`. The first AWS run downloads the Python and
model dependencies; subsequent runs reuse the EBS-backed virtual environment and
Hugging Face cache.

The first live AWS proof used recording `demo-20260822T194745Z`: 64 wrist frames became
a normalized 1,024-dimensional `float32` embedding on CUDA at
`/home/ubuntu/docker/isaac-sim/data/quantis/recordings/demo-20260822T194745Z/wrist_vjepa2_embedding.npy`.
### Stage similarity demo

Build or reuse the four cached wrist-camera embeddings for a recording:

```bash
./ops/aws.sh jepa-stage-embed demo-<UTC timestamp> wrist
```

Evaluate a completely separate query run against a reference run:

```bash
./ops/aws.sh jepa-stage-report REFERENCE_RECORDING QUERY_RECORDING wrist
```

The report command creates missing embeddings, reuses unchanged `.npy` files on
later runs, prints the cosine score for every stage pair, and persists
`jepa/wrist/stage_report.json` beside the query recording. On AWS, reference
`demo-20260822T214233Z` and held-out query `demo-20260822T215537Z` classified all
four deterministic stages correctly. The minimum winning margin was `0.0083`,
so this is a plumbing and nominal-separation result—not a robustness claim.

`jepa.stage_gate.StageGate` implements the controller-side safety contract: it
rejects stale observation IDs, pauses on unknown/unexpected/low-confidence
predictions, and requires repeated confirmation before advancing.
`python -m jepa.online_worker` is the separate JSONL prediction process; it keeps
the encoder loaded and assigns a fresh monotonic ID to every 64-frame request.
The worker-to-Isaac transport and live controller call site are intentionally not
connected. A perturbed/misaligned evaluation must pass before enabling that
connection. See [`docs/control-loop.md`](docs/control-loop.md) for the process
boundary and remaining acceptance gates.

## 7. Load action-conditioned JEPA-WM

The same EC2 L4 now hosts the official DROID JEPA-WM and Isaac Sim. Check the
persistent installation or run a real encoder-plus-predictor rollout with:

```bash
./ops/aws.sh jepa-wm-status
./ops/aws.sh jepa-wm-smoke
```

`jepa-wm-smoke` loads DINOv3 ViT-L/16, loads the epoch-315 DROID predictor,
encodes one 256×256 observation, and predicts the next latent state conditioned
on one zero-valued 7D action. The process exits after reporting JSON; the future
online planner service will keep the model resident.

On the `g6.2xlarge` L4 proof, the headless model used 2.01 GiB peak allocated
VRAM, loaded in 10–12 seconds, and completed the warm single-action rollout in
0.23–0.30 seconds. With Isaac streaming concurrently, total GPU use peaked at
about 10.4 GiB of 23.0 GiB and Isaac remained healthy. The machine has enough
capacity for an offline planner prototype, but model validity—not hardware—is
the current blocker.

### Offline action validation

Capture a motion-only trajectory at the DROID checkpoint's native 4 FPS, then
compare recorded actions with a zero-action baseline:

```bash
./ops/aws.sh demo-record-actions
./ops/aws.sh jepa-wm-eval trajectory-<UTC timestamp> wrist 0 20 1
./ops/aws.sh jepa-wm-eval trajectory-<UTC timestamp> presentation 0 20 1
```

The v3 recording schema stores the hand pose in the Franka base frame and a
synchronized DROID-format action for every transition: delta XYZ, relative XYZ
Euler rotation, and gripper-closedness delta. Evaluation uses the released
planner's native temporal contract—one observed frame followed by three actions
and a terminal target frame. It filters out-of-bounds rollouts, compares each
recorded three-action rollout with three zero actions, and scores both with the
official terminal latent-L2 objective. Reports persist under each recording's
`jepa_wm/` directory.

The released checkpoint does not pass this gate on Isaac imagery. On the fresh
held-out recording `trajectory-20260823T041000Z`, its wrist-camera recorded
actions won 10% of 20 rollouts, with mean improvement over zero of `-0.002089`.
The exterior presentation view also failed on the training trajectory, so the
problem is not resolved by swapping cameras.

There is now a bounded calibration path that freezes DINOv3 and the complete
predictor, trains only the action encoder's 7,168 action-dependent weights, and
persists the overlay on the EBS-backed model volume:

```bash
./ops/aws.sh jepa-wm-adapt trajectory-<training timestamp> wrist 100
./ops/aws.sh jepa-wm-eval-adapted trajectory-<held-out timestamp> wrist 0 20 1
```

The adapter improved the same held-out run to a positive `0.000173` mean and a
50% win rate while preserving the base model's zero-action prediction. This is
a real repeatable signal, but it is below the 75% held-out acceptance gate. The
exact run used training recording `trajectory-20260823T034710Z`, 100 optimizer
steps, batch size 2, learning rate `1e-3`, seed 234, and 30 native rollouts. The
adapter and training report remain available at:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth.json
```

The untouched held-out recording and its reports are:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/recordings/trajectory-20260823T041000Z
  jepa_wm/wrist_rollout_eval_000000_020.json
  jepa_wm/wrist_adapted_rollout_eval_000000_020.json
```

The adapter is not connected to the arm. The next model milestone is
action-rich, camera/geometry-diverse Isaac data followed by broader
action-conditioned predictor adaptation. Only after that held-out gate passes
should CEM candidate selection or controller wiring begin.

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
- [Official JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms)
- [Official DINOv3 repository](https://github.com/facebookresearch/dinov3)
