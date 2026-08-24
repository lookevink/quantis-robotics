# Quantis robotics lab

Simulation-first JEPA research for autonomous data-center manipulation.

## Mission

Quantis is building robotics for lights-out data-center operations: facilities
that can remain dark and operate without routine human presence. The long-term
intent is to help data-center operators:

- reduce technicians' exposure to energized and high-voltage equipment;
- respond to physical faults faster, at any hour;
- reduce lighting and human-comfort cooling overhead where the facility's
  operating envelope permits; and
- make inspection and maintenance more consistent, repeatable, and auditable.

This repository is the lab for that work. Its current cable reach-and-grasp
experiments are deliberately narrow research milestones, not evidence that
autonomous, human-free data-center operation is production-ready.

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
DINOv3 + current pose + previous action → bounded 3×7D proposal
              │
              ▼
    resident Unix-socket worker → safety gate → first-action receding-horizon control
```

The bootstrap proves the simulation, capture, stage-recognition, and base
JEPA-WM runtimes. The separate offline workflow additionally proves native
three-action rollouts and persistent lightweight action adaptation. A
whole-seed domain experiment clears both the action-conditioning and
inverse-action proposal gates. A simulator-only bridge now repeatedly captures,
infers, executes only the first fresh bounded proposal, measures the outcome,
and replans after workspace, joint, velocity, collision, force, and tracking
interlocks. It does not yet use JEPA-WM candidate-energy search or control the
cable task.

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

Enable the lean CloudWatch agent configuration for one-minute host RAM, GPU
utilization, and GPU memory metrics:

```bash
./ops/aws.sh cloudwatch-enable
./ops/aws.sh cloudwatch-status
```

This idempotently attaches `CloudWatchAgentServerPolicy` to the instance role
and starts the existing agent with four custom `CWAgent` metrics:
`mem_used_percent`, `nvidia_smi_utilization_gpu`,
`nvidia_smi_memory_used`, and `nvidia_smi_memory_total`. View them in the AWS
console under **CloudWatch → Metrics → All metrics → CWAgent**. EC2 detailed
monitoring remains disabled because it does not provide host RAM or GPU data.

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

Neither downloaded pack supplies the task-ready power cord and matching receptacle needed for the plug-in lab task. `/assets/cable` is reserved for that custom asset. For the first JEPA-controlled lab experiment, use a vendor CAD/mesh pair for the connector and socket, convert them to USD, and author accurate collision geometry and insertion tolerances. Keep the connector rigid and represent the trailing cable as either a short articulated chain or a visually updated curve at first. Full flexible-cable contact is a separate physics milestone and would make policy debugging substantially harder.

Bootstrap provisions and mounts the source assets. The implemented lab scene composes
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
| `/home/ubuntu/docker/isaac-sim/data/quantis/control_sessions` | `/mnt/quantis-assets/quantis-state/isaac/control_sessions` |
| `/home/ubuntu/docker/isaac-sim/data/quantis/control_rollouts` | `/mnt/quantis-assets/quantis-state/isaac/control_rollouts` |
| `/home/ubuntu/docker/isaac-sim/data/quantis/control_baselines` | `/mnt/quantis-assets/quantis-state/isaac/control_baselines` |
| `/home/ubuntu/docker/isaac-sim/data/quantis/control_candidates` | `/mnt/quantis-assets/quantis-state/isaac/control_candidates` |
| `/home/ubuntu/docker/jepa-wm/checkpoints` | `/mnt/quantis-assets/quantis-state/jepa-wm/checkpoints` |

The backup command fails closed unless `/mnt/quantis-assets` is the exact mount
point of a filesystem distinct from both live source filesystems. This prevents
an unmounted asset-volume directory from being mistaken for a recovery copy on
the deletable root disk.

`LAST_BACKUP_UTC` records the most recent verified copy time. The recovery copy
includes reusable/result stages, synchronized recordings, control-session and
rollout evidence, realized-baseline reports, evaluation reports, DINOv3 and
JEPA-WM checkpoints, and the Isaac action adapter. Source code is preserved
separately in this Git repository. Bootstrap does not restore this backup
automatically; after replacing or terminating the instance, copy these seven
trees back to their corresponding live paths before starting Isaac or JEPA-WM.

### GPU capacity after a stop

Stopping an On-Demand GPU instance releases its physical GPU capacity even though its EBS volumes and instance configuration persist. A later `./ops/aws.sh up` can therefore fail with `InsufficientInstanceCapacity` when the instance's type is temporarily unavailable in its Availability Zone. This is not an IAM, profile, quota, or data-loss error.

Keep the instance stopped and retry the same `./ops/aws.sh up` command every 15 minutes. The wrapper always verifies the `quantis` profile against AWS account `686410906008`, starts only the configured instance ID, discovers its new public IP, refreshes the managed firewall rules, and starts Isaac Sim. Stop retrying when `./ops/aws.sh isaac-status` reports `running healthy`.

Do not automatically change the instance type, Availability Zone, volumes, or security rules as part of the retry. If capacity remains unavailable, explicitly choose between temporarily changing the stopped instance to the compatible `g5.2xlarge`, migrating a replacement into another Availability Zone, or purchasing an On-Demand Capacity Reservation. A Capacity Reservation avoids this restart gap but incurs charges while the capacity is held.

## 3. Stream the UI

Isaac Sim uses TCP `49100` for WebRTC signaling and UDP `47998` for media. `firewall-webrtc` replaces only the Quantis-managed rules in the instance's EC2 security group and restricts SSH and both streaming ports to your current public IP. Override the source with `WEBRTC_SOURCE_CIDR` if needed. The stream has no authentication or encryption, so do not open it to `0.0.0.0/0`. The container uses host networking; Docker bridge port publishing is not sufficient for Isaac WebRTC.

Install the [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html) on the local workstation and connect to the EC2 public IP printed by `./ops/aws.sh ip`.

Streaming is only remote display/control. It is **not** the training-data capture path.

## 4. Run the deterministic plug-in lab sequence

With `isaacsim.code_editor.python_server` enabled in the live Isaac session, the AWS wrapper can preflight and execute the arm sequence through the server's loopback-only port:

```bash
./ops/aws.sh demo-reset
./ops/aws.sh demo-preflight
./ops/aws.sh demo-run
./ops/aws.sh demo-capture
./ops/aws.sh demo-record
```

The existing `demo-*` command and artifact identifiers remain unchanged for
compatibility; “lab” is the project and capability language.

The ordered sequence is `ready → pre-grasp → grasp → pre-insertion → insert → release`. Preflight solves all six poses before physics advances. The executor interpolates Isaac articulation positions, keeps the placeholder plug kinematic, carries it with the hand after grasp, and pauses on the final pose. It exports the result beside the reusable scene as `datacenter_demo_sequence_result.usda`; `demo-reset` reopens the clean starting stage.

This is deliberately a deterministic coordinate/constraint lab sequence. Plug collision is disabled while attached, and the final seating position is enforced geometrically. It does **not** yet model grasp force, insertion force, deformable cable dynamics, or collision-aware path planning. Those belong in the later force/contact-control milestone.

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
scripted position control; this lab sequence does not report torque or claim JEPA control.
The final `dashboard.mp4` is written beside the source camera videos and lossless
frame streams.

To film a high-resolution replay of a completed, strictly validated JEPA-WM
reset-trial candidate, run:

```bash
./ops/aws.sh jepa-wm-candidate-film STRICT_CANDIDATE_REPORT [recording-name]
```

This first reconstructs the persisted candidate report from its raw sessions and
requires its strict comparison gate to pass. It then recreates the candidate's
held-out reset, verifies the initial and final arm/gripper state, monitors contact
on every filmed frame, and records the already-realized transition from both the
1080p wrist and presentation cameras. The dashboard reports the bounded CEM
search, energy improvement, realized control action, and replay-derived tracking
and contact evidence. It is a visualization artifact and does not create new
control evidence or production authority.

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

`latest` prefers the newest timestamped `demo-*` lab recording and falls back to the
legacy capture-smoke episodes. The command uses the official
`facebook/vjepa2-vitl-fpc64-256` checkpoint, automatically selects CUDA, Apple
Metal, or CPU, and stores the normalized embedding beside the persistent source
as `<camera>_vjepa2_embedding.npy`. The first AWS run downloads the Python and
model dependencies; subsequent runs reuse the EBS-backed virtual environment and
Hugging Face cache.

The first live AWS proof used recording `demo-20260822T194745Z`: 64 wrist frames became
a normalized 1,024-dimensional `float32` embedding on CUDA at
`/home/ubuntu/docker/isaac-sim/data/quantis/recordings/demo-20260822T194745Z/wrist_vjepa2_embedding.npy`.
### Stage similarity evaluation

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
This stage-classification worker remains separate from the action-conditioned
control bridge. See [`docs/control-loop.md`](docs/control-loop.md) for both
process boundaries and remaining acceptance gates.

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
about 10.4 GiB of 23.0 GiB and Isaac remained healthy. The resident action
worker and Isaac therefore share this instance without another GPU.

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

The first adapter improved the same held-out run to a positive `0.000173` mean
and a 50% win rate while preserving the base model's zero-action prediction.
That diagnostic remained below the 75% acceptance gate. The original run used
training recording `trajectory-20260823T034710Z`, 100 optimizer steps, batch
size 2, learning rate `1e-3`, seed 234, and 30 native rollouts.

### Reproduce the domain-data milestone

The milestone command records deterministic, bounded wrist-camera exploration
data, splits complete seeds between training and evaluation, fits one adapter
over all training recordings, evaluates each unseen seed, aggregates the real
per-rollout outcomes, and refreshes the recovery-volume backup:

```bash
./ops/aws.sh jepa-wm-milestone 4 2 500 1400
```

Each seed produces 69 synchronized 512x512 wrist frames centered on a seeded
offset from the IK-verified ready pose. Its 17 segments excite every Franka arm
joint twice and include stationary, failed-grasp, and recovery outcomes. The
capture varies the wrist-camera offset, plug/socket offset, receptacle scale,
and light exposure. Each adjacent sample is verified against actual simulation
time at exactly 0.25 seconds. Seeds `1400` through `1403` are training-only;
seeds `11400` and `11401` are held out as complete runs. This prevents adjacent
frames from one trajectory leaking across the split.

Experiment `domain-20260823T113209Z-1400` trained the 7,168-parameter action
adapter on 264 native three-action rollouts for 500 steps. On 80 unseen
rollouts, recorded actions beat zero actions 78 times (`97.5%`) with positive
mean energy improvement (`+0.0010540774`). Both held-out seeds passed
individually: seed `11400` scored `97.5%` and `+0.0010193158`; seed `11401`
scored `97.5%` and `+0.0010888389`. This clears the repository's offline gate
of positive mean improvement and at least 75% wins.

The current adapter and training report are available at:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth.json
```

The aggregate experiment report and held-out evidence are available at:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/experiments/domain-20260823T113209Z-1400.json
/home/ubuntu/docker/isaac-sim/data/quantis/recordings/domain-20260823T113209Z-1400-held-00/jepa_wm/wrist_adapted_rollout_eval_000000_040.json
/home/ubuntu/docker/isaac-sim/data/quantis/recordings/domain-20260823T113209Z-1400-held-01/jepa_wm/wrist_adapted_rollout_eval_000000_040.json
```

### Reach-and-grasp data checkpoint

The next dataset adds a scripted, labeled task tail after the same seeded
exploration prefix: approach, close at the connector, an explicit rigid-body
attachment transition, a 10 cm retained retreat, and a hold. Record and
validate one trajectory with:

```bash
./ops/aws.sh demo-record-grasp grasp-example 11401 held_out
./ops/aws.sh jepa-wm-grasp-validate grasp-example held_out
```

The validator rejects wrong split/task provenance, incomplete or off-cadence
telemetry, missing/lost attachment, and less than 2 cm of retained connector
motion. The first AWS artifact, `grasp-20260824-held-11401-v1`, contains 102
true-4-FPS frames, acquired the connector at frame 89, retained it for 13
observations, and moved it `99.9997 mm` while attached.

This artifact is a scripted training/evaluation example, not evidence
that JEPA completed the task. Build the required 12-training-seed/two-held-out
proposal dataset and offline gate with:

```bash
./ops/aws.sh jepa-wm-grasp-milestone 12 2 3000 2400
```

Live control now persists connector position and attachment state after every
action. It can only acquire the rigid connector when the hand is within 25 mm
of the IK-defined grasp pose and the gripper width is at most 30 mm. Rollout
task success additionally requires an unattached-to-attached transition,
continuous retention over at least two observations, at least 20 mm of retained
motion, passed tracking, no collision, and no force above 2 N.

For task execution, context index `86` reconstructs the exact held-out
joint/gripper prefix three actions before the scripted attachment frame. Each
follow-up advances the reference target by one frame, allowing the receding
horizon to continue from acquisition into the retained retreat instead of
fixating on the first grasp image. The rollout report admits that moving target
only for a validated `reach_and_grasp` recording and binds the initial connector
pose/attachment state into reset-equivalence checks.

The first complete proposal experiment is preserved as
`grasp-20260824T041009Z-2400`: 12 training seeds and two disjoint held-out
seeds, each with 102 true-4-FPS frames, frame-89 acquisition, 13 attached
observations, and approximately 100 mm of retained connector motion. The
generic 1,188-rollout proposal failed the task gate with aggregate mean cosine
`0.6238` and `75.9%` active-direction passes. Restricting the fit to all 30
task-window rollouts per seed improved those figures to `0.6861` and `79.6%`,
but still failed the `0.9`/`98%` thresholds. A 16-unit regularized head reached
`0.7112`/`85.2%` and also failed. These are retained negative results, not live
task or validated lab evidence; no failed checkpoint is allowed to command the arm.

Task-specific training and evaluation include the final stationary attached
hold windows rather than silently dropping zero-action labels:

```bash
./ops/aws.sh jepa-wm-grasp-proposal-train \
  TRAIN_RECORDING[,TRAIN_RECORDING...] 3000 PROPOSAL
./ops/aws.sh jepa-wm-grasp-proposal-eval HELD_OUT_RECORDING PROPOSAL
./ops/aws.sh jepa-wm-grasp-proposal-summarize \
  HELD_OUT_RECORDING[,HELD_OUT_RECORDING...] PROPOSAL
```

The next iteration kept held-out seeds 12400/12401 fixed and expanded training
variation from 12 to 20 validated seeds (600 exact task-window rollouts). Pure
capacity changes still failed: h16/h32/h128 heads reached aggregate cosine
`0.6786`/`0.6726`/`0.7421` and active-direction pass rates
`81.5%`/`75.9%`/`88.9%`. The proposal contract now additionally conditions on
the provenance-bound delta from the current DROID pose to the three-frame goal
pose. This reduced aggregate MSE to `0.0000226`; the h128 head reached cosine
`0.7717`, a `90%` first-action gate rate, and `92.6%` active-direction passes on
both held-out seeds. It remains a retained negative result: phase reversals at
contexts 76 and 83 and final hold precision still miss the strict gate, so no
live rollout or video is authorized. The next model iteration adds explicit
task-progress conditioning while preserving the same held-out seeds and
fail-closed thresholds.

That task-progress experiment is also retained as a negative. A 661,013-
parameter h128 head trained on the same 600 rollouts reached held-out active
cosines `0.844`/`0.871`, gate rates `90%`/`90%`, and active-direction rates
`88.9%`/`92.6%`; it did not clear either whole-seed gate. The directional
metric now averages cosine only across active labels because cosine is
undefined for a zero vector, while stationary labels still face the unchanged
translation/rotation/gripper hold gate.

The promoted grasp proposal separates gripper timing from visual pose motion.
Its frozen JEPA features and current pose/history/goal/task-progress inputs
still predict the six Cartesian action axes, while a conditioning-only gripper
head prevents seed-specific visual features from closing one step early. The
5.1 MB h256 checkpoint
`grasp-20260824T041009Z-2400_task20_gripper_head_h256_s3000` cleared the strict
offline gate on both untouched held-out seeds: 60/60 first-action gates,
60/60 active-direction gates, aggregate active cosine `0.9769`, and mean
sequence MSE `0.00001145`. Its readiness artifact is:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/experiments/grasp-20260824T041009Z-2400_task20_gripper_head_h256_s3000_grasp_readiness.json
```

The first four-step live rollout is preserved as
`rollout-20260824T101920Z-12400`. All four JEPA actions passed freshness, IK,
joint, tracking, force, contact, and collision gates; all four shadow searches
and counterfactual safety projections also passed. It reduced translation error
by `5.95 mm`, rotation error by `0.000868 rad`, and gripper error by `0.1041`,
but did not attach the connector. The safety selector scaled gripper commands
to one-quarter or one-eighth alongside pose corrections, leaving final
closedness at `0.5541`; therefore `reach_and_grasp.passed` is false with
`no_attachment_transition`. This is meaningful closed-loop progress, not task
completion or authority to claim a successful lab result. The next checkpoint must decouple bounded
gripper scaling from pose/IK scaling and repeat the attachment gate.

That projection checkpoint now preserves full, bounded gripper intent while
independently reducing translation and rotation for IK/joint-velocity safety;
all existing action, freshness, tracking, force, contact, and collision gates
remain unchanged. Historical coupled-scale shadow evidence remains readable.
The strict rerun `rollout-20260824T105547Z-12400` applied all four JEPA actions
and acquired the connector on its first action. Attachment persisted for all
four observations at 0 N without collision, while the final retreat retained
the connector over `5.615 mm`. Its task gate is still false only for
`insufficient_lift` because the defined completion threshold is 20 mm. This is
the first validated JEPA-controlled cable grasp, but it is not yet the complete
reach-and-grasp task and does not authorize filming. The next run must extend
the same held-out receding-horizon sequence through at least 20 mm of retained
motion, then reproduce it on held-out seed 12401.

The eight-step follow-ups complete that gate on both fixed held-out seeds.
Fingerprint-bound rollouts `rollout-20260824T124729Z-12400` and
`rollout-20260824T132311Z-12401` each applied `8/8` JEPA actions, acquired the
connector on action one, retained it for all eight observations, and moved it
`55.258 mm` and `58.508 mm` respectively at 0 N without collision. Independent
reset-identical zero trials never attached, while scripted trials attached and
retained `70.549 mm`/`70.557 mm`. The task-specific readiness gate reloads all
six raw rollouts, validates scripted responses and reset provenance, and binds
both direct trials to proposal fingerprint
`6aa4b94b610bfd8fff07e9356e932574a11342533d55c69b06c1c2ab20e9fd2d`.
It passes `2/2` whole seeds and authorizes a truthful reach-and-grasp film:

```bash
./ops/aws.sh jepa-wm-grasp-control-summarize \
  grasp-control-readiness-20260824-v2 \
  grasp-baseline-20260824-seed12400-v2,grasp-baseline-20260824-seed12401-v2
```

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_readiness/grasp-control-readiness-20260824-v2/readiness.json
```

The preserved v1 readiness report is superseded because those earlier live
responses predated execution-time proposal fingerprint binding. A later
seed-12401 attempt, `rollout-20260824T130455Z-12401`, is also preserved as
negative evidence: seven actions applied before the final action failed closed
at 2.89 seconds of observation age. No freshness threshold was relaxed.

The generic endpoint-pose baseline gate remains false because the direct
rollouts stop `12–15 mm` short of the scripted endpoint (and seed 12400 does
not beat zero's small rotation drift). That stricter result remains visible in
the readiness artifact; task filming readiness is not production authority and
does not establish cable insertion.

Record a high-resolution visualization of one readiness-validated rollout with:

```bash
./ops/aws.sh jepa-wm-grasp-film \
  grasp-control-readiness-20260824-v2 12401 \
  grasp-demo-20260824-seed12401-v1
```

The recorder reconstructs the saved readiness gate and all six raw baseline
rollouts, then replays the eight realized direct endpoints with synchronized
1080p wrist and presentation cameras. The side panel separates original task
evidence from replay-measured tracking/contact/collision and labels the result
as a visualization with no production authority. Outputs are under
`/home/ubuntu/docker/isaac-sim/data/quantis/recordings/<recording>/`, including
`wrist.mp4`, `presentation.mp4`, and the 2560×1440 `dashboard.mp4`.

The seed-12401 visualization completed as
`grasp-demo-20260824-seed12401-v1`: 78 frames at 12 FPS, with the 1080p wrist
view as the primary image. It is bound to proposal SHA-256
`6aa4b94b610bfd8fff07e9356e932574a11342533d55c69b06c1c2ab20e9fd2d` and
source rollout `rollout-20260824T132311Z-12401`. The source task acquired the
connector at action 1 and retained it across eight observations for
`58.508 mm`; replay verification measured at most `0.633 mrad` joint error,
`0.0011 mm` gripper error, zero contact force, and no collision. This is a
truthful reach-and-grasp visualization, not evidence of cable insertion or
production authority.

### Reach-and-insert geometry checkpoint

Insertion now uses a gripper target `40 mm` behind the connector tip. The hand
endpoint is offset by the same amount at the socket, so the connector tip—not
the palm or finger-link origin—is the quantity that seats. The corrected
six-phase baseline is IK-reachable and completed on the live scene with at most
`0.105 mm` hand settle error.

The first independently validated held-out artifact is
`insert-20260824-held-12402-v5`. It contains 124 true-4-FPS wrist observations
covering the action-rich exploration prefix, rearward grasp, alignment,
insertion, and a four-observation seated hold. Raw schema-v5 telemetry
reconstructs `40.005 mm` of gripper-frame clearance, `0.0022 mm` insertion-depth
error, `0.1075 mm` lateral error, and zero orientation error. Validate it with:

```bash
./ops/aws.sh jepa-wm-insertion-validate \
  insert-20260824-held-12402-v5 held_out
```

This is deliberately a `kinematic_scripted_baseline`: plug collision remains
disabled while attached and seating is not force/compliance evidence. The v1
artifact was rejected because palm-frame telemetry could not prove clasp
clearance, v2 failed safely on a recorder type error, and v3 was rejected
because USD finger-link origins are not the Lula gripper frame. All remain
preserved as negative evidence. It is superseded by the contact-aware baseline
below and remains useful only as geometry/provenance evidence.

That contact gate now has a nominal pass. Held-out artifact
`contact-insert-20260824-held-12405-v9` contains 112 true-4-FPS task
observations. The connector is a dynamic rigid body held by a physics fixed
joint; its body/contact colliders and tip-local contact sensor are active. The
RJ45 latch remains visual but non-colliding to represent its real compliant
motion, and the nominal socket uses an explicitly recorded `1.05×` clearance
allowance. Insertion is divided into 64 increments, and the live interlock
polls every intermediate physics/render update and aborts above 2 N rather
than waiting for the next recorded 4-FPS observation.

Independent raw validation reconstructs `40.005 mm` gripper-frame clearance,
`0.0036 mm` depth error, `0.0797 mm` lateral error, `0.00159 rad` orientation
error, four retained seated observations, `6.287 mrad` maximum arm tracking
error, `0.0048 mm` gripper error, no collision, and `0 N` maximum contact. The
same dynamic sensor rejected v3 after measuring a rise from `5.6 N` to
`142.4 N`; v4/v5 then stopped at their first `105.8 N` event under the new
live abort. Those failures prove that the v9 zero-force result is measured
clearance rather than an inactive sensor. Validate the pass with:

```bash
./ops/aws.sh jepa-wm-contact-insertion-validate \
  contact-insert-20260824-held-12405-v9 held_out
```

The v6 pass has the same measured geometry but is superseded because its live
interlock sampled only at recording cadence and its manifest did not bind the
fixed-joint, latch-exclusion, scale, phase, and frame-count contract. V7 added
those bindings but retained only the last sub-limit force sample in each 4-FPS
window. V8 added interval maxima but did not reject a non-finite raw sensor
reading before aggregation. V9 binds the runtime facts, accumulates each
interval's maximum force, rejects invalid sensor values live, and fail-closes
if any raw phase, stage, attachment, or telemetry field does not match.

This authorizes insertion-dataset collection, not JEPA control or filming.

Start or resume that exact corpus with a stable experiment ID:

```bash
./ops/jepa_wm_insertion_corpus.sh \
  12 2 2600 contact-insertion-v9-2600
```

The workflow reuses only recordings that already pass the strict v9 validator,
fails closed on an invalid existing artifact, keeps TRAIN seeds `2600–2611`
disjoint from HELD_OUT seeds `12600–12601`, and recovery-backs up on every exit.
Rerun the same command after an interruption to continue without recapturing
validated recordings.

Once the corpus is complete, run the insertion proposal milestone against the
same stable experiment identity:

```bash
./ops/jepa_wm_insertion_milestone.sh \
  3000 2600 contact-insertion-v9-2600
```

The insertion proposal deliberately starts at context `21`, the validated
fixed-joint attachment observation. It covers all 88 native three-action
rollouts through context `108` and the seated hold. The promoted grasp proposal
remains responsible for acquisition; this checkpoint trains retreat,
alignment, insertion, and stop/hold behavior. Training and held-out evaluation
include stationary actions, bind pose/action-history/goal-delta/task-progress
conditioning to the proposal SHA-256, and require both held-out recordings to
reconstruct the exact contact-aware v9 task. A failed two-seed readiness gate
exits nonzero but is still recovery-backed up. Passing this offline gate does
not by itself authorize live insertion or filming.

The exact corpus and proposal gate completed on August 24, 2026. TRAIN seeds
`2600–2611` were kept disjoint from HELD_OUT seeds `12600–12601`. Proposal
fingerprint
`ea7ce27cce72b4ca09f69c65ea16d5475e9d56648830c9131497d81e125ea255`
passed all 176 held-out rollouts: active-direction and first-action gate rates
were both `100%`, mean active first-action cosine was `0.998534`, and mean
sequence MSE was `4.41223e-8`. The exact roster and training-selection
fingerprint are persisted in the readiness artifact. This remains imitation
evidence; it does not show that JEPA-WM ranks the insertion actions correctly.

The first checkpoint adapted JEPA-WM on the same contexts `21–108`, evaluated
recorded insertion actions against zero action on both whole held-out seeds,
and requires every seed plus the aggregate energy gate to pass:

```bash
./ops/jepa_wm_insertion_wm_milestone.sh \
  500 2600 contact-insertion-v9-2600
```

Adapter fingerprint
`4341ea852c5db41b87522b1cb965f17571d8c06f0b81e2ca2cb9e029d457be48`
completed 500 batch-one updates, but failed the strict gate on August 24, 2026.
Seed `12600` produced `-2.31725e-5` mean improvement and `12.5%` recorded-action
wins; seed `12601` produced `-4.12553e-5` and `47.7273%`. Across all 176
rollouts the result was `-3.22139e-5` and `30.1136%`, versus the required
positive mean and `75%` wins. A TRAIN-seed diagnostic also failed
(`-7.37809e-5`/`4.54545%`), identifying underfitting rather than only a
held-out generalization failure. The base model on seed `12600` was much worse
at `-6.45639e-4`/`2.27273%`, so the adapter learned useful domain structure but
did not learn a valid insertion energy. The 500 with-replacement updates also
covered fewer samples than the 1,056-rollout corpus contains. The next adapter
uses a seeded shuffled epoch schedule so complete corpus coverage is explicit.

The adapter checkpoint, sidecar, held-out reports, and failed readiness
artifact bind the exact roster, context selection, and adapter SHA-256 and are
recovery-backed up. Neither the proposal pass nor an energy pass alone
authorizes live insertion or filming.

The next exact run uses one seeded shuffled pass over all 1,056 rollouts:

```bash
./ops/jepa_wm_insertion_wm_milestone.sh \
  1056 2600 contact-insertion-v9-2600
```

That exact run passed on August 24, 2026. Adapter fingerprint
`47969a0a7869fbf9ba507599cd30ba20937a6b826d26c812f6d8dc6f1f6ddaf9`
visited all 1,056 training rollouts once in a seeded shuffled epoch. HELD_OUT
seed `12600` produced `+6.13514e-5` mean improvement and `82.9545%`
recorded-action wins; seed `12601` produced `+5.94737e-5` and `85.2273%`.
Across all 176 rollouts the strict offline gate passed at `+6.04125e-5` and
`84.0909%`. Peak PyTorch allocation was `7.845 GiB` and the adapter trained in
`1719.489` seconds while sharing the L4 with Isaac Sim. The model, sidecar,
two raw evaluation reports, readiness artifact, and exact roster are
checksum-verified under `/mnt/quantis-assets/quantis-state`.

This proves that the adapted JEPA-WM ranks the demonstrated insertion actions
above zero action on two whole held-out seeds. It does **not** prove that a
JEPA-selected action inserts the connector. The next checkpoint is offline,
proposal-centered bounded candidate search on both held-out recordings. The
passed insertion proposal supplies the initial three-action sequence; JEPA-WM
may refine it only inside the planner bounds and must retain task direction,
stationary holds, and positive energy improvement before any simulator trial.

The first grasp-domain JEPA-WM action adapter used all 1,980 bounded rollouts
from the 20 training recordings. In a fixed eight-context CEM diagnostic it
lowered latent energy below both zero and recorded actions on every context,
but produced only `0.434` mean active cosine and `37.5%` first-action passes:
evidence that the planner was exploiting the learned energy rather than
recovering the demonstrated action. Adding a uniformly sampled mismatched
rollout action improved those figures to `0.489`/`50%`, still far below
the live gate. Both adapters and benchmark reports are preserved as offline
negative evidence. The next adapter iteration needs stronger candidate-aware
ranking and proposal-centered regularization; no current grasp artifact may
command or film the arm.

The next train-only adapter adds online bounded candidate mining: for every
positive rollout it samples four local candidates inside the simulator planner
limits, selects the candidate the current JEPA energy ranks most deceptively,
and learns a margin against it. On the same fixed held-out eight-context CEM
diagnostic, this moved active cosine to `0.497` and first-action passes to
`62.5%` (`37.5%` baseline, `50%` mismatched-negative). The improvement is real
but remains an offline negative: the proposal alone scores `0.703`/`75%` on
those contexts, showing that weak proposal-centered regularization still lets
CEM trade away action agreement for small latent-energy gains. No live or
filming authority is granted.

Proposal-prior calibration is split-safe and explicit. Planner calibration
accepts only a TRAIN recording that belongs to both the adapter and proposal;
normal evaluation accepts only HELD_OUT recordings absent from both training
sets. On TRAIN seed 2400, a `1e-3` prior reached `0.842`/`87.5%`, while `1e-2`
restored the proposal's `0.986`/`100%`, so `1e-2` was frozen before one held-out
run. On held-out seed 12400 it preserved the proposal exactly at
`0.703`/`75%`, improving over candidate-aware CEM's `0.497`/`62.5%` but not over
the proposal itself. This fixes planner degradation; it does not make the
current proposal lab-ready or grant live/filming authority.

### Inverse-action proposal milestone

The repository now has a deterministic bounded CEM implementation, empirical
action priors, and a frozen-JEPA inverse-action proposal head. The first visual
heads overfit or lost spatial direction. Adding spatial latent moments and the
current DROID pose reached roughly 81% active-direction accuracy. Adding the
previous DROID action—the arm's direction of travel—resolved the sweep
turnarounds. The promoted head was trained over 792 three-action rollouts from
12 complete training seeds and contains 659,989 trainable parameters; DINOv3
and JEPA-WM remain frozen.

After a required four-frame controller warm-up, both complete unseen seeds
passed all 62 operational rollouts: 124/124 first-action gates, aggregate mean
cosine `0.975956`, and mean sequence MSE `0.000212981`. Reproduce the held-out
gate with:

```bash
./ops/aws.sh jepa-wm-proposal-eval \
  domain-20260823T113209Z-1400-held-00 wrist 4 62 1 \
  quantis_isaac_wrist_action_proposal_motion_state_12seed
./ops/aws.sh jepa-wm-proposal-eval \
  domain-20260823T113209Z-1400-held-01 wrist 4 62 1 \
  quantis_isaac_wrist_action_proposal_motion_state_12seed
./ops/aws.sh jepa-wm-proposal-summarize \
  domain-20260823T113209Z-1400-held-00,domain-20260823T113209Z-1400-held-01 \
  wrist 4 62 1 quantis_isaac_wrist_action_proposal_motion_state_12seed
```

The strict summary verifies whole-seed split provenance, recomputes every
aggregate, and requires each seed to clear the thresholds. Artifacts persist at:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_proposal_motion_state_12seed.pth
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_proposal_motion_state_12seed.pth.json
/home/ubuntu/docker/jepa-wm/checkpoints/experiments/quantis_isaac_wrist_action_proposal_motion_state_12seed_readiness.json
```

A tight CEM search around the proposal lowered JEPA-WM latent energy but did
not improve directional accuracy, so it remains an offline diagnostic rather
than the promoted command path.

### Simulator-only closed-loop bridge

Start the resident worker once, inspect it, and run one consume-once control
session against a whole held-out reference trajectory:

```bash
./ops/aws.sh jepa-wm-control-worker-configure \
  quantis_wrist_control \
  quantis_isaac_wrist_action_proposal_motion_state_12seed \
  quantis_isaac_wrist_action_adapter
./ops/aws.sh jepa-wm-control-worker-start quantis_wrist_control
./ops/aws.sh jepa-wm-control-worker-status
./ops/aws.sh jepa-wm-control-step \
  domain-20260823T113209Z-1400-held-00 11400 \
  quantis_wrist_control
```

Run a bounded repeated rollout on the same live Isaac stage with:

```bash
./ops/aws.sh jepa-wm-control-rollout \
  domain-20260823T113209Z-1400-held-01 11401 3 \
  quantis_wrist_control 44
```

Isaac validates that the goal recording is the matching whole held-out seed,
then replays a deterministic exploration prefix through a complete segment
boundary (context index `44` above; the optional argument defaults to `4`),
captures synchronized observations, and writes a versioned request. A
session-derived nonce, response timestamp, and exact promoted
checkpoint path bind the separate GPU worker's native three-action proposal to
that request. Isaac consumes only a conservative first action after checking a
final simulator-only freshness deadlines (3.0 seconds from the synchronized
observation and 2.5 seconds from the model response), Cartesian and gripper
bounds, base-frame workspace, Franka joint
limits and velocity, actual joint-state drift, and a live hand contact sensor.
The live selector tries bounded translation/rotation/gripper scale profiles in
order and accepts the first profile that clears the same gate and IK branch;
uniform quarter- and one-eighth-scale profiles remain fallbacks.
Tight IK is followed by measured joint and Cartesian action tracking; a
tracking/contact failure is rolled back. The session is claimed before
actuation, so an interrupted command cannot be replayed. Raw requests,
responses, pre-state, post-state, contact readings, and the 512×512 post-action
frame persist under:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_sessions/<session>/
```

Two unseen one-step proofs passed the same conservative safety policy:

- seed `11400`, session `step-20260823T153339Z-11400`: 1.883 s observation age,
  translation/rotation cosine `0.9447`/`0.9300`, 0.059 mm translation error,
  0 N contact;
- seed `11401`, session `step-20260823T152202Z-11401`: 0.948 s observation age,
  translation/rotation cosine `0.9926`/`0.9913`, 0.060 mm translation error,
  0 N contact.

The first repeated canary, `rollout-20260823T155348Z-11401`, then completed all
three fresh observe-infer-apply cycles on unseen seed `11401`. Mean observation age
was `1.322 s` (maximum `1.525 s`), all three steps measured 0 N contact, and the
terminal observation reduced translation error by `0.519 mm`, rotation error by
`0.001793 rad`, and gripper-closedness error from `0.4746` to `0.4094`. Its
validated report persists at:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_rollouts/rollout-20260823T155348Z-11401/report.json
```

The report distinguishes originally requested, actually attempted, and applied
steps and validates the single-use session chain, unique observation IDs, fixed
goal/proposal provenance, ordered capture times, warm-up progression, and
previous-action continuity. Each follow-up first verifies that the live joints
and end-effector still match the preceding measured result, then captures RGB
and state at the same update boundary. An exit finalizer also persists a typed
terminal report when capture, inference, transport, or execution orchestration
fails, including the incomplete final attempt rather than silently losing it.

The live process now retains one bound Isaac articulation/runtime across
follow-up calls, avoiding stale paused-physics tensors and repeated setup.
Motion-rich held-out rollout `rollout-20260824T032653Z-11401` selected context
`44`, applied all three JEPA proposals with 0 N contact, moved the end effector
`15.440 mm`, and reduced translation-to-target error by `11.127 mm`. This is
useful receding-horizon motion evidence, not a grasp or cable insertion: the
plug was never attached and the task was not completed. It must not be shown as
a successful lab result.

This remains narrow free-space execution evidence. The scale profiles are a
small deterministic command-safety projection set. Each completed session
now also triggers a separate proposal-centered CEM search over 256 bounded
three-action candidates. The adapted JEPA-WM ranks candidates by terminal
latent goal energy plus a proposal prior; a strict first-action direction gate
and a 1 mm / 4 mrad / 0.02-gripper trust region constrain the comparison. The
winning candidate is labeled `shadow_only`, then Isaac counterfactually runs
its first action through the same pose, IK, workspace, joint, collision, and
force projection without moving the robot. `shadow.json` and
`shadow_safety.json` persist beside the command evidence, and rollout reports
aggregate energy improvement, planning time, direction-gate passes, and safety
passes. This path has no command authority and does not delay the direct
proposal's freshness deadline.

Held-out rollout `rollout-20260823T203534Z-11401` proved the isolated shadow
path on AWS. All three direct actions applied with 0 N contact before any CEM
work began. Mean direct inference was `0.328 s`; mean/max observation age was
`1.366`/`2.013 s`, and mean response-to-actuation command age was `1.038 s`.
All three 256-candidate searches then improved latent energy
and retained first-action direction, with mean improvement `0.000221416`; all
three winners passed the no-actuation Isaac safety projection. The direct
rollout improved gripper-closedness error by `0.1104`, but translation and
rotation error worsened by `0.755 mm` and `0.002142 rad`, so the shadow candidate still has no command
authority. The validated aggregate report persists at:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_rollouts/rollout-20260823T203534Z-11401/report.json
```

The realized comparison uses independent reset-identical rollouts rather than
inferring baselines from model scores. Run the non-model trials and then build
the strict provenance-bound report with:

```bash
./ops/aws.sh jepa-wm-control-baseline \
  domain-20260823T113209Z-1400-held-01 11401 3 zero
./ops/aws.sh jepa-wm-control-baseline \
  domain-20260823T113209Z-1400-held-01 11401 3 scripted
./ops/aws.sh jepa-wm-control-baselines \
  baseline-proof-20260823-11401 \
  rollout-20260823T203534Z-11401 \
  zero-20260823T204640Z-11401 \
  scripted-20260823T205005Z-11401 \
  domain-20260823T113209Z-1400-held-01 11401 3 \
  quantis_isaac_wrist_action_proposal_motion_state_12seed
```

All three trials applied three actions with 0 N contact. Zero-action physics
drift improved translation/rotation error by `0.371 mm`/`0.000481 rad`.
Direct control worsened them by `0.755 mm`/`0.002142 rad`, while improving
gripper-closedness error by `0.1104`. The scripted reference improved all three
dimensions: `13.325 mm`, `0.006869 rad`, and `0.08350`. The report therefore
fails direct-versus-zero for translation and rotation and fails the scripted
tolerance for the same dimensions. Candidate command authority remains false.
The strict report persists at:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_baselines/baseline-proof-20260823-11401/report.json
```

This does not establish grasp, cable, contact-recovery, or insertion
capability. The next gate is an isolated reset-identical trial of the shadow
CEM winner, followed by repetition across whole held-out seeds.

That isolated trial is available as an explicitly non-production command:

```bash
./ops/aws.sh jepa-wm-control-candidate \
  domain-20260823T113209Z-1400-held-01 11401 \
  rollout-20260823T203534Z-11401-00 \
  baseline-proof-20260823-11401
```

The command resets Isaac, captures a fresh observation under the reserved
`experimental_shadow_candidate` identity, and accepts a prior shadow winner
only when its source search and no-actuation safety evidence passed and the new
reset matches the source pose, all seven joints, target, action history,
collision state, and contact force. The fresh execution still passes the full
live safety/freshness/IK/tracking gate. Its binding is labeled
`reset_trial_only`; a missing binding blocks execution, and a normal session
rejects the experimental artifact.

Candidate experiment `candidate-proof-20260823T213129Z-11401` applied safely at
`0.815 s` observation age with 0 N contact, but failed the realized outcome
gate. It worsened translation error by `0.565 mm` and rotation error by
`0.000672 rad`; it beat neither zero nor direct on those axes and also trailed
direct gripper progress. The CEM source had improved predicted latent energy by
`0.000125954`, demonstrating that this model objective is not yet aligned with
realized task-space progress. Production authority remains false. The strict
report persists at:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_candidates/candidate-proof-20260823T213129Z-11401/report.json
```

The planner can now fit a non-production action-response calibration from
realized direct, scripted, or reset-candidate sessions:

```bash
./ops/aws.sh jepa-wm-objective-calibrate \
  quantis_action_response_mixed_11401 \
  rollout-20260823T203534Z-11401-00,rollout-20260823T203534Z-11401-01,rollout-20260823T203534Z-11401-02,scripted-20260823T205005Z-11401-00,scripted-20260823T205005Z-11401-01,scripted-20260823T205005Z-11401-02
```

The calibration artifact persists every raw proposed/realized action and
recomputes its gains, alignments, and per-axis directional coverage whenever it
is loaded. A fitted scalar cannot be edited independently of that evidence.
The six realized actions produced translation, rotation, and gripper
alignments of `0.9993`, `0.9275`, and `1.0`; reranking is enabled only when the
translation and rotation evidence each cover at least three distinct
directions. The worker artifact manifest below binds the proposal, adapter, and
calibration as one identity. Starting it adds a continuous task-space
regression penalty to the latent-energy objective while retaining
`shadow_only` authority:

```bash
./ops/aws.sh jepa-wm-control-worker-configure \
  quantis_calibrated_control \
  quantis_isaac_wrist_action_proposal_motion_state_12seed \
  quantis_isaac_wrist_action_adapter \
  quantis_action_response_mixed_11401
./ops/aws.sh jepa-wm-control-worker-start quantis_calibrated_control
```

Calibrated experiment `candidate-proof-20260823T220355Z-11401` safely applied
at 0 N contact. Unlike the latent-only winner, it beat the direct proposal on
all three axes and improved translation by `0.0267 mm`, rotation by
`0.000911 rad`, and gripper progress by `0.04426`. It still trailed zero's
translation drift by about `0.052 mm` and missed scripted translation
tolerance, so its strict gate and production authority remain false. The
report persists at:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_candidates/candidate-proof-20260823T220355Z-11401/report.json
```

Before calibrated shadow reranking, the target frame and target pose are
revalidated together against the exact reference telemetry. Every unresolved
task axis must now clear a persisted minimum predicted error reduction: `0.1
mm` translation, `0.001 rad` rotation, and `0.005` gripper closedness. The
report records initial error, predicted error, predicted reduction, required
reduction, maximum allowed error, and the resulting pass decision for both the
direct and planned candidates. Legacy reports retain their original zero-margin
interpretation when reloaded.

The worker manifest owns these three thresholds, so stricter shadow experiments
do not require code changes. Supply all three together after the calibration;
uncalibrated workers reject them:

```bash
./ops/aws.sh jepa-wm-control-worker-configure \
  quantis_seed11400_margin500_control \
  quantis_isaac_wrist_action_proposal_motion_state_12seed \
  quantis_isaac_wrist_action_adapter \
  quantis_action_response_seed11400_v1 \
  0.0005 0.001 0.005
```

The manifest also owns the bounded CEM planner identity. Append the seed,
iterations, samples, and elite count together to make search experiments exact
and replayable. The native three-action horizon remains fixed:

```bash
./ops/aws.sh jepa-wm-control-worker-configure \
  quantis_train1400_margin500_seed235_control \
  quantis_isaac_wrist_action_proposal_motion_state_12seed \
  quantis_isaac_wrist_action_adapter \
  quantis_action_response_train1400_v1 \
  0.0005 0.001 0.005 \
  235 5 128 12
```

Live shadow session `step-20260823T224924Z-11401` cleared all three margins. It
predicted reductions of `0.218 mm` translation, `0.00233 rad` rotation, and
`0.0419` gripper error, improved latent energy by `0.000182`, and passed the
no-actuation simulator safety projection at quarter
translation/rotation/gripper scale. It remained `shadow_only`; the planned
candidate was not applied.

Use the repeatable collection command to capture 3–12 fresh, live-gated direct
trials with shadow search deferred and fit one calibration from their measured
proposed/realized actions:

```bash
./ops/aws.sh jepa-wm-control-calibration-collect \
  quantis_action_response_train1400_v1 \
  domain-20260823T113209Z-1400-train-00 \
  1400 6 quantis_uncalibrated_control
```

The collection runner persists the dedicated `calibration_collection` policy.
That policy accepts only `train` recordings, while ordinary direct, baseline,
candidate, and evaluation sessions continue to require `held_out`. It uses the
same model response, simulator safety, action-tracking, and raw-evidence fitter,
but cannot act as candidate evaluation evidence.
The fitter repeats the policy/split check for every raw session. Candidate
readiness repeats it again from the calibration trial IDs, so held-out direct,
scripted, or reset-candidate sessions cannot be relabeled as training evidence.

The seed-11400-only artifact generalized to seed 11401 in shadow session
`step-20260823T230446Z-11401`. Its candidate predicted `0.171 mm`, `0.00152
rad`, and `0.0206` gripper progress, cleared every margin, improved latent
energy by `0.000143`, and passed no-actuation safety at quarter scale. Isolated
reset trial `candidate-20260823T230817Z-11401` then realized `0.137 mm`,
`0.00118 rad`, and `0.0412` progress at 0 N contact. Strict report
`disjoint-candidate-20260823-11401` shows the candidate beat both direct and
zero on every axis, matched scripted rotation/gripper tolerance, and still
missed scripted translation tolerance. The overall gate and production
authority therefore remain false.

When a candidate source is a standalone control step rather than a rollout,
pass its exact session as the optional final argument to
`jepa-wm-control-baselines`; the report retains and revalidates that source
instead of inferring a `ROLLOUT-00` session name.

The stricter `0.5 mm` translation gate produced cross-seed improvement in both
directions. Calibration on seed 11400 and evaluation on seed 11401 produced
shadow session `step-20260823T232339Z-11401`; its candidate predicted `0.697
mm`, `0.00143 rad`, and `0.0222` gripper progress. Reset trial
`candidate-20260823T232732Z-11401` realized `1.531 mm`, `0.00262 rad`, and
`0.04444`, beat zero and direct on every axis, reached scripted tolerance on
every axis, and passed the strict candidate gate at 0 N contact.

The reverse experiment fit calibration only on seed 11401 and evaluated seed
11400. Shadow session `step-20260823T233919Z-11400` predicted `0.958 mm`,
`0.00110 rad`, and `0.0374`; reset trial
`candidate-20260823T234655Z-11400` realized `2.188 mm`, `0.000477 rad`, and
`0.03741`, again beating zero and direct on every axis at 0 N. It missed the
scripted translation tolerance by about `0.057 mm`, so strict aggregate
readiness and production authority remain false.

The reciprocal seed-11400/11401 experiments are two-fold cross-validation, not
a globally held-out readiness set: each evaluation seed appears in the other
trial's calibration. The readiness command reloads every trial from raw
sessions and rejects that global overlap:

```bash
./ops/aws.sh jepa-wm-control-candidate-summarize \
  candidate-proof-20260823T232732Z-11401,candidate-proof-20260823T234655Z-11400 \
  cross-seed-candidate-readiness-20260823-v1
```

The historical report below records the pre-gate cross-validation summary (two
seeds and one strict pass), but is not held-out readiness evidence. A valid
readiness artifact requires a calibration set disjoint from both 11400 and
11401, two strict held-out passes, and still cannot itself grant production
authority:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_candidates/cross-seed-candidate-readiness-20260823-v1/readiness.json
```

Training-only calibration `quantis_action_response_train1400_v1` was fit from
six live-gated trials on `domain-20260823T113209Z-1400-train-00` (seed 1400).
It has translation/rotation alignments `0.9991`/`0.8686` and six directions on
both vector axes. With the same `0.5 mm` worker gate, held-out session
`step-20260824T001739Z-11401` passed shadow search and safety, predicting
`0.538 mm`, `0.00166 rad`, and `0.0207` gripper progress. Held-out session
`step-20260824T001352Z-11400` failed closed despite `0.740 mm` predicted
translation: rotation missed the threshold by `0.000011 rad` and latent energy
worsened slightly. The next experiment varies only the persisted CEM search
configuration; no candidate is applied for the failed seed.

Seed 235 at the same 5-iteration, 128-sample, 12-elite budget cleared the 11400
search, but the accompanying 11401 result came from a different planner seed.
The strict readiness aggregator now reconstructs and compares the complete
worker/search identity and rejects that mixed configuration.

One fixed seed-237 worker then passed both held-out seeds. Source sessions
`step-20260824T014148Z-11400` and `step-20260824T014915Z-11401` bind the same
proposal, adapter, training-only calibration fingerprint, progress margins,
and `5 × 128 / 12 elites` planner. Their reset-only candidates are
`candidate-20260824T014710Z-11400` and
`candidate-20260824T015225Z-11401`. They respectively realized
`2.843 mm`/`0.001144 rad`/`0.03301` and
`2.659 mm`/`0.002756 rad`/`0.03845` translation/rotation/gripper progress.
Both beat zero and direct on every axis, reached scripted tolerance on every
axis, passed action tracking at 0 N with no collision, and remain
non-production. Resident baseline/candidate response construction keeps the
live command fresh, while repeated warm-start IK probes select the successful
solution closest to the captured seven-joint state before the unchanged safety
gate evaluates it.

The first globally disjoint two-seed readiness artifact is:

```text
/home/ubuntu/docker/isaac-sim/data/quantis/control_candidates/train1400-seed237-candidate-readiness-20260824-v1/readiness.json
```

It reconstructs training-only calibration seed `1400`, held-out evaluation
seeds `11400` and `11401`, the identical seed-237 worker policy, and reports
`strict_pass_count: 2` with
`candidate_readiness_passed: true`. `production_authority_granted` remains
hard-coded `false`; this milestone proves bounded isolated candidate search,
not autonomous repeated control.

Stop the worker independently with
`./ops/aws.sh jepa-wm-control-worker-stop`.

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
