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
    resident Unix-socket worker → safety gate → versioned receding-horizon control
```

The bootstrap proves the simulation, capture, stage-recognition, and base
JEPA-WM runtimes. The separate offline workflow additionally proves native
three-action rollouts and persistent lightweight action adaptation. A
whole-seed domain experiment clears both the action-conditioning and
inverse-action proposal gates. A simulator-only bridge now repeatedly captures,
infers, executes one fresh bounded command, measures the outcome, and replans
after workspace, joint, velocity, collision, force, and tracking interlocks.
Ordinary policies execute the first proposal action. After authenticated
contact-grasp attachment, the version-4 policy instead sums only the native
three-action translation while retaining the first action's rotation and
gripper command. This narrow, schema-bound exception clears measured
translation resolution without amplifying unqualified axes; every resulting
command still passes the same gates before execution.

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

The first exact bounded search exposed a systematic latent shortcut. With
proposal prior `1e-5`, a `1 mm / 0.004 rad / 0.02` trust region, and fixed
`4 × 64 / 8 elites` CEM, JEPA lowered energy below both zero and recorded
actions on all eight sampled insertion contexts for both held-out seeds.
Nevertheless, seed `12600` reached only `0.119557` mean first-action cosine and
seed `12601` only `0.0687623`; both failed the task-direction gate `0/8`.
These fingerprinted reports are preserved and recovery-backed up. Lower energy
alone is therefore not insertion control. The next offline profile adds a
fail-closed goal-action alignment objective computed from the observable
current-to-target DROID delta; it must preserve that direction as well as beat
the proposal/zero/recorded energies before any live trial.

That aligned profile exposed the next boundary rather than clearing it. On
held-out seed `12600`, all eight searched actions met the goal-direction gate,
but their mean latent improvement over zero was `-5.76454e-6` and only `1/8`
beat the recorded insertion action. On seed `12601`, the search improved over
zero by `+2.52775e-5` and beat the recorded action in `7/8`, but only `6/8`
actions met the pinned `0.95` goal cosine. The reports are preserved as a
second exact negative: adding a task penalty fixes most directions but does not
make the learned energy and insertion constraint agree on both whole seeds.

The planner now treats CEM output as a searched candidate, not automatically
as the selected action. A refinement is accepted only when it passes the goal
alignment gate and lowers JEPA latent energy relative to the proposal by at
least `1e-6`; otherwise the offline report falls back to the proposal only when
that proposal independently passes the observable goal-alignment gate. If both
fail, the context is explicitly blocked and has no selected action. This
Pareto gate is an evidence boundary, not live authority. The exact v3 reports
accepted `1/8` refinements on seed `12600` and `3/8` on seed `12601`. Proposal
fallbacks left `7/8` and `6/8` contexts selected and blocked the remaining
misaligned contexts instead of silently retaining them. Every selected action
passed goal alignment, with mean cosine `0.989333` and `0.973448`; selected
actions improved over zero by `+1.13512e-5` and `+4.70251e-5`, but the first
seed beat the recorded action only `1/7`. The following train-only adapter
checkpoint must therefore mine goal-aligned local candidates, rather than
generic perturbations, so the energy model learns to distinguish plausible
insertion refinements before the same fixed search is evaluated on fresh whole
held-out seeds.

Run that train-only checkpoint as a distinct artifact:

```bash
./ops/jepa_wm_insertion_wm_milestone.sh \
  1056 2600 contact-insertion-v9-2600 goal_aligned
```

Its candidate miner keeps all local sequences inside the persisted planner
bounds. Active demonstrated first actions must meet the `0.95` goal cosine;
sampled off-direction first actions are replaced with that recorded action.
Stationary demonstrated first actions—translation and rotation norms at or
below `1e-5` and gripper delta at or below `0.005`—remain stationary instead of
being forced through a direction cosine that is undefined at zero motion.
Later actions remain perturbed, so the adapter learns
a margin against plausible wrong-future insertion sequences rather than
receiving easy off-task negatives. The first attempted run correctly stopped
when 72 zero-action frames and the 12 sub-`1.6 µm` context-41 transition
actions exposed the missing stationary branch. No checkpoint was written, and
the negative result was recovery-backed up. The corrected axis-aware preflight
classifies 960 active and 96 stationary first actions across the exact
1,056-rollout corpus; every active action clears the cone with minimum cosine
`0.988497`. A second fail-closed run found 24 active first actions at contexts
36–37 whose demonstrated translation was `21.0–21.9 mm`, just outside the
planner's `20 mm` bound. The canonical bounds projection brings those fallbacks
to `20 mm` while preserving the cone (minimum remains `0.988497`); candidate
replacement now uses that bounded fallback and rechecks alignment. That run
also wrote no checkpoint and was recovery-backed up. The profile has a
distinct checkpoint name and serialized mining contract.
That bounded run completed as adapter fingerprint `236514a7...d9dd1`, but the
strict diagnostic gate remained closed. Seed `12600` retained positive mean
improvement (`+6.11294e-5`) while falling to `60.2273%` recorded-action wins;
seed `12601` passed at `+6.30901e-5`/`82.9545%`. The 176-rollout aggregate was
`+6.21097e-5`/`71.5909%`, below the pinned `75%` win-rate threshold. The
failure is concentrated in the sub-millimetric insertion tail: the global
planner-bound perturbation averaged `5.34 mm` there (95th percentile
`13.22 mm`), even though the demonstrated first actions had decayed to
`0.02–0.70 mm`. The result and its 12 GB recovery copy are preserved.

The next distinct `goal_aligned_relative` profile keeps the same bounds,
stationary rule, and `0.95` cone, but scales candidate noise from each
demonstrated action and caps it at the planner bounds. On the exact corpus this
reduces late-tail perturbations to `0.109 mm` mean and `0.361 mm` at the 95th
percentile without changing early large-motion bounds. Its explicit noise
floors are `10 µm` translation, `10 µrad` rotation, and `0.005` normalized
gripper delta; they are serialized independently from the stationary
classifier. Run it as a new artifact so the failed global-noise checkpoint is
not overwritten:

```bash
./ops/jepa_wm_insertion_wm_milestone.sh \
  1056 2600 contact-insertion-v9-2600 goal_aligned_relative
```

That diagnostic completed as fingerprint `93735353...97320`. Seed `12600`
improved to `+6.91447e-5`/`64.7727%`; seed `12601` passed at
`+7.02181e-5`/`84.0909%`. Aggregate evidence rose to
`+6.96814e-5`/`74.4318%`—131 wins in 176 rollouts, one aggregate win below the
nominal `75%` threshold—but strict readiness remains false because every whole
seed must pass. Relative scaling recovered five wins over the global-noise
profile. Its remaining losses versus the generic adapter are concentrated at
contexts 80–106, where the demonstrated insertion motion decays toward zero.
The checkpoint, reports, and 12 GB recovery copy are preserved.

The next adapter checkpoint must retain the generic adapter's already passing
late-tail discrimination while learning the aligned local margin: use the
fingerprinted generic adapter as an explicit warm start and a lower-rate
fine-tune, with that parent identity serialized into the child checkpoint.
The typed profile derives the generic `s1056` parent from the output identity,
requires identical corpus/selection/source provenance, and pins learning rate
to `1e-4`:

```bash
./ops/jepa_wm_insertion_wm_milestone.sh \
  1056 2600 contact-insertion-v9-2600 goal_aligned_relative_finetune
```

Training and evaluation remain offline; both current held-out seeds have
informed this design, so even a passing diagnostic rerun still requires fresh
whole held-out seeds before readiness. That exact warm-start diagnostic
completed as child fingerprint `396496bf...fadc`, bound to generic parent
fingerprint `47969a0a...af9` and training-config fingerprint
`f66e1d46...f485`. It visited all 1,056 TRAIN rollouts once at the pinned
`1e-4` rate. Seed `12600` passed at `+7.05642e-5`/`87.5%`; seed `12601`
passed at `+7.06248e-5`/`87.5%`; aggregate evidence was
`+7.05945e-5`/`87.5%` (154/176 wins). The exact checkpoint, reports, and
12 GB recovery copy are preserved. This closes the two diagnostic seeds, not
fresh-seed readiness: freeze this artifact and evaluate it without further
tuning on new whole held-out recordings before opening any planner or live
insertion gate.

That frozen evaluation is now complete. Without further adaptation, child
fingerprint `396496bf...fadc` passed two newly captured whole held-out seeds
`22600` and `22601`, which are disjoint from the complete source corpus and the
two diagnostic seeds. Seed `22600` scored `+4.78526e-5` mean improvement with
`80.6818%` recorded-action wins (71/88); seed `22601` scored `+1.97309e-5`
with `87.5%` wins (77/88). The aggregate was `+3.37918e-5` and `84.0909%`
(148/176), so the strict fresh whole-seed offline energy gate passed 2/2. Both
raw 112-frame contact-insertion recordings, their evaluation reports, the
typed fresh roster, the readiness report, and the checksum-verified 12 GB
recovery copy are preserved. This authorizes the frozen task-aware planner
benchmark only; it does not authorize JEPA actuation, insertion filming, or
production control.

The frozen task-aware planner benchmark then evaluated eight fixed insertion
contexts on each fresh seed with the exact adapter above, proposal fingerprint
`ea7ce27c...ea255`, and unchanged `4 × 64 / 8 elites` search. Seed `22601`
passed all eight contexts: selection, first-action direction, and goal
alignment were `100%`, and selected actions beat recorded actions `8/8` with
`+2.02691e-5` mean improvement. Seed `22600` passed only `7/8`: context `44`
was correctly blocked because both the proposal initializer (`-0.360` goal
cosine) and searched candidate (`0.891`) missed the `0.95` goal cone; among
the seven selected contexts, only `4/7` beat recorded energy despite positive
mean improvement. The reconstructive whole-seed planner gate therefore records
`1/2`, keeps live insertion closed, and grants no production authority. The
two raw planner reports, failed readiness artifact, and checksum-verified 12 GB
recovery copy are preserved. The next revision must improve early-stroke
proposal alignment using TRAIN-only evidence, then freeze and evaluate on a
new disjoint fresh pair rather than tuning on seeds `22600–22601`.

That TRAIN-only diagnosis found a concentrated initializer failure. Under
proposal `ea7ce27c...ea255`, context `44` cleared the `0.95` observable-goal
cone in only `3/12` TRAIN recordings, with `0.863752` mean and `0.451620`
minimum goal cosine. A separately named proposal adds a
first-action-to-three-step-goal direction loss while retaining the existing
action, history, goal-delta, task-progress, gripper, and stationary-hold
supervision. Its fingerprint is `efdf848c...7596f`; it was trained on the same
exact 1,056 TRAIN rollouts with goal-direction weight `1.0` and no held-out
adaptation. Context `44` then passed `12/12` at `0.998467` mean and `0.997284`
minimum cosine. Across all 1,020 nonzero-goal TRAIN rollouts, `1,019` passed the
goal cone. The ordinary proposal readiness gate also passed all 176 diagnostic
held-out rollouts with `100%` first-action and active-direction pass rates and
`0.998676` mean active cosine.

With the adapter and planner policy unchanged, the revised initializer cleared
the two now-diagnostic planner seeds: both selected `8/8` contexts and passed
all selected first-action and goal-alignment checks. Seed `12600` beat zero and
recorded actions `8/8`, with `+1.70849e-5` and `+8.94780e-6` mean improvements.
Seed `12601` beat zero `8/8` and recorded `6/8`, with `+5.04660e-5` and
`+3.22150e-5` mean improvements. Context `44`, formerly blocked on fresh seed
`22600`, is now selected on both diagnostic seeds with goal cosine above
`0.998`. These results justify freezing the exact proposal bytes, but they do
not reopen live insertion: the next gate is the identical planner on a new
two-seed recording pair disjoint from every seed used above.

That first frozen-proposal fresh pair, seeds `32600–32601`, exposed a distinct
late-tail search failure rather than another direction failure. The unchanged
256-candidate CEM selected all contexts with `100%` direction/goal passes, but
seed `32601` beat zero on only `6/8` contexts and recorded actions on `5/8`;
seed `32600` beat them `8/8` and `7/8`. The strict result was therefore `1/2`.
The misses at contexts `84`, `92`, and `100` occurred even though each exact
same-context TRAIN sequence lay inside the existing proposal trust region.
A lower goal-direction-weight proposal (`2f704c2a...e392`) improved tail action
MSE but regressed the diagnostic planner to `5/8` recorded-action wins on seed
`12600`, so it was preserved as a negative and not promoted.

The planner now supplements its unchanged `4 × 64 / 8` CEM search with the 12
same-context TRAIN sequences, one per training recording. Each sequence is
projected through the existing proposal-centered trust region before scoring;
no unbounded or held-out action is replayed. This produces 268 bounded
candidates per rollout and selects a context-matched candidate only when its
complete latent/prior/task objective is lower than CEM's. On diagnostic seeds
`12600–12601`, the revised policy beat zero and recorded actions `8/8` on both.
On the now-observed `32600–32601` pair it also reconstructed a strict `2/2`:
seed `32600` improved over zero/recorded by `+5.34571e-5`/`+3.82265e-5`, and
seed `32601` by `+2.61852e-5`/`+2.11138e-5`, with `8/8` wins and all gates
passing on each seed. Historical 256-candidate reports remain preserved; typed
readiness selects exactly one report matching the current complete policy
instead of guessing by filename. Because these seeds informed this revision,
the result is diagnostic only. The identical frozen policy must pass a new
disjoint two-seed pair before any live insertion gate can open.

That untouched promotion pair is now complete. Contact-aware v9 recordings
`contact-insertion-v9-2600-fresh-42600-held-00` and `...held-01` were captured
at reserved seeds `42600–42601` after the policy was committed, with zero
measured contact and strict 112-frame validation. The frozen adapter first
passed all 88 rollouts per seed: `+6.19085e-5`/`81.8182%` and
`+4.72146e-5`/`79.5455%` mean improvement/recorded-action wins. The exact
268-candidate planner then selected all eight fixed contexts on both seeds,
passed every first-action and goal-alignment gate, and beat zero and recorded
actions `8/8` on each. Seed `42600` improved by
`+5.84725e-5`/`+3.94733e-5`; seed `42601` by
`+2.33684e-5`/`+1.70866e-5`. Reconstructive readiness passed `2/2`, and the
raw recordings, reports, typed rosters, readiness artifacts, and verified
12 GB recovery copy are preserved. This closes the frozen offline planner
milestone and opens development of the live simulator shadow/safety gate. It
does not yet authorize actuation, insertion filming, or production control.

### Dense insertion planner checkpoint

The first full-stroke diagnostic then exposed an indexing error rather than a
model failure. The initial dense profile evaluated result frames `44–107` as
if they were command contexts; context `107` is the last insertion result, not
an observation that launches another insertion command, and its following
settle frames make its three-frame goal zero. The strict run therefore failed
closed there. The command contract now covers observations `43–106`, whose 64 first actions
produce insertion-result frames `44–107`; the historical eight-context profile
remains unchanged. Rerunning the corrected profile on the now-diagnostic
`42600–42601` pair passed every command, but those recordings had informed the
command-window correction and were not used for promotion.

The corrected dense policy has now passed an untouched frozen-artifact pair.
Contact-aware v9 recordings
`contact-insertion-v9-2600-fresh-52600-held-00` and `...held-01` were captured
at reserved seeds `52600–52601` after the correction was committed. Before
planning, frozen adapter `396496bf...fadc` independently passed all 88
post-attachment rollouts on each seed: `+1.40812e-5`/`84.0909%` and
`+3.80305e-5`/`86.3636%` mean improvement/recorded-action wins. Aggregate
evidence was `+2.60558e-5`/`85.2273%` over 176 rollouts.

The exact 268-candidate dense planner then evaluated all 64 command contexts
on both seeds with proposal `efdf848c...7596f`, adapter
`396496bf...fadc`, base checkpoint `daa69198...f4aa`, and unchanged
`4 × 64 / 8 elites`, seed-234 search. Seed `52600` selected `64/64`, passed
every direction/goal gate, and beat zero/recorded actions `64/64`, with
`+3.37709e-5`/`+1.59319e-5` mean selected improvements. Seed `52601`
selected `64/64`, passed every gate, beat zero `64/64` and recorded actions
`63/64`, and improved by `+4.97625e-5`/`+3.58768e-5`. Reconstructive dense
readiness passed `2/2`; all raw recordings, adapter reports, planner reports,
typed rosters, readiness artifacts, and the checksum-verified 12 GB recovery
copy are preserved.

This closes the frozen dense offline planner checkpoint, not the live one.
Planning took `2064.849 s` and `2066.569 s` per 64-command recording
(`4.687 GiB` peak), about 32 seconds per command—far beyond the current
3.0-second observation and 2.5-second command-age limits. The next checkpoint
is therefore a separately identified resident replay/shadow path that proves
the frozen JEPA-encoded proposal, or a reduced search that independently
retains the gate, can meet freshness before Isaac evaluates it without
actuation. Live insertion, filming, and production authority remain false.

The resident no-actuation checkpoint now passes `2/2` untouched seeds at
command context `43`. Seed-`52600` session
`insertion-safety-20260825T124011Z-52600-c43` used the exact proposal
fingerprint `efdf848c...7596f`; the resident response arrived `0.423 s` after
capture and the independent live safety snapshot was timestamped at `0.803 s`,
inside both freshness limits. Its first action requested `0.746 mm` translation in
the insertion direction. Isaac projected it at full translation/gripper and
quarter rotation, found an IK solution with `0.00250 rad` maximum joint
change, and measured `0 N` contact, no collision, and retained attachment.
The typed artifact passed with `authority: no_actuation`. The session contains
only request, response, state, and direct-safety evidence; it has no execution
claim or result, and a direct `claim_execution()` check is rejected by policy.
Seed-`52601` session `insertion-safety-20260825T125150Z-52601-c43` reproduced
the result with the same proposal and projection scale. Its `0.862 mm` first
translation received a response at `0.444 s` and a live-safety timestamp at
`1.081 s`; IK required `0.00289 rad` maximum joint change with `0 N` contact,
no collision, and retained attachment. Its typed evidence also passes with
`authority: no_actuation`, contains no execution/result files, and rejects an
execution claim. Both exact sessions and the checksum-verified 12 GB recovery
copy are preserved.

Two preceding attempts failed before capture because the persistent Isaac
process retained stale bytecode and then a stale recording-identity import;
both failed closed, ran backup, and produced no model action. Commits
`464e975` and `b3ac4db` make project reloads source-exact and dependency
ordered, with clean-process and generation-identity regressions. The `2/2`
result closes resident first-action freshness and no-actuation safety only. A
predeclared bounded live action trial remains required before any multi-step
insertion attempt. Insertion filming and production authority remain false.

That bounded trial is now complete, but it is a task-progress negative. An
initial invocation failed before capture because the long-lived Isaac process
had cached the absence of the newly synced insertion-trial module; backup still
completed, and commit `621b89a` invalidates finder caches before source
discovery and gives reset-trial failures typed report phases. The successful
retry, session `insertion-trial-20260825T133416Z-52600-c43`, rebound the exact
seed-`52600` source, proposal fingerprint, reset, and full-translation/
quarter-rotation scale. Its one `0.746 mm` requested translation was fresh
(`1.155 s` observation age, `0.6255 s` command age), tracked with `0.97734`
translation cosine and `0.159 mm` translation error, retained attachment, and
measured `0 N` peak contact with no collision. However, the action overshot its
three-frame target: translation error increased from `0.189 mm` to `0.567 mm`
(`-0.378 mm` progress), while rotation error also increased. The terminal
rollout and checksum-verified recovery copy are preserved. This opens no
multi-step, filming, insertion-completion, or production authority.

The follow-up controller gate therefore requires an insertion first-action
projection to reduce the current three-frame target translation error by at
least `25%`; a missing target fails closed. Against the measured trial, the
existing ordered scale policy rejects the full and half translation projections
as overshoots and first accepts the quarter-scale projection. Persisted
no-actuation and reset-trial projection evidence is reconstructed through the
same rule, so the earlier source evidence cannot authorize another trial under
the stronger code. The next live checkpoint is fresh `2/2` no-actuation
evidence for this target-progress policy, not another actuation attempt.

That fresh reevaluation initially exposed a separate paused-physics lifecycle
blocker. The first seed-`52600` attempt,
`insertion-safety-20260825T134911Z-52600-c43`, failed closed after a 34-hour
resident runtime pushed response-to-safety latency just outside the freshness
limits. Restarting Isaac restored normal RPC latency, but four subsequent
no-actuation attempts showed that the runtime reconstructed experimental
physics wrappers before the first resumed application update had restored a
usable PhysX tensor view. The latest negative,
`insertion-safety-20260825T144319Z-52600-c43`, received the resident response
but stopped before projection, safety approval, execution claim, result, or
model-commanded post-capture motion. Every failure ran the recovery backup.

The lifecycle fix now orders the boundary as play, one application/physics
update, pre-refresh contact/interlock continuity, wrapper reconstruction, a
second contact interlock, full safety-state read, captured-state continuity
validation, and pause in `finally`. A regression fails if reconstruction
precedes the physics-ready update. With the same frozen
proposal fingerprint `efdf848c...7596f`, exact-code session
`insertion-safety-20260825T154207Z-52600-c43` received its response in `0.437 s`
and recorded the live safety timestamp at `1.447 s`; session
`insertion-safety-20260825T154741Z-52601-c43` recorded `0.507 s` and `1.110 s`.
Both rejected the full and half translation projections as
`target_progress_insufficient`, accepted the quarter translation/rotation/
gripper projection, retained attachment, and measured `0 N` contact with no
collision. Their maximum predicted joint changes were `0.000508 rad` and
`0.000569 rad`. Each artifact passes with `authority: no_actuation` and contains
no execution claim or result. The fresh target-progress checkpoint is `2/2`,
and the checksum-verified recovery copy remains 12 GB. This authorizes only the
predeclared bounded one-action trial; multi-step insertion, filming, and
production authority remain false.

That bounded trial is complete and remains a progress negative. Session
`insertion-trial-20260825T155926Z-52600-c43` rebound the exact seed-`52600`
source, proposal fingerprint, captured reset, and quarter-scale projection. The
single fresh command requested `0.186 mm` translation and passed IK with
`0.000508 rad` maximum joint change. It executed with `0 N` peak contact, no
collision, retained attachment, and passing bounded tracking, but realized only
`0.0285 mm` translation progress: target error fell from `0.189 mm` to
`0.161 mm`, or `15.1%`, below the required `25%`. Rotation error increased from
`0.0000106 rad` to `0.000566 rad`. The terminal rollout and 12 GB recovery copy
are preserved. No second action was authorized. The next controller checkpoint
must fail closed on realized post-action target progress before opening another
step, then address the sub-millimeter tracking/noise floor; multi-step
insertion, filming, and production authority remain false.

### Insertion control-resolution checkpoint

This checkpoint is paused for correction before it can authorize another JEPA
action. The measurements below do **not** establish Franka arm precision: the
shared runtime path set drive targets and also called
`articulation.set_dof_positions()`, so the reported response combined direct
simulator state-setting with subsequent physics settling. Direct state-setting
is now restricted to explicit scene reset/initialization; runtime probes and
rollbacks use drive targets only. The corrected diagnostic also requires a
stable observed pre-probe baseline before measuring zero-command drift.

Accordingly, the prior `30`-`100 um` result supports only a roughly `0.187 mm`
settling/noise floor under the old simulator path. The prior `0.5 mm` request
realizing `0.558 mm` describes that path only. The rejected `1.0 mm` request
was a fixed-`0.25 s` joint-velocity-gate result, not evidence that the arm
cannot resolve `1.0 mm`. The corrected benchmark keeps the `3 mm` insertion
tolerance and every force, collision, attachment, joint, and velocity limit;
it gives the `1.0 mm` request a persisted `0.5 s` period instead of weakening
the velocity gate. It will measure zero, `0.5`, and `1.0 mm` drive-only probes
three times at representative insertion contexts `43`, `74`, and `106`, both
with and without the attached plug/load. Until those results establish a
defensible command deadband and orientation-hold tolerance, no further JEPA
insertion action is authorized.

The first drive-target-faithful full-roster attempt failed closed on its first
attached context before any nonzero probe. Session
`insertion-resolution-attached-20260825T231836Z-52600-c43` preserved a current
schema-v4 failure with zero completed samples. The HOLD probe issued no drive
command, began `0.825 mrad` from the unchanged active target, and exhausted its
32-update window as target-relative error rose from `0.820 mrad` to
`0.977 mrad`; command-free recovery also timed out at `1.003 mrad`. Peak
contact remained `0 N`, collision stayed false, attachment was retained, and
both backup layers verified the 12 GB recovery copy. The failure showed that
two locally small `0.25 s` baseline intervals could hide cumulative drift and
that zero-command drift had been checked against a controller target it never
wrote. The corrected baseline now requires all states in an eight-interval
window to be mutually within the unchanged tolerances. HOLD settlement is
measured from its exact live start while its separately reported steady-state
tracking error remains bound to the unchanged active drive target. This is a
preserved diagnostic negative, not control authority.

The first global-window retry then exposed that baseline qualification was
performed inside capture before an authenticated session existed. It stopped
without a probe but could not preserve its trace. An interim diagnostic moved
qualification behind strict current-frame capture so the authenticated
measurement boundary could retain the exact failure. Exact-code session
`insertion-resolution-attached-20260825T235817Z-52600-c43` subsequently
preserved a schema-v4 failure with all 41 states from the bounded `10 s`
attempt. The arm moved `33.538 mm` and `129.945 mrad` over the full attempt as
it approached the unchanged drive target. Its best window was the final
`2 s`: translation drift was `0.117 mm`, orientation drift `0.091 mrad`, and
plug-axis drift `0.117 mm`, all within their unchanged limits, with `0 N`
contact, no collision, and retained attachment. Maximum joint drift was still
`0.461 mrad`, above the unchanged `0.250 mrad` baseline limit. Because the
best window was the final one and was still improving, the next retry extends
only the bounded observation cap from `10 s` to `20 s`; it does not loosen a
tolerance, force/collision/attachment limit, settlement rule, or velocity
gate. Both backup layers again verified the 12 GB recovery copy. No HOLD or
translation probe executed, and no control authority was granted.

The bounded-`20 s` retry then qualified after 44 intervals, but failed the
independent capture-to-baseline continuity gate. Session
`insertion-resolution-attached-20260826T001258Z-52600-c43` reconstructed the
raw capture and found that the first resumed baseline state was already
`5.941 mm` and `23.614 mrad` away; the arm then moved another `33.568 mm` and
`130.025 mrad` before reaching its stable endpoint. Its qualifying final
window remained strict: `0.086 mm` translation, `0.148 mrad` rotation,
`0.237 mrad` joint, and `0.069 mm` plug-axis drift, with `0 N`, no collision,
and retained attachment. This proves that a strict image/telemetry read alone
does not make the preceding replay state a settled reset. The final lifecycle
therefore uses the same unchanged global-window policy at two distinct
boundaries: capture must settle before recording the strict current frame, and
measurement must requalify after pause/resume before any probe. Capture-side
timeout now writes a dedicated typed failure artifact with the complete trace,
source identity, load, limits, and false authority claims. The post-resume
gate still authenticates continuity to the settled capture. No tolerance or
safety limit changed, no probe executed, and both 12 GB backups were verified.

The first run with that final lifecycle,
`insertion-resolution-attached-20260826T003402Z-52600-c43`, completed all three
HOLD repetitions before the first `0.5 mm` probe failed closed at safety
projection with `joint_velocity_violation`. HOLD issued no drive command.
Maximum realized zero-command translation was `0.00188 mm`, orientation drift
was `0.00416 mrad`, and settlement-relative joint error was `0.00393 mrad`;
each repetition settled over its explicit `0.25 s` observation interval.
Start repeatability remained within `0.00686 mm` translation and `0.0138 mrad`
joint difference, while rollback repeatability remained within `0.00907 mm`
and `0.0176 mrad`. The separately reported error to the unchanged active
controller target was `1.003 mrad`. Contact remained `0 N`, collision stayed
false, and attachment was retained. The first translation did not actuate.
Rather than weaken the unchanged `0.5 rad/s` velocity gate, the next exact-code
retry gives both `0.5 mm` and `1.0 mm` probes a persisted `0.5 s` command
period. Both 12 GB backups were verified; this is still diagnostic evidence,
not insertion authority.

The exact-code `0.5 s` retry,
`insertion-resolution-attached-20260826T010729Z-52600-c43`, again completed all
three HOLD repetitions and then rejected the first `0.5 mm` probe before any
drive command. Its authenticated schema-v5 failure now preserves the complete
rejected projection rather than only an error string. The IK result required a
`1.613083 rad` maximum joint change, so the unchanged `0.5 rad/s` velocity gate
would require at least `3.226165 s` even before tracking and settling. That is
an inadmissible joint-configuration jump for a `0.5 mm` diagnostic probe, not a
reason to extend the period blindly. The completed HOLD intervals retained the
plug with `0 N` peak contact and no collision; both 12 GB recovery backups were
verified. The remaining poses, unloaded cases, and `1.0 mm` probes were not
attempted. Local IK continuity must be repaired or bounded independently before
the drive-only roster can resume. No JEPA action, multi-step insertion,
filming, or production authority is granted.

The bounded local-branch follow-up,
`insertion-resolution-attached-20260826T012114Z-52600-c43`, found a
`0.006576 rad` IK solution and admitted the same `0.5 mm` request without
changing the velocity gate. The drive-only forward interval realized
`0.784349 mm` along the requested axis, with `0.321669 mm` translation error,
`1.394731 mrad` orientation drift, and `1.499214 mrad` settled joint error
after `2.25 s`. Rollback then improved monotonically from `3.591876 mrad` to
`1.325758 mrad`, but failed closed after `3.0 s` because it had not reached the
unchanged `1.257109 mrad` threshold. Peak contact remained `0 N`, collision
stayed false, attachment was retained, and both 12 GB backups were verified.
The next diagnostic retry extends only the bounded settlement observation cap
from 32 to 40 updates; it does not change the tracking threshold, velocity
gate, or any other safety limit. This is a measured control/rollback negative,
not insertion authority.

That retry, `insertion-resolution-attached-20260826T013600Z-52600-c43`, let
rollback finish its command-relative settlement but then rejected the exact
reset. Every other repeatability component passed: `0.074179 mm` translation,
`0.174079 mrad` rotation, `0.001199` normalized gripper difference, and
`0.063896 mm` plug-axis difference, with `0 N`, no collision, and retained
attachment. Maximum joint difference was `0.508428 mrad`, just above the
unchanged `0.500000 mrad` reset limit. The mismatch exposed a policy gap: the
rollback command-relative threshold could be looser than the reset criterion
that immediately followed it. An interim protocol therefore added a persisted
`0.400000 mrad` rollback tracking cap, requiring two consecutive passing
updates within the existing 40-update bound before checking the unchanged
full reset contract. Both 12 GB backups were verified. No second translation
probe ran, and no authority was granted.

The strict-cap retry,
`insertion-resolution-attached-20260826T015256Z-52600-c43`, preserved another
typed rollback timeout after 40 updates (`3.75 s`). Error to the active drive
target fell from `3.591876 mrad` to `1.193316 mrad`, still above the new
`0.400000 mrad` cap, while the exact final state was already only
`0.433058 mrad` from the persisted stable reference reset. This exposed a
target-semantics error rather than inadequate settling: with the attached load,
the stable observed joints are about `1.003 mrad` away from the drive target.
The runtime must command and verify the baseline drive target, but rollback
repeatability must settle against the stable observed reference. That
distinction is now explicit; the strict cap and all safety limits remain
unchanged. Peak contact was `0 N`, collision was false, attachment was
retained, and both 12 GB backups were verified.

The reference-target retry,
`insertion-resolution-attached-20260826T020712Z-52600-c43`, then reached
`0.433058 mrad` reference-relative error for its final two observations but
timed out because the interim `0.400000 mrad` margin was stricter than the
authoritative `0.500000 mrad` reset contract. The full 40-update trace fell
monotonically from `3.591876 mrad`; peak contact remained `0 N`, collision was
false, attachment was retained, and both 12 GB backups were verified. The
current rollback cap is now derived directly from the persisted reset joint
tolerance (`0.500000 mrad`) rather than maintaining a second arbitrary limit.
It still requires two consecutive passes and is followed by the unchanged
translation, rotation, gripper, plug, attachment, contact, and collision reset
checks. This removes an unjustified stricter duplicate; it does not loosen the
reset contract.

The reset-bound retry,
`insertion-resolution-attached-20260826T022159Z-52600-c43`, completed all three
HOLDs and the first two `0.5 mm` repetitions, each with exact rollback evidence.
The third forward interval also settled, but its rollback stopped after 40
updates (`3.75 s`) at `0.504404 mrad`, only `0.004404 mrad` above the unchanged
`0.500000 mrad` limit. The last trace segment decreased monotonically from
`0.966281 mrad`, and peak contact remained `0 N` with no collision and retained
attachment. The next retry extends only the bounded settlement observation cap
from 40 to 48 updates; the reset-derived tracking cap, all other reset checks,
and every safety limit remain unchanged. Both 12 GB backups were verified, and
the `1.0 mm` probe still has not run.

The fixed 48-update roster has now reached a terminal result at all six
predeclared pose/load cases. The current exact-code artifacts reconstruct on
the host and are present in the checksum-verified 13 GB recovery copy:

| Load | Context | Terminal result |
| --- | ---: | --- |
| attached | 43 | Session `insertion-resolution-attached-20260826T023754Z-52600-c43` completed all nine probes. Zero drift was `0.00188 mm` translation and `0.00416 mrad` orientation. The `0.5 mm` probes realized `0.789 mm` mean along-axis motion with `0.323 mm` maximum translation error, `1.395 mrad` maximum orientation drift, `1.608 mrad` maximum controller-tracking error, and `2.25 s` settlement. The `1.0 mm` probes realized `1.175 mm` mean with `0.226 mm`, `1.257 mrad`, `0.652 mrad`, and `2.25 s` corresponding maxima/time. |
| attached | 74 | Session `insertion-resolution-attached-20260826T025726Z-52600-c74` completed three HOLDs, then the first `0.5 mm` forward probe timed out after `4.5 s` at `1.004 mrad` versus the unchanged `0.500 mrad` requirement. Its interlocked rollback recovered in 16 updates at `0.443 mrad`. |
| attached | 106 | Session `insertion-resolution-attached-20260826T030823Z-52600-c106` completed all nine probes. Zero drift was `0.00140 mm` translation and `0.00290 mrad` orientation. The `0.5 mm` probes realized `0.863 mm` mean with `0.366 mm` maximum translation error, `1.240 mrad` maximum orientation drift, `1.766 mrad` maximum controller-tracking error, and `2.25 s` settlement. The `1.0 mm` probes realized `1.203 mm` mean with `0.252 mm`, `1.382 mrad`, `0.503 mrad`, and `2.083 s` mean (`2.25 s` maximum) settlement. |
| unloaded | 43 | Session `insertion-resolution-unloaded-20260826T032622Z-52600-c43` completed three exact HOLDs. Its first `0.5 mm` forward probe settled in `2.25 s`, realizing `0.784 mm` with `0.322 mm` translation error, `1.381 mrad` orientation drift, and `1.499 mrad` controller-tracking error; rollback then timed out after `4.5 s` at `0.688 mrad`. |
| unloaded | 74 | Session `insertion-resolution-unloaded-20260826T033559Z-52600-c74` completed three exact HOLDs, then the first `0.5 mm` forward probe timed out after `4.5 s` at `0.925 mrad`. Its interlocked rollback recovered in 13 updates at `0.498 mrad`. |
| unloaded | 106 | Session `insertion-resolution-unloaded-20260826T034609Z-52600-c106` completed three exact HOLDs. Its first `0.5 mm` forward probe settled in `2.25 s`, realizing `0.859 mm` with `0.361 mm` translation error, `1.161 mrad` orientation drift, and `1.605 mrad` controller-tracking error; rollback then timed out after `4.5 s` at `0.718 mrad`. |

All six cases recorded `0 N` peak contact and no collision; attached runs
retained attachment and unloaded runs remained unattached. The attached HOLDs
showed only `0.00140`-`0.00188 mm` translation drift and
`0.00290`-`0.00416 mrad` orientation drift, while unloaded HOLDs were unchanged
at the stored precision. Error to the unchanged active controller target was
nevertheless about `1.0 mrad` in every HOLD case. Across the two complete
attached reports, worst-case start/rollback repeatability remained within
`0.267 mm` translation, `0.394 mrad` rotation, `0.498 mrad` joints, and
`0.272 mm` plug-axis position.

This separates noise from command response, but it does not close the control
checkpoint. A `0.5 mm` request is clearly distinguishable from zero drift, yet
only two of six pose/load cases completed its full forward-and-reset cycle; the
settled responses also overshot to roughly `0.78`-`0.86 mm`. The `1.0 mm`
roster was reached only in those two successful attached cases and realized
roughly `1.17`-`1.20 mm`. Settled nonzero orientation drift reached
`1.395 mrad`, so the earlier `1.25 mrad` hold value is not defensible across
this roster. The next milestone remains controller-side: make drive-only
settlement and rollback repeatable across pose and load, then rerun this frozen
roster before selecting a command deadband or orientation-hold tolerance. The
`3 mm` insertion tolerance and all force, collision, attachment, joint, reset,
and velocity gates remain unchanged. No JEPA action, second receding-horizon
step, filming, or production authority is granted.

The next diagnostic measured the simulator/controller noise floor before
authorizing another insertion action. The first exact-code run stopped before
any probe because the provisional `20 um`/`0.1 mrad` repeat contract was
tighter than the resumed baseline. A second pre-actuation run exposed a missing
probe-observation import. After both fixes, strict session
`insertion-resolution-20260825T184027Z-52600-c43` executed one zero-action
probe and rejected its rollback with reconstructible raw evidence: `0.218 mm`
end-effector translation drift, `0.501 mrad` rotation drift, `0.396 mrad`
maximum joint drift, and `0.242 mm` plug-axis drift, with `0 N` contact, no
collision, and retained attachment. That negative established that the
original repeat threshold was measuring simulator settling, not useful control
resolution.

The diagnostic-only protocol therefore admits a bounded `0.5 mm` reset
envelope while recording the exact start and rollback repeatability of every
sample. This change does not alter insertion execution safety or grant control
authority. Each probe recomputes a translation-only retreat direction from its
exact live start, and reconstruction requires every nonzero target to increase
distance from the recorded insertion target before IK. The completed run is:

```bash
./ops/aws.sh jepa-wm-insertion-resolution \
  contact-insertion-v9-2600-fresh-52600-held-00 52600 43
```

Session `insertion-resolution-20260825T185437Z-52600-c43` completed all 12
probes (three each at `0`, `30`, `100`, and `200 um`) and independently
reconstructed the raw report. Maximum zero-action drift was `0.219 mm`
translation, `0.501 mrad` orientation, and `0.396 mrad` joint tracking.
Maximum start/rollback repeatability drift was `0.218 mm` translation,
`0.501 mrad` rotation, `0.396 mrad` joints, and `0.242 mm` plug-axis position.
The `30 um` and `100 um` requests both realized the same `0.187 mm` mean
along-axis motion, with maximum translation errors of `0.194 mm` and
`0.143 mm`; they are below effective control resolution. The `200 um` request
realized `0.387 mm` mean along-axis motion, but still had `0.219 mm` maximum
translation error and `0.915 mrad` maximum orientation drift. Every sample
retained attachment and recorded `0 N` contact with no collision. The typed
report, preceding negatives, and checksum-verified 12 GB recovery copy are
preserved.

The command-relative follow-up uses a `0.5 mrad` absolute joint-tracking floor
or `25%` of requested joint motion, whichever is larger. It requires two
consecutive passing physics updates and fails after 32 updates. Forward motion
and rollback each persist the complete passing window and independent peak
force/collision evidence. Session
`insertion-resolution-20260825T192355Z-52600-c43` completed all three zero and
all three `0.5 mm` probes, then rejected the first `1.0 mm` probe before
actuation as `joint_velocity_violation`; the limits were not weakened. Its
typed failure report preserves the six completed samples and the exact
pre-actuation rejection.

The safely narrowed exact-code run,
`insertion-resolution-20260825T193221Z-52600-c43`, then completed all six
probes (three each at `0` and `0.5 mm`) and independently reconstructed the raw
report. Maximum zero-command drift was `0.0851 mm` translation, `0.195 mrad`
orientation, and `0.154 mrad` joint tracking. Maximum start/rollback
repeatability drift was `0.0851 mm` translation, `0.195 mrad` rotation,
`0.154 mrad` joints, and `0.0943 mm` plug-axis position. The `0.5 mm` requests
realized `0.558 mm` mean along-axis motion with `0.0854 mm` maximum translation
error, `1.120 mrad` maximum orientation drift, and `0.154 mrad` maximum joint
tracking error. Every forward and rollback interval settled in two updates,
retained attachment, and recorded `0 N` peak contact with no collision. The
successful report, the `1.0 mm` safety negative, and the checksum-verified
12 GB recovery copy are preserved.

This was initially treated as closing the bounded-resolution diagnostic and
motivated selecting a target at least `0.5 mm` away plus a persisted
`1.25 mrad` orientation hold. That interpretation is now superseded by the
drive-only correction above; the earlier no-actuation policy evidence remains
preserved, but it does not reopen action authority.

That farther-target/orientation-hold no-actuation checkpoint passed both
reserved seeds. Session `insertion-safety-20260825T201612Z-52600-c43` selected
frame 48 at `0.513522 mm`; its frozen proposal requested `0.792258 mm`
translation while the `0.026835 mrad` target-orientation error caused the
safety policy to persist scale `(translation=1.0, rotation=0.0, gripper=1.0)`.
The response arrived in `0.505 s`, safety was timestamped at `1.929 s`, and IK
required `0.002203 rad` maximum joint change. Session
`insertion-safety-20260825T202426Z-52601-c43` selected frame 48 at
`0.512412 mm`; its `0.902936 mm` proposal failed the full-scale target-progress
gate, then passed at `(0.5, 0.0, 1.0)` with `0.001156 rad` maximum joint
change. Its response arrived in `0.442 s` and safety was timestamped at
`2.552 s`. Both retained attachment with `0 N` contact and no collision, carry
`authority: no_actuation`, contain no execution/result artifact, and are in the
checksum-verified 12 GB recovery copy.

This `2/2` result authorizes no motion. Before retesting exactly one JEPA
action, the corrected drive-only benchmark must establish the command-relative
settlement rule, command deadband, and orientation-hold tolerance. Execution
must also persist a realized target-progress decision; a failed decision must
terminate and roll back rather than opening another action. No second
receding-horizon action, insertion filming, or production authority is granted.

The corrected drive-only checkpoint is now terminal on the frozen six-case
pose/load roster. Runtime motion uses only drive targets; direct articulation
state-setting remains reset/initialization-only. Every physics advance is
interlocked, every run independently qualifies its stable baseline, and the
report reconstructs the exact active drive target, forward projection,
settlement trace, rollback target, and at most one bounded feedback correction.
The canonical artifacts are in the checksum-verified 13 GB recovery copy:

| Load | Context | Terminal result |
| --- | ---: | --- |
| attached | 43 | Session `insertion-resolution-attached-20260826T070513Z-52600-c43` completed `9/9`. Its report appeared after the original 900-second client bound; the runtime remained bounded and finished normally, the transport bound was corrected to 1800 seconds, and a second backup taken after Isaac was idle verified the report copy. |
| attached | 74 | Session `insertion-resolution-attached-20260826T064625Z-52600-c74` completed `9/9`. |
| attached | 106 | Session `insertion-resolution-attached-20260826T073301Z-52600-c106` completed `9/9` under the corrected transport lifecycle. |
| unloaded | 43 | Session `insertion-resolution-unloaded-20260826T062829Z-52600-c43` completed `8/9`; the final `1.0 mm` probe was rejected before actuation because IK jumped to a branch `1.612791 rad` away. The unchanged velocity gate was not weakened. |
| unloaded | 74 | Session `insertion-resolution-unloaded-20260826T081635Z-52600-c74` completed `7/9`; the second `1.0 mm` motion plateaued at `0.853741 mrad` versus its `0.784999 mrad` threshold, and interlocked recovery plateaued at `0.623346 mrad` versus the unchanged `0.500000 mrad` rollback cap. |
| unloaded | 106 | Session `insertion-resolution-unloaded-20260826T083236Z-52600-c106` completed `7/9`; the second `1.0 mm` motion plateaued at `0.840743 mrad` versus `0.776817 mrad`, and recovery plateaued at `0.630200 mrad` versus `0.500000 mrad`. |

All `27/27` attached probes completed with `0 N` contact, no collision, and
retained attachment. Maximum attached zero-command drift was `0.005516 mm`
translation and `0.012655 mrad` orientation. Across the three poses, the nine
`0.5 mm` requests realized `0.382719 mm` mean along-axis motion; the lowest
per-session mean was `0.349062 mm`, maximum translation error was
`0.173489 mm`, maximum orientation drift was `0.784174 mrad`, maximum
controller-target error was `1.468461 mrad`, and maximum settlement time was
`2.75 s`. The nine `1.0 mm` requests realized `0.746263 mm` mean; corresponding
maxima were `0.341800 mm` translation error, `0.784828 mrad` orientation drift,
`1.704818 mrad` controller-target error, and `3.5 s` settlement. Maximum
attached start/rollback translation repeatability was `0.188904 mm`.
`12/18` translating rollbacks required the sole bounded correction, so this is
controller evidence for one action, not autonomous repeated-control evidence.

The unloaded negatives are deliberately retained. They show that the same
controller is pose/load sensitive: context 43 encounters a discontinuous IK
branch on its third `1.0 mm` attempt, while contexts 74 and 106 reproducibly
exhaust the bounded motion and recovery windows on their second `1.0 mm`
attempts. Both current-schema settlement failures now persist and authenticate
their exact 48-update motion and rollback traces. A validator fix applies the
same persisted rollback cap used at runtime; projection-failure reconstruction
allows only the existing cross-runtime floating-point tolerance (the context-43
Euler recomputation differed by about `1.6e-13 rad`) and still rejects
`1e-10` tampering. All three unloaded sessions remained at `0 N` with no
collision and the plug intentionally absent.

This establishes the existing attached insertion-control target policy without
changing the `3 mm` task tolerance or any force, collision, attachment, joint,
reset, or velocity gate. Its `0.5 mm` minimum translation is more than sixty
times the worst attached zero drift, and its `1.25 mrad` orientation-hold
tolerance exceeds the observed `0.784828 mrad` maximum with margin. At context
43 the policy selects frame 48. Context 74's nearest target is `2.109314 mm`,
outside this corrected command benchmark, and context 106 has no future
bounded-horizon target.

The controller checkpoint did not make the earlier no-actuation sources valid
under the new drive-only replay. Trial
`insertion-trial-20260826T090105Z-52600-c43` rejected its old source binding
before persisting a response or action. The first new safety capture then
failed live continuity before IK. After the capture path qualified the same
stable all-pairs baseline used by resolution measurement, seed 52600 exposed a
second conditioning bug: its final frame was stable, but `previous_action`
spanned the last replay frame through the entire settling interval and falsely
reported `45.922 mm` of prior translation. The frozen proposal consequently
requested `13.518 mm`, and every scale failed target progress. Re-evaluating
the identical image, pose, and target with stationary action history produced
a `0.610 mm` proposal, isolating the failure without actuation.

Stable captures first conditioned action history on the final `0.25 s`
baseline interval. Sessions
`insertion-safety-20260826T092733Z-52600-c43` and
`insertion-safety-20260826T093844Z-52601-c43` measured only `0.00764 mm` and
`0.00797 mm` of prior translation. Their exact frozen proposals requested
`0.611583 mm` and `0.636877 mm`; both passed at full translation scale with
rotation held and maximum IK changes of `1.638727 mrad` and `1.714668 mrad`.
The target distances were `0.741350 mm` and `0.718949 mm`. Both retained
attachment with `0 N` contact and no collision, carry
`authority: no_actuation`, and are preserved in the checksum-verified 13 GB
recovery copy. The full local suite passes `523` tests with `70` optional
dependency skips.

Reset-trial binding deliberately requires its source and execution action
history to match exactly, while micrometer-scale stable drift varies across
replays. A qualified terminal baseline therefore now represents its final
command history as the canonical HOLD action (all zeros); the physical drift
continues to be enforced by the typed reset and safety evidence. Fresh sessions
`insertion-safety-20260826T095029Z-52600-c43` and
`insertion-safety-20260826T095901Z-52601-c43` passed `2/2` with that exact
history. Their frozen proposals requested `0.609921 mm` and `0.635914 mm`,
passed at full translation scale with rotation held, and required
`1.634619 mrad` and `1.712302 mrad` maximum IK changes. Both again retained
attachment at `0 N` with no collision, and both 13 GB backups verified.

The first exact-bound trial,
`insertion-trial-20260826T100755Z-52600-c43`, authenticated the source,
independent reset, target, action history, and response, then stopped before IK
or motion. The required live resume/reconstruction/continuity snapshot made
the original observation `7.792 s` old and its bound response `6.695 s` old,
so all projections failed the unchanged `3.0 s` observation and `2.5 s`
command freshness limits. A typed execution refresh now reauthorizes only the
timing of that exact bound identity/action after the interlocked live snapshot
passes full captured-state continuity. It persists that live state and derives
the execution observation/response for report reconstruction; it does not
extend either freshness limit or rerun/replace the model action. The blocked
trial and its 13 GB recovery copy remain negative evidence.

Only this corrected fresh `2/2` evidence reopens the predeclared single
attached context-43 JEPA action with the realized-progress terminal gate. A
failed progress decision must stop and roll back. No second action, multi-step
insertion, filming, or production authority is granted.

That single action was attempted in session
`insertion-trial-20260826T112537Z-52600-c43`, bound to source session
`insertion-safety-20260826T095029Z-52600-c43` and frozen proposal fingerprint
`efdf848c120a2e4bba5b5e08f16093eb9b20695940e525906313e5cb1057596f`.
The refreshed gate accepted the full translation scale with rotation held;
the projected maximum joint change was `1.641288 mrad`, IK position error was
`0.000958 mm`, IK orientation error was `0.938542 mrad`, and the bound
observation age was `1.746403 s`.

Execution then failed closed before post-action progress measurement. Its
command-relative joint error decreased from `1.410369 mrad` to
`0.803017 mrad` over the bounded 32-update trace but did not reach the
unchanged `0.500000 mrad` settlement threshold. Interlocked rollback to the
refreshed live reset target was accepted, retained the plug, and reduced its
joint error from `1.315117 mrad` to `1.069546 mrad`, but it also exhausted the
32-update bound. The terminal result is therefore `rollback_failed`, with
`0/1` applied steps and no claimed Cartesian progress. Both phases remained at
`0 N` maximum contact with no collision and retained attachment. No second
action was requested. The exact forward and arm-plus-gripper rollback traces,
failure reasons, refreshed state, response, and rollout are preserved in the
verified 13 GB recovery copy.

This is controller-settlement negative evidence, not a JEPA, IK, contact, or
attachment failure and not an autonomous insertion success. Action authority
is closed again. Before another JEPA command, the drive-only controller must
demonstrate bounded forward settlement and verified rollback at this action
scale under the same `0.500000 mrad` threshold, force/collision/attachment
interlock, and reset-equivalence contract. Multi-step insertion, filming, and
production authority remain false.

The failure exposed one policy mismatch with the corrected drive-only
checkpoint: the insertion trial observed only 32 updates, while the validated
attached context-43 control-resolution roster uses 48. The insertion-specific
cap is now 48 and is persisted in the trial binding. The `0.500000 mrad`
threshold, two-consecutive-pass rule, velocity gate, gripper tolerance, and all
force, collision, attachment, and reset limits are unchanged; generic
settlement remains at 32 updates. The full local suite passes `536` tests with
`70` optional dependency skips, and both independent reviews pass.

Fresh current-code no-actuation sessions then requalified both reserved seeds.
Session `insertion-safety-20260826T113919Z-52600-c43` selected full translation
with rotation held and a `1.633083 mrad` maximum projected joint change.
Session `insertion-safety-20260826T120217Z-52601-c43` independently selected
the same scale with a `1.712435 mrad` maximum projected joint change. Both
recorded `0 N` contact, no collision, retained attachment, and
`authority: no_actuation`; both verified 13 GB backups. The first seed-52601
attempt, `insertion-safety-20260826T115339Z-52601-c43`, is retained as a
freshness negative: evaluation arrived about `2.824 s` after its response and
all scales failed the unchanged `2.5 s` command-age limit before IK.

This restores only the exact `2/2` pre-action boundary for a separately invoked
single context-43 retry. No action was launched from these safety sessions, and
they do not authorize an automatic second action, multi-step insertion,
filming, or production use.

The separately invoked 48-update retry,
`insertion-trial-20260826T130225Z-52600-c43`, disproved the hypothesis that the
32-update cap alone caused the prior failure. Its projected maximum joint
change was `1.639752 mrad`. Forward settlement requested `1.408707 mrad`; the
error reached a best `0.799910 mrad`, then regressed to
`0.899683 mrad` at update 48 without reaching the unchanged `0.500000 mrad`
threshold. Rollback requested `1.521826 mrad` and improved to
`1.032352 mrad` at update 48, so the terminal result again was
`rollback_failed` with `0/1` applied steps. Contact remained `0 N`, collision
stayed false, attachment was retained, and the 13 GB recovery copy verified.

That negative exposed a target-semantics mismatch rather than a JEPA, IK,
contact, or attachment failure. Under the attached load, the active Isaac
drive target and the stable realized reset differ by about a milliradian. The
insertion path had commanded the raw desired joints and then used the stable
realized reset itself as the rollback drive target. Commit `967e28e` separates
those concepts: capture persists the exact active drive target; the forward
command applies the same bounded `2 mrad` load-bias compensation validated by
the drive-only checkpoint while settlement remains relative to the desired
physical joints; rollback reapplies the captured active target while judging
return against the stable reset. The fixed `0.25 s` command period and complete
drive targets, including gripper width, are persisted and reconstructed. A
missing or over-bound target fails before actuation. No force, collision,
attachment, velocity, settlement, reset, progress, or orientation limit was
weakened. The full local suite passes `538` tests with `70` optional skips, and
both independent correctness and standards reviews pass.

Fresh exact-code safety then requalified the reserved seeds. The first seed
52600 attempt, `insertion-safety-20260826T134024Z-52600-c43`, is retained as a
freshness negative: its live safety snapshot/evaluation timestamp was about
`2.815 s` after the response, so every scale failed the unchanged `2.5 s`
command-age limit before IK.
Independent sessions `insertion-safety-20260826T134906Z-52600-c43` and
`insertion-safety-20260826T135750Z-52601-c43` then passed `2/2` at full
translation with rotation held. Their projected maximum joint changes were
`1.634438 mrad` and `1.711919 mrad`; both retained attachment at `0 N` with no
collision, carry `authority: no_actuation`, and are present in verified 13 GB
backups.

One separately invoked action then passed in session
`insertion-trial-20260826T140640Z-52600-c43`. It authenticated the fresh seed
52600 source and frozen proposal fingerprint
`efdf848c120a2e4bba5b5e08f16093eb9b20695940e525906313e5cb1057596f`, selected
full translation with rotation held, and projected a `1.641107 mrad` maximum
joint change. The compensated drive command settled in 19 updates; terminal
joint error was `0.327749 mrad`, below the unchanged `0.500000 mrad` limit.
The realized action tracked the translation direction at cosine `0.999314`,
reduced target translation error from `0.740117 mm` to `0.441414 mm`, and
therefore passed the terminal progress gate at `40.3589%` reduction versus the
required `25%`. Orientation error increased by only `0.180730 mrad`, within
the persisted `1.25 mrad` allowance. The result is exactly `1/1` applied with
retained attachment, `0 N` maximum contact, no collision, no rollback, and a
verified 13 GB recovery copy.

This is the first successful bounded JEPA-WM-driven insertion action under the
corrected drive-only controller contract. It is not autonomous multi-step
insertion: no second action was requested, and the session is terminal.
Authorizing a receding-horizon follow-up requires a new milestone that binds a
fresh observation to the post-action state and proves the same safety,
tracking, progress, attachment, and rollback contracts across that boundary.
Insertion filming and production authority remain false.

The first post-action follow-up milestone was implemented but stopped twice
before capture, inference, execution claim, or motion. Orchestration attempt
`insertion-followup-safety-20260826T145452Z-52600` exposed a compatibility
error while reconstructing the predecessor's v1 no-actuation safety artifact:
the historical reset binding predated evaluated drive-target evidence. Commit
`4811238` keeps that evidence readable while requiring every newly constructed
binding, and every follow-up binding, to carry an exact fresh drive target.
Attempt `insertion-followup-safety-20260826T145813Z-52600` then exposed the
resident upgrade boundary: source reload had cleared the sole in-memory live
runtime before follow-up capture. The subsequent lifecycle repair introduces
a neutral cross-generation handoff and rebuilds the articulation, fixed-joint
plug, collision policy, hand sensor, and connector sensor only after the
resumed physics update, with interlocks before and after wrapper
reconstruction.
Local validation now passes `548` tests with `70` optional skips, and both
independent correctness and standards reviews pass.

Both failed attempts triggered and verified the 13 GB recovery backup. They do
not provide no-actuation safety evidence because no follow-up observation was
captured, and they consumed no action authority. The already-running Isaac
generation nevertheless lost the pre-fix in-memory runtime handle; recreating
it in place would require an unobserved physics update and would violate the
per-update contact/attachment interlock. Therefore no follow-up command was
issued. A valid retry must begin a fresh reset-bound first-action session under
the existing one-action gate and then perform the authenticated follow-up in
the same corrected resident generation. That is a new two-action experiment,
not an automatic continuation of the preserved session, and remains
unauthorized here. Multi-step insertion, filming, and production authority
remain false.

The authorized fresh two-action checkpoint is implemented by commit `1a2832a`.
It derives four session identities from one run, requires action 1 to
reconstruct with passing realized progress before capturing action 2, and
requires the final typed rollout to reconstruct as exactly `2/2` applied.
Blocked or rolled-back action 2 is retained as negative evidence and makes the
workflow fail. Local validation passes `553` tests with `70` optional skips;
both independent correctness and standards reviews pass.

The first orchestration invocation,
`insertion-two-step-20260826T152726Z-52600-c43`, used the abbreviated recording
name `insertion-fresh-held-00`. It failed at capture preflight because that
directory has no manifest. It stopped before observation, inference, execution
claim, or motion and verified the 13 GB recovery backup. The preserved remote
roster identified the exact held-out seed-52600 source as
`contact-insertion-v9-2600-fresh-52600-held-00`.

The corrected run, `insertion-two-step-20260826T153038Z-52600-c43`, captured
fresh action-1 safety session
`insertion-two-step-20260826T153038Z-52600-c43-safety1` at context 43 and
targeted frame 48. The response used frozen proposal fingerprint
`efdf848c120a2e4bba5b5e08f16093eb9b20695940e525906313e5cb1057596f`.
It arrived `0.485050 s` after capture, but the live safety
snapshot/evaluation timestamp was `3.129752 s` after the response and
`3.614802 s` after capture. All six projection scales therefore failed the
unchanged freshness gates with `stale_observation` and
`command_time_invalid`; no scale was selected. The live state retained the
plug at `0 N` contact with no collision, and the safety artifact remains
`authority: no_actuation`.

Action-1 source preflight then failed closed. Its rollout records `0/1`
applied steps and `reset_trial_source_preflight` failure. No action-1 execution
claim or motion occurred; action 2 was never captured, inferred, claimed, or
attempted. The 13 GB recovery backup verified. This is a command-freshness
negative, not a JEPA action-quality, IK, contact, collision, attachment, or
two-step-control result. The next work must reduce the capture/response/safety
handoff latency under the existing freshness limits and requalify the same
fresh boundary. No retry, automatic follow-up, multi-step insertion, filming,
or production authority is granted.

Commit `42eb5e0` closed that freshness handoff without extending either age
limit. Direct insertion safety schema v3 now persists the exact interlocked
live pose, full safety snapshot, active drive target, and refresh timestamp,
then deterministically reauthorizes the already-bound observation and response
against those live inputs. Local validation passes `554` tests with `70`
optional skips, and independent correctness and standards reviews pass.

The separately authorized fresh chain
`insertion-two-step-20260826T163629Z-52600-c43` then cleared action-1 safety at
full translation scale with orientation hold: the safety refresh followed its
JEPA response by about `1.215 s`, projected a maximum `1.641 mrad` joint
change, and retained attachment at `0 N` with no collision. Reset-bound action
session `insertion-two-step-20260826T163629Z-52600-c43-action1` subsequently
terminalized `applied`. It settled in 19 updates with `0.328 mrad` final joint
tracking error, translation tracking cosine `0.9993`, and no contact,
collision, or attachment failure. Realized target translation error fell from
`0.7401 mm` to `0.4419 mm`, a `40.30%` reduction and `0.2983 mm` net progress;
orientation error increased only `0.1800 mrad`, within the unchanged hold
tolerance. The independently reconstructed action-1 rollout is exactly `1/1`
applied with passing realized progress.

The action-2 boundary nevertheless failed before producing its fresh
observation. After the interlocked live-state refresh had paused the timeline,
the wrist-camera helper attempted its retry updates while the timeline remained
paused; every annotator read was empty (`wrist: (0,)`). The typed two-step
rollout therefore records `1/2` applied and `orchestration_failed` at
`followup_capture`. No action-2 observation, inference, execution claim, or
motion occurred, and the 13 GB recovery backup verified. The corrected capture
lifecycle now keeps the timeline live through wrapper refresh and every
force/collision/attachment-observed camera update, reads telemetry only after
the successful RGB frame, and pauses in `finally`. That repair is locally
regression-tested, with `555` tests passing and `70` optional skips, but has not
been requalified live. The run proves one fresh JEPA-WM action with positive
realized progress; it does not yet prove a two-action closed loop. No retry,
third action, multi-step insertion, filming, or production authority is
granted.

The explicitly authorized retry
`insertion-two-step-20260826T170137Z-52600-c43` requalified that capture repair.
Action-1 safety refreshed about `1.624 s` after its JEPA response and passed at
full translation scale with a `1.640 mrad` projected joint change. Session
`insertion-two-step-20260826T170137Z-52600-c43-action1` then applied the action
in 19 settlement updates with `0.328 mrad` final tracking error, translation
cosine `0.9993`, retained attachment, `0 N`, and no collision. Its realized
target error fell from `0.7401 mm` to `0.4417 mm`, a `40.32%` reduction.

The corrected interlocked follow-up capture successfully persisted safety-2
observation `6187058072559216615` and its frame-49 target. JEPA inference also
completed, but all six projection scales failed before IK with
`target_progress_insufficient`; no action-2 execution claim or motion occurred.
The typed rollout therefore remains `1/2` applied with
`followup_source_preflight` failure, and the 13 GB recovery backup verified.
Raw reconstruction shows why the gate was correct: frame 49 was about
`0.492 mm` from the refreshed live pose, outside the `0.1 mm` deadband, while
no scaled proposal reduced that distance by the required `25%`.

The mismatch came from target ownership, not from the safety threshold. The
follow-up selector had measured its `0.5 mm` resolution floor from the recorded
context pose even though action 1 ended off that exact reference trajectory.
From the synchronized live pose, frame 50 is about `0.656 mm` away and the same
half-scale proposal predicts about `37.4%` progress; frame 51 is about
`0.894 mm` away and the full-scale proposal predicts about `49.4%`. Follow-up
target policy now persists a typed `live_observation` origin, selects only after
the interlocked pose is captured, and reconstructs the identical 3–8-frame
horizon search. Initial reset targets keep their established recorded-context
origin. The `0.5 mm` floor, `0.1 mm` deadband, `25%` progress requirement,
orientation hold, projection roster, action bounds, and every safety limit are
unchanged. Local validation passes `556` tests with `70` optional skips and
both independent reviews pass, but this correction has not yet been
requalified live. The result proves repeatable action-1 control and a genuine
fresh second observation; it does not yet prove a second applied action.

Run `insertion-two-step-20260826T182601Z-52600-c43` requalified the live-origin
target selection: action 2 selected frame 50 and retained the half-scale
proposal that the earlier offline reconstruction predicted. It still failed
closed before IK or motion because insertion binding reconstruction occurred
after the exact live refresh; the resulting action-2 command age was
`4.839203 s`. Action 1 remained applied with `40.32%` target-error reduction,
so the terminal rollout was `1/2`. The 13 GB recovery backup verified.

Commit `bf5f58a` moved all disk-backed binding reconstruction and current-policy
validation before execution claim, physics initialization, and synchronized
live refresh. The refreshed observation, response, safety state, pose, and
drive target still pass through the unchanged freshness and safety gates; the
change removes no limit. Its invalid-binding regression proves that a rejected
binding cannot resume physics. Local validation passes `557` tests with `70`
optional skips, and both independent reviews pass.

The subsequent fresh run
`insertion-two-step-20260826T184753Z-52600-c43` is the first independently
reconstructed two-action JEPA-WM closed loop. Safety-1 passed at full
translation scale with a `1.638828 mrad` projected joint change. Action session
`insertion-two-step-20260826T184753Z-52600-c43-action1` applied with a
`9.262 ms` command age, settled in 19 updates, ended at `0.327545 mrad` joint
tracking error, and reduced its target error from `0.740117 mm` to
`0.441824 mm` (`40.30%`).

The interlocked live-origin follow-up selected frame 50. Full scale failed the
unchanged projected-progress gate; half scale passed with a `1.387768 mrad`
projected joint change. Action session
`insertion-two-step-20260826T184753Z-52600-c43-action2` then applied with an
`8.744 ms` command age, settled in 16 updates, ended at `0.280594 mrad` joint
tracking error, and reduced its live target error from `0.637628 mm` to
`0.412927 mm` (`35.24%`). Translation tracking cosines were `0.9993` and
`0.9985`. Both actions retained the plug at `0 N` with no collision; neither
orientation increase exceeded the unchanged `1.25 mrad` hold tolerance.

The terminal typed report records `attempted_steps: 2`, `applied_steps: 2`,
`all_steps_applied: true`, no orchestration failure, and `0.731469 mm` aggregate
translation progress. The 13 GB recovery backup verified. This proves a
genuine bounded two-step closed loop from a fresh reset. It does not by itself
prove arbitrary-length insertion, a seated terminal state, cross-reset
repeatability, filming readiness, or production authority. The next bounded
milestone is a hard-capped multi-step rollout whose lineage, per-step gates,
terminal status, and recovery evidence reconstruct independently; no third
action is inherited from this completed run.

The first hard-capped four-action attempt,
`insertion-demo-20260826T215726Z-52600-c43`, applied actions 1 and 2 before
action-3 capture failed closed with no bounded-horizon target. Its terminal
report records `2/3` applied and `followup_capture` failure; action 3 has no
observation, inference, execution claim, or motion. The synchronized live pose
had advanced beyond the old 3-8-frame target roster: the saved action-2
endpoint placed frame 53 only marginally above the `0.5 mm` floor, and the
subsequent interlocked settling state placed it about `0.278 mm` ahead while
frame 54 remained about `0.660 mm` ahead. Commit `5a4ed94` therefore extends
only the new-policy maximum search horizon from 8 to 12. Selection still
returns the first forward target at least `0.5 mm` away; persisted older
policies retain their exact maximum. No action, projection, IK, velocity,
settlement, progress, force, collision, attachment, or orientation limit was
changed. Local validation passes `569` tests with `70` optional skips, and both
independent correctness and standards reviews pass. The failed run's 13 GB
backup verified.

Fresh held-out run `insertion-demo-20260826T221944Z-52600-c43` then
independently reconstructed all four actions as applied. It selected reference
frames 48, 51, 54, and 56. The first two actions passed at full translation
with rotation held; actions 3 and 4 used the first passing projected scale,
`0.5` translation and `0.125` rotation. Their realized target-error reductions
were respectively `40.32%`, `50.55%`, `51.07%`, and `41.71%`; maximum terminal
joint-tracking errors were `0.328`, `0.655`, `0.452`, and `0.467 mrad`.
The terminal report records `4/4`, `all_steps_applied: true`, no orchestration
failure, maximum command age `8.697 ms`, and `2.841710 mm` aggregate
translation progress. Every action measured `0 N` peak contact, no collision,
and retained attachment.

The same unchanged code and gates then passed held-out seed 52601 in
`insertion-demo-20260826T224600Z-52601-c43`, using reference recording
`contact-insertion-v9-2600-fresh-52600-held-01`. It selected frames 48, 51,
54, and 57. Realized target-error reductions were `42.56%`, `47.38%`,
`45.79%`, and `61.06%`; maximum terminal joint-tracking errors were `0.330`,
`0.635`, `0.533`, and `0.761 mrad`. Its terminal report likewise records
`4/4`, `all_steps_applied: true`, no orchestration failure, maximum command age
`10.226 ms`, and `3.314140 mm` aggregate translation progress. All four
actions retained the plug at `0 N` with no collision. Both final workflows
returned `demo_rollout_applied`, closed their persisted four-step lineage, and
verified the 13 GB recovery backup.

These two held-out `4/4` results meet the repository definition of a
demo-suitable autonomous drive: repeatable, safety-gated JEPA-WM
receding-horizon control from fresh attached resets. They do not establish full
seating, an unknown-start approach/grasp/insertion run, filming authority, or
production readiness. No fifth action or broader rollout authority is implied.

The first integrated grasp-to-insertion canary is retained as a negative, not
as a full-motion claim. A new typed workflow now keeps one contact-aware Isaac
stage live across an eight-action grasp roster, an authenticated proposal
handoff, and a four-action insertion roster; the terminal report requires
exactly `8/8 + 4/4` APPLIED and grants no production authority. Initial runs
failed closed before motion on an invalid pre-grasp context, post-capture
state drift, and incomplete warm-up. Canonical mapping then established that
grasp context `86` corresponds to contact-insertion context `18`, three
reference actions before attachment, and an execution-time live-pose/time
refresh reduced command age to `0.010 s` without changing freshness limits.

Run `grasp-to-insertion-20260827T034816Z-52601` produced genuine JEPA-powered
motion on the held-out contact scene, but did not grasp. Action 1 passed
freshness, IK (`0.00024 mm` position, `0.502 mrad` orientation), joint,
workspace, velocity, force, and collision gates. It realized `14.738 mm` of a
`17.100 mm` translation command at `0 N` with no collision, but missed the
unchanged Cartesian tracking limit by recording `2.411 mm` error, left the
connector unattached, and its interlocked rollback remained `3.695 mrad` from
the captured target versus the unchanged `1.000 mrad` requirement. The exact
grasp proposal also repeatedly commanded the gripper toward opening at the
contact-scene attachment frame. Therefore the run ended `0/8`; insertion was
never inferred or attempted, and a verified `14 GB` recovery copy was taken.
The next gate is a dedicated contact-grasp proposal trained only on the 12
existing TRAIN contact recordings (contexts `18-25`), followed by a new
disjoint two-seed offline gate before another integrated live action. No
tracking, rollback, force, collision, velocity, or freshness threshold is
opened by this negative.

The contact-aware corpus refresh for that gate now uses a 284-frame drive-only
recording: 48 samples each for pre-grasp, open-grasp approach, retreat, and
alignment; 16 for gripper closure; 64 for insertion; and the existing bounded
holds. The first canonical TRAIN recording,
`contact-insertion-v10-drive-slow-2600-train-00`, completed at `0 N` with
`8.402 mrad` maximum arm error and `1.936 mm` maximum gripper error. The first
seed-2601 attempt stopped during insertion when connector force crossed the
unchanged `2 N` gate at `2.180 N`; the partial recording and job are preserved
under timestamp `20260828T201133Z` in the recovery quarantine.

An exact no-camera fixed-step reproduction localized the negative to insertion
sample 39 of 64: connector force rose from `0.114 N` to `2.180 N` within that
single interval while the hand sensor remained at `0 N`. Extending insertion
to 96 samples did not remove the obstruction. The scripted waypoint solver was
still accepting the first inverse-kinematics result inside a `3 mm` Cartesian
position tolerance, which left the failing seed roughly `0.55 mm` laterally
offset before socket contact. Scripted waypoints now use the same bounded
nine-start local-branch search as live control, with `0.1 mm` position and
`1 mrad` orientation solver tolerances. No force, tracking, seating, velocity,
or task tolerance changed.

The corrected full camera-backed seed-2601 artifact reused the same recording
identity and completed all 284 frames. Independent reconstruction reports
`0 N` maximum force, `7.644 mrad` maximum arm error, `1.936 mm` maximum gripper
error, `0.0021 mm` seating-depth error, `0.0715 mm` lateral error, `0.437 mrad`
orientation error, attachment at frame 113, and four seated observations. A
separate held-out canary `contact-insertion-v10-drive-slow-72600-held-00` also
passes the new contract. The resumable workflow subsequently completed all 12
TRAIN recordings (seeds `2600-2611`) and both canonical HELD_OUT recordings
(seeds `12600-12601`). All 15 exact roster entries, including the development
canary, independently validate as 284-frame contact-aware artifacts with `0 N`
maximum connector force and four seated observations. Their manifests, the
base checkpoint, and the control adapter match the verified recovery copy.

### Contact-insertion action-conditioning checkpoint

A frozen offline experiment then tested whether the retained-retreat failure
was caused by sampling imbalance, insufficient nonlinear action capacity, or
conflicting physical regimes. Configuration
`a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14`
fit only the exact 12 TRAIN recordings with 2,016 balanced updates. It compared
the existing linear control A, balanced global-linear B, a small
`7 -> 32 -> 1024` nonlinear residual C, and diagnostic-only oracle-routed
retained/post residuals D. The DINOv3 encoder and 12-block JEPA-WM predictor
remained frozen.

All B/C/D artifacts authenticated after atomic writes. Their final losses were
`0.013490`, `0.007792`, and `0.011364`; artifact fingerprints are respectively
`7bb9b546...a0754`, `65272ab9...a9908`, and `243c80ea...0610a`. The sealed
seed-72600 canary was then evaluated exactly once per treatment:

| Treatment | Overall win | Retained win | Post win | Mean improvement | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0.702381 | 0.056604 | 1.000000 | 0.000052213 | retreat failure reproduced |
| B | 0.297619 | 0.943396 | 0.000000 | -0.000084014 | learned retreat, reversed post |
| C | 0.267857 | 0.849057 | 0.000000 | -0.000001553 | learned retreat, reversed post |
| D | 0.982143 | 0.943396 | 1.000000 | 0.001082767 | main regimes pass; diagnostic only |

No promotion-eligible artifact passed the frozen gate. D passed the unchanged
repository action-control gate and every main motion-regime threshold, but
failed the stronger every-segment requirement on the four-frame
`retreat_hold` segment (`0.25` wins and negative mean). Its oracle phase router
also makes it non-deployable. Canonical held-out seeds `12600-12601` therefore
remain unopened for model evaluation; no threshold was weakened and no live
action followed.

The result is a scoped routing/capacity blocker, not evidence that DINOv3 is
insufficient. On the canary, all 48 retreat rollout actions have negative mean
base-frame X, while all 48 alignment and all 64 insertion rollout actions have
positive mean X. The split-safe TRAIN roster shows the same invariant across
all 12 recordings: `576/576` retreat rollouts are negative-X, while `576/576`
alignment and `768/768` insertion rollouts are positive-X. Hold windows contain
only measured drift and mix signs, so sign alone is insufficient. The next
bounded offline experiment is a runtime-command-derived three-way router:
negative-X motion and positive-X motion receive small residual experts, while
neutral hold commands retain the shared base path. Its deadband must be
selected from TRAIN command statistics before freeze, it must not depend on
scripted phase or context index, and it requires a fresh noncanonical scripted
canary because seed 72600 informed the design. Same-reset causal captures are
deferred unless that observable router fails. This checkpoint grants no new
JEPA action, filming, hardware, or production authority.

### Runtime-command routing checkpoint

The frozen three-path follow-up used configuration
`98fc2af503919d52a3853d3181bf007d56360136e5c1d27cd1a08a4db18bf66d`.
It retained the authenticated global action map bitwise, added one
zero-initialized linear residual for active negative-X horizons and one for
active positive-X horizons, and sent neutral or active non-X commands through
the unchanged base. Runtime routing used only the complete candidate 7D
command; phase, context index, and seed were unavailable. The exact TRAIN
roster contained 564 negative-X, 1,296 positive-X, 106 neutral-base, and 50
active-non-X-base rollouts. Training drew 1,008 deterministic updates from
each learned route and no base-route updates.

Fresh noncanonical canary
`contact-insertion-v10-drive-slow-72601-held-00` was captured once at seed
72601. It independently validates as 284 contact-aware frames with `0 N`
maximum force, `7.728 mrad` maximum arm tracking error, `1.936 mm` maximum
gripper error, four seated observations, `0.0101 mm` depth error, `0.1261 mm`
lateral error, and `0.905 mrad` orientation error. Its selected insertion
inputs bind to fingerprint
`056b08827e26b1925a3ca4d1cd96f6ed6ea0a879d6231f9cab17c1ad29b8505e`.

The one permitted router trained for 2,016 updates without changing the base
map. Loss fell from `0.021449` to `0.013905` (minimum `0.008303`); artifact
fingerprint is
`45326210f5a47f74a9008670e9bf0be03b3ef40955b3c9af79017588d9b79c30`.
The control and router were then evaluated once on the same canary bytes:

| Treatment | Overall win | Retained win | Post win | Retreat win | Retreat signed order | Align signed order | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control A | 0.702381 | 0.056604 | 1.000000 | 0.000000 | 0.000000 | 1.000000 | retained retreat fails |
| router R | 0.988095 | 0.962264 | 1.000000 | 0.979167 | 0.000000 | 0.020833 | signed counterfactual gate fails |

R therefore solved the main retreat-versus-forward win-rate conflict and made
every semantic segment's mean improvement positive, but it did not learn the
required direction ordering. On retreat, recorded action beat both X-zero and
X-opposed in `47/48` cases, yet X-zero never beat X-opposed. On alignment and
insertion, recorded action always beat X-zero but beat X-opposed in only
`1/48` and `4/64` cases. The training objective imposed separate margins from
recorded to X-zero and recorded to X-opposed; it did not directly impose the
gate's `recorded < X-zero < X-opposed` ordering. Candidate-dependent routing
also sends an opposed-X counterfactual through the opposite residual expert,
so the counterfactual changes both the command and the learned map. This is a
training-objective/architecture negative, not a representation-insufficiency
result and not an Isaac/controller regression.

The authenticated summary fingerprint is
`3469d4e73cbed26d63aad4751bf0ebf88b31d651b34feb7a5a678d02a2dc93a4`;
it records `router_failed`, selects no artifact, and keeps canonical offline
and live-action authority false. Canonical seeds `12600-12601` remained sealed
from model evaluation. All terminal artifacts match the verified 16 GB
recovery copy. Per the stopping rule, no canary retuning, second training run,
canonical evaluation, JEPA action, filming, hardware, or production step
followed. Diagnosis and any revised counterfactual objective belong to a new
milestone.

### Observed-context routing checkpoint

The next frozen TRAIN-only experiment removed candidate-dependent switching.
It derived one continuous route from the previous realized 7D action, then
held that route fixed while scoring the recorded, zero, X-zero, X-opposed,
mismatched, and mined candidates. The authenticated configuration fingerprint
is `2b57e748...3abf6`. Only two zero-initialized linear residuals trained; the
global action map remained bitwise unchanged. The observed route agreed with
the future recorded command on `1,895/2,016` TRAIN transitions, with 576
negative-X, 1,283 positive-X, and 157 base observations.

The one permitted artifact trained for 2,016 updates. Loss fell from
`0.018564` to `0.010682` (minimum `0.007192`), and its fingerprint is
`27d00e129...b30912`. The complete frozen TRAIN gate then produced:

| Slice | Recorded win | Signed order | Mean improvement |
| --- | ---: | ---: | ---: |
| all TRAIN | 0.988095 | 0.988095 | 0.001372884 |
| retained | 0.962264 | 0.962264 | 0.001469931 |
| post | 1.000000 | 1.000000 | 0.001328158 |
| grasp attach | 0.000000 | 0.000000 | -0.000091481 |
| retreat | 1.000000 | 1.000000 | 0.001623500 |
| align | 1.000000 | 1.000000 | 0.002226017 |
| insert | 1.000000 | 1.000000 | 0.000716770 |

The experiment failed two unchanged conjunctive requirements. `grasp_attach`
lost to zero on all 12 TRAIN cases and had negative mean improvement. The
positive-X residual/base embedding ratio also reached `0.175160`, above the
frozen `0.15` ceiling; the negative-X maximum was `0.119134`. Every aggregate,
retained, post, and main-motion ordering threshold otherwise passed. In
particular, candidate-independent routing changed retreat, alignment, and
insertion signed order from the prior router's failure to `1.0` on each
segment.

This is an attachment-boundary and residual-magnitude negative, not evidence
of representation insufficiency. The terminal evaluation fingerprint is
`493f42e4...e5965`; it and the trained artifact/report exactly match the
verified 16 GiB recovery copy. Per the frozen stopping rule, no retraining,
fresh canary, canonical evaluation, JEPA action, filming, hardware, or
production step followed. Full evidence is in
[`RESULT.md`](.scratch/jepa-observed-context-routing-v1/RESULT.md). Any
residual constraint or attachment-boundary change belongs to a new milestone.

### Physical-state routing redesign

The next router design separates demonstrated intent labels from the failed
visual-context classifier. Declared `retreat_hold`, `align_hold`, and
`seated_hold` windows are semantic holds even when recorded joint telemetry
contains sub-deadband drift. Runtime routing uses a versioned 26-value,
task-relative physical observation: plug/socket, end-effector/socket, and
gripper/socket geometry; plug orientation; gripper, tracking, force, and
attachment telemetry; plus the previous realized 7D action. It cannot inspect
the visual latent, candidate or future action, phase, context index, or seed.

An architecture-selection diagnostic over all 2,016 exact TRAIN transitions
selected the smallest passing nonlinear router, `26 -> 64 -> 64 -> 4`. In
leave-one-recording-out evaluation it reached `0.987103` accuracy, `0.992908`
retreat recall, `0.989780` advance recall, `1.0` grasp-attachment accuracy,
and `0.028274` fail-closed fraction, with zero retreat/advance activations in
all three semantic hold segments. The label roster is 142 hold, 564 retreat,
1,272 advance, and 38 active-other transitions.

The production action-conditioning seam now authenticates that feature order,
uses a sign-invariant relative quaternion, refuses inference until fitted
normalization is serialized, holds one route fixed across every candidate,
preserves the original action map exactly for hold/active-other, and enforces
the `0.15` residual/base norm bound by construction. The one-shot authenticated
probe is frozen in
[`PLAN.md`](.scratch/jepa-physical-state-routing-v2/PLAN.md); it has not yet
authorized residual training, held-out access, live JEPA action, or filming.

After insertion control clears its offline and live safety gates, the requested
lab stopping point is one reconstructible end-to-end Isaac run from a
predeclared, bounded held-out unknown start. That run must not replay a recorded
motion prefix or use manually staged task phases: it must approach, grasp,
retain, align, insert, and hold through JEPA-controlled, safety-gated actions
from ordinary synchronized camera observations and robot proprioception. One
valid run is sufficient at this stage; multi-seed repeatability, autonomous
recovery, real-hardware transfer, and production readiness are intentionally
deferred. See milestone 20 in [`docs/control-loop.md`](docs/control-loop.md) for
the exact acceptance contract.

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
