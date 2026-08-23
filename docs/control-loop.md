# JEPA control integration

## What can be loaded immediately

The AWS workflow now loads the official Hugging Face
`facebook/vjepa2-vitl-fpc64-256` encoder and converts either recorded wrist or
presentation-camera observations into a normalized latent vector. Demo capture
guarantees the model's 64-frame input window, keeps both views synchronized with
robot state, and stores the embedding beside the persistent recording. The live
CUDA proof produced a 1,024-dimensional wrist-camera vector from
`demo-20260822T194745Z`.

The offline stage pipeline now uses that vector to select a stage and measure
progress toward prerecorded goal-image windows:

1. `approach cable`
2. `grasped cable`
3. `aligned with socket`
4. `plug seated`

Two complete deterministic AWS runs produced 4/4 correct held-out wrist-camera
stage predictions. Their winning cosine margins ranged from `0.0083` to `0.0341`.
The result proves nominal separation and caching, but the small margins do not
justify online advancement until camera/geometry perturbations and failure cases
are evaluated.

The intended resulting stage/subgoal is executed by Isaac Sim's motion generation controller as an online visual feedback loop, not open-loop playback.

## Why action-conditioned JEPA-WM is not a policy yet

The installed DROID JEPA-WM checkpoint is useful but is not a drop-in robot
policy. It predicts future DINOv3 latents conditioned on seven-dimensional
DROID pose deltas. Control additionally requires:

- matching its action and proprioception normalization;
- sampling candidate actions;
- rolling the predictor forward;
- scoring rollouts against a goal latent;
- cross-entropy-method refinement;
- applying only the first action and replanning;
- enforcing workspace, velocity, collision, and force limits.

The repository now tests the official checkpoint through:

```bash
./ops/aws.sh demo-record-actions
./ops/aws.sh jepa-wm-eval RECORDING wrist 0 20 1
./ops/aws.sh jepa-wm-adapt TRAINING_RECORDING wrist 100
./ops/aws.sh jepa-wm-eval-adapted HELD_OUT_RECORDING wrist 0 20 1
```

The action contract was checked against the pinned upstream DROID loader. Each
v3 recording stores robot-base-frame poses at 4 FPS, derives the same
translation/relative-Euler/closedness deltas, and evaluates the planner's native
one-frame/three-action rollout. The released model failed on both camera views.

A lightweight adapter can now calibrate only the action encoder's weight matrix
while the DINOv3 encoder, predictor, and zero-action bias remain frozen. It is
small enough to train beside Isaac on the L4 and persists on EBS. The first
nominal held-out run improved from a 10% base-model win rate to 50% after
adaptation, which was still below the gate. The later whole-seed domain
experiment described below reaches 97.5% across two unseen seeds. The
worker-to-controller boundary nevertheless remains unwired until candidate
search and safety checks are validated in simulation.

### Reproducible domain adaptation proof

Run the complete capture, split, adaptation, evaluation, aggregation, and
backup workflow with:

```bash
./ops/aws.sh jepa-wm-milestone 4 2 500 1400
```

The command records four training seeds (`1400`–`1403`) and two complete
held-out seeds (`11400`, `11401`). Each recording has 69 synchronized 512x512
wrist frames: 17 segments centered on a seeded offset from the IK-verified ready
pose, two excitations per arm joint, alternating gripper targets, explicit
stationary/failed-grasp/recovery outcomes, and seeded wrist-camera, plug/socket,
receptacle-scale, and lighting variants. Actual simulation timestamps verify an
exact 0.25-second interval between every sample. No trajectory contributes
frames to both sides of the split.

Experiment `domain-20260823T113209Z-1400` fitted only the 7,168 action-encoder
weights for 500 optimizer steps over 264 native one-frame/three-action
rollouts. Batch size was 2, learning rate `1e-3`, optimization seed 234,
contrastive weight 1.0, and margin `1e-3`. It took 338 seconds to train and
peaked at 7.772 GiB allocated by the JEPA process while Isaac remained running.
Its checkpoint and JSON training report are stored together at:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth
/home/ubuntu/docker/jepa-wm/checkpoints/quantis_isaac_wrist_action_adapter.pth.json
```

Forty rollouts from each held-out seed were scored. Seed `11400` passed at 97.5%
wins and `+0.0010193158` mean improvement over zero; seed `11401` also passed at
97.5% and `+0.0010888389`. The aggregate was 78/80 wins (`97.5%`) and
`+0.0010540774`, so both individual and aggregate gates passed. The persisted
summary is:

```text
/home/ubuntu/docker/jepa-wm/checkpoints/experiments/domain-20260823T113209Z-1400.json
```

Both the live artifacts and a checksum-verified recovery copy are preserved by
`./ops/aws.sh backup-state`; see the persistence inventory in the README.

### Next control milestone

1. Add bounded candidate generation and CEM refinement over the JEPA-WM
   three-action horizon.
2. Score candidates against explicit subgoal latents and verify that the
   selected first action improves a held-out simulated scene more often than
   zero, random, and scripted baselines.
3. Reject stale observations and enforce workspace, velocity, joint, collision,
   and force limits before an action reaches the articulation.
4. Apply only the first bounded action, observe again, and replan. Pause on low
   confidence or contradictory stage/goal evidence.
5. Add stationary, failed-contact, recovery, and broader geometry variants
   before treating the offline gate as evidence beyond this narrow Isaac domain.

## Recommended process boundary

Keep Isaac Sim and JEPA in different processes:

```text
Isaac Sim                          JEPA worker
---------                          -----------
RGB frame window  ───────────────► encoder
robot state                       stage/goal score
candidate action metadata         optional AC rollout
                  ◄────────────── subgoal or bounded 7D delta
motion controller
safety limits
```

The first interface should return a discrete subgoal and confidence:

```json
{
  "observation_id": 1234,
  "stage": "cable_grasped",
  "similarity": 0.99,
  "margin": 0.02
}
```

The implemented `StageGate` consumes this contract, rejects stale IDs, pauses on
unknown or unexpected stages, and requires consecutive confident observations.
The standalone `jepa.online_worker` process consumes JSONL requests containing
64 ordered frame paths, keeps V-JEPA loaded, and emits this response with a fresh
monotonic observation ID. The worker-to-Isaac transport and controller call site
remain deliberately unwired until perturbed evaluation establishes usable
thresholds.

Later, the action-conditioned interface can return a bounded 7D command:

```json
{
  "observation_id": 1234,
  "action": {
    "delta_xyz": [0.002, 0.0, 0.0],
    "delta_rpy": [0.0, 0.0, 0.0],
    "gripper": 1.0
  },
  "predicted_goal_energy": 0.18
}
```

## Dataset contract

Each v3 action trajectory contains:

- ordered RGB frames at a fixed capture rate;
- one 7D action per transition;
- joint/end-effector state sampled at the same point;
- task and camera metadata;
- one observable-stage label per synchronized step;
- motion-only observations at the model's 4 FPS cadence;
- success/failure outcome.

End-effector pose and action axes are expressed in the Franka base frame. This
matters because the demo's Franka root is rotated relative to the Isaac world;
world-space deltas silently invert the model's X/Y semantics.

WebRTC is only for viewing. Capture directly from Replicator and the controller because WebRTC compression and network latency break action/frame alignment.

## Milestones

1. [x] Capture smoke test succeeds on AWS EC2.
2. [x] Scripted Franka success runs produce at least 64 synchronized wrist and
   presentation-camera observations per observable stage with robot state.
3. [x] Frozen V-JEPA encodes a real wrist-camera run on the AWS GPU and persists
   its normalized latent vector.
4. [x] Cache four stage windows and classify a separate deterministic run with
   frozen V-JEPA cosine centroids.
5. [x] Implement the stale/unknown/confidence/consecutive-confirmation gate.
6. [ ] Record failed and camera/geometry-perturbed runs and calibrate thresholds.
7. [ ] Close validated stage predictions around Isaac's bounded motion controller.
8. [x] Evaluate JEPA-WM action conditioning offline against a synchronized
   simulated trajectory; the first recorded-action-versus-zero gate failed.
9. [x] Match the native DROID base-frame and one-frame/three-action contract,
   and validate a persistent frozen-backbone action adapter on a held-out run.
10. [x] Collect varied Isaac dynamics data and reach at least a 75% held-out
    recorded-action win rate with positive mean improvement over zero.
11. [ ] Allow the action-conditioned planner to propose bounded candidate
   actions, validate them against baselines, and execute only through the
   simulator safety gate.
