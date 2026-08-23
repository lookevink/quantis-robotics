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
./ops/aws.sh jepa-wm-eval RECORDING wrist 7 8 1
```

`trajectory-20260823T025416Z` proved the recording and evaluation path but failed
the action-validity gate. Across eight bounded, nonzero actions, recorded actions
were worse than zero on average for both wrist and presentation cameras. The
worker-to-controller boundary therefore remains unwired. Validate against native
DROID samples and adapt the model to Isaac imagery/actions before implementing
CEM or sending any sampled action to the simulator articulation.

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

Each v2 action trajectory contains:

- ordered RGB frames at a fixed capture rate;
- one 7D action per transition;
- joint/end-effector state sampled at the same point;
- task and camera metadata;
- one observable-stage label per synchronized step;
- motion-only observations at the model's 4 FPS cadence;
- success/failure outcome.

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
9. [ ] Only then allow the action-conditioned planner to propose bounded
   simulated actions.
