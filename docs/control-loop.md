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
worker-to-controller boundary remains unwired until the promoted proposal is
wrapped in simulator safety checks.

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

### Promoted inverse-action proposal

The bounded CEM and empirical-prior implementations are reproducible, but CEM
refinement lowered model energy without improving held-out action direction. It
stays diagnostic-only. The promoted inverse head consumes frozen spatial JEPA
features, current base-frame DROID pose, and the previous DROID action. Twelve
training seeds produced 792 rollouts. After a four-frame warm-up, both unseen
seeds passed 62/62 operational rollouts, for 124/124 aggregate, mean cosine
`0.975956`, and mean sequence MSE `0.000212981`. The persisted readiness artifact
explicitly scopes this to simulator-only inverse action.

The first simulator-only execution boundary is now implemented. Isaac and the
resident JEPA-WM worker exchange `quantis.jepa_wm_control.v1` JSON over a Unix
socket and shared EBS-backed session directory. Isaac replays a deterministic
exploration prefix through a complete segment boundary (at least four frames),
sends the current wrist frame, reference goal frame, base-frame DROID
pose, and previous action, then consumes only the returned first action. Every
session is single-use and persists its immutable request/response and measured
outcome.

Before actuation, the bridge proves that the goal recording has the same unseen
seed and DROID/wrist-camera contract as the live variant. The request is bound
to a unique session-derived nonce, an exact promoted-checkpoint path, and an
ordered response timestamp. The gate then requires final simulator-only
freshness checks—3.0 seconds from the synchronized observation and 2.5 seconds
from the model response—an established observation context, bounded 7D action, workspace membership,
Franka joint position/velocity limits, no live hand contact, and at most 2 N.
The executor tries bounded translation/rotation/gripper scale profiles, applies
only the first profile whose pose, IK branch, joint velocity, and remaining
safety evidence pass, and claims the session before motion, preventing replay
after interruption. Uniform quarter- and one-eighth-scale actions remain
fallbacks. After actuation it requires measured joint tracking plus Cartesian
translation/rotation direction and error; failures roll back.
Sessions `step-20260823T153339Z-11400` and
`step-20260823T152202Z-11401` passed on the two unseen seeds at 1.883 s and
0.948 s observation age respectively, with no contact.

Repeated receding-horizon execution is also proven in narrow free space.
Rollout `rollout-20260823T155348Z-11401` generated three distinct fresh
observations, inferred three native proposals in the resident worker, consumed
only each first action, measured each result, and fed the new pose/frame/action
back into the next request. All three actions passed with 0 N contact; mean
observation age was `1.322 s`, and translation/rotation goal error improved by `0.519 mm` and
`0.001793 rad`. The persisted `quantis.jepa_wm_control_rollout.v1` report
separates requested, attempted, and applied steps and validates chain, goal,
proposal, observation-ID, capture-order, warm-up, and previous-action
provenance. Follow-up capture rejects a live articulation that no longer matches
the preceding measured pose/joints and samples RGB plus state at one update
boundary. The rollout exit finalizer persists incomplete orchestration attempts
as terminal evidence. This does not validate cable contact or insertion.

The live bridge now retains one bound articulation/runtime across follow-up
calls, avoiding stale paused-physics tensors and repeated setup. Held-out
rollout `rollout-20260824T032653Z-11401` selected the complete exploration
boundary at context `44`, applied three JEPA proposals with 0 N contact, moved
the end effector `15.440 mm`, and reduced translation-to-target error by
`11.127 mm`. The plug was not attached. This is motion-rich closed-loop
evidence, not a successful grasp, cable insertion, or publishable task demo.

### Next control milestone

1. [x] Keep Isaac and the resident JEPA worker in separate processes and exchange a
   versioned observation/action envelope.
2. [x] Require a synchronized seeded observation prefix of at least four frames
   before inference and reject stale or out-of-order observation IDs.
3. [x] Enforce workspace, per-step Cartesian, gripper, joint, collision, and force
   limits before the first proposed action reaches the articulation.
4. [x] Apply only the first bounded action in Isaac, observe again, and replan.
   A three-step held-out canary passed with fresh single-use requests and 0 N
contact.
5. [x] Run bounded proposal-centered candidate search in shadow mode and record
   predicted energy, direction, and no-actuation Isaac safety. Held-out rollout
   `rollout-20260823T203534Z-11401` passed all three search and safety gates with
   mean latent-energy improvement `0.000221416`.
6. [x] Compare realized goal progress against direct-proposal, zero, and
   scripted baselines before granting the candidate planner command authority.
   The strict three-step AWS comparison
   `baseline-proof-20260823-11401` measured actual reset-identical rollouts, not
   inferred counterfactuals. Direct beat zero only on gripper closure and was
   worse on translation and rotation, so the gate failed and candidate
   authority remains false.
7. [x] Execute the shadow CEM winner only as an explicitly isolated
   reset-identical experimental policy and compare its realized outcome against
   the three established policies. Candidate trial
   `candidate-proof-20260823T213129Z-11401` applied safely with 0 N contact but
   worsened translation/rotation error and failed every direct-comparison axis;
   production authority remains false.
8. [x] Calibrate predicted action response from realized trials and rerank
   latent energy with a continuous per-axis task-space regression penalty.
   Candidate `candidate-proof-20260823T220355Z-11401` beat the direct proposal
   on all axes and reversed the prior translation/rotation regression, but
   still trailed zero on translation and missed scripted translation tolerance.
9. [x] Add a positive translation margin, fit on disjoint whole-seed evidence,
   and repeat one fixed worker/search identity across whole held-out seeds.
   The seed-237 worker passed the strict `2/2` isolated-candidate readiness gate;
   production authority remains false.
10. [x] Keep one live Isaac runtime resident across follow-up actions and select
    a deterministic motion-rich context boundary. Rollout
    `rollout-20260824T032653Z-11401` moved `15.440 mm` and made `11.127 mm`
    target progress at 0 N, without grasping the plug.
11. [ ] Complete a JEPA-controlled reach-and-grasp before filming again. The
    hand must enter a defined pre-grasp region, close on the rigid connector,
    persist a bound `plug_attached` transition, and retain the object through a
    lift/hold while tracking, collision, and force gates pass. Validate the raw
    multi-step sessions on two whole held-out seeds against zero and scripted
    baselines. Free-space displacement or gripper closure alone does not count;
    insertion follows only after this gate passes.

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
11. [x] Train a bounded inverse-action proposal and validate it on two complete
    held-out seeds with strict provenance and an explicit warm-up boundary.
12. [x] Execute only the first proposal through the simulator safety gate and
    capture the measured post-action observation on two unseen seeds.
13. [x] Feed each accepted post-action observation back to the resident worker,
    replan, and compare bounded multi-step proposal, zero, and scripted rollouts
    before approaching cable contact. Three-step reset-identical trials now
    establish the comparison, and its failed direct baseline gate prevents
    premature promotion.
14. [x] Establish repeatable realized shadow-candidate improvement on whole
    held-out seeds. A `0.5 mm` manifest-bound translation gate now transfers in
    both whole-seed directions and beats zero/direct on every axis.
    Calibration checkpoints retain reconstructible proposed/realized trials,
    require per-axis directional coverage, and are loaded through one manifest
    that binds proposal, adapter, and calibration. Target frame/pose identity is
    revalidated against reference telemetry before every shadow search. Every
    unresolved axis must also clear a persisted minimum predicted reduction
    (`0.1 mm` translation, `0.001 rad` rotation, `0.005` gripper closedness),
    with before/after/required progress exposed as reconstructible evidence.
    Calibration on seed 11400 and evaluation on 11401 passed the full realized
    gate with `1.531 mm` translation progress. Calibration on 11401 and
    evaluation on 11400 realized `2.188 mm` but missed scripted translation
    tolerance by `0.057 mm`. This is repeatable improvement, but not yet two
    strict passes; production authority remains false. These reciprocal trials
    are two-fold cross-validation, not a globally held-out readiness set. The
    readiness gate rejects any calibration seed appearing anywhere in the
    evaluation-seed union; the next experiment must calibrate on training-only
    seeds and then pass both 11400 and 11401. A dedicated
    `calibration_collection` policy now permits only training references while
    retaining live safety/tracking. Seed-1400 calibration passed shadow search
    on held-out 11401, while 11400 missed rotation by `0.000011 rad` and had
    slightly worse latent energy. Search robustness is now the active blocker.
    The fitter and readiness reconstruction revalidate every calibration trial's
    policy, training manifest, seed, realized action, tracking, collision, and
    contact evidence before it can count.
    Worker manifests now also own the CEM seed, iterations, samples, and elites,
    and readiness rejects results whose full worker/search identities differ.
    One fixed seed-237, `5 × 128 / 12 elites` worker passed held-out 11400 and
    11401. Its isolated candidates realized `2.843 mm` and `2.659 mm`
    translation progress, beat zero/direct and reached scripted tolerance on
    every axis, and passed tracking at 0 N without collision. The strict
    readiness summary passes `2/2` globally disjoint held-out seeds against
    calibration seed 1400; production authority remains false.
15. [ ] Demonstrate a meaningful JEPA-controlled reach-and-grasp before
    producing another video. Require a defined pre-grasp approach, verified
    rigid-connector attachment, retained lift/hold, safe force/collision and
    tracking evidence, and two whole held-out seeds compared with zero and
    scripted baselines. Free-space displacement and gripper closure are not
    task success. Cable insertion is the subsequent milestone. The data path is
    now active: scripted held-out artifact `grasp-20260824-held-11401-v1`
    validates 102 true-4-FPS frames, acquisition at frame 89, 13 retained
    observations, and `99.9997 mm` attached displacement. This is training
    evidence only; JEPA has not yet passed the live task gate.
