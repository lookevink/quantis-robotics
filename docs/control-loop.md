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
11. [x] Complete a JEPA-controlled reach-and-grasp before filming again. The
    hand must enter a defined pre-grasp region, close on the rigid connector,
    persist a bound `plug_attached` transition, and retain the object through a
    lift/hold while tracking, collision, and force gates pass. Validate the raw
    multi-step sessions on two whole held-out seeds against zero and scripted
    baselines. Free-space displacement or gripper closure alone does not count;
    insertion follows only after this gate passes. The strict held-out rollout
    `rollout-20260824T105547Z-12400` is the first validated partial success: JEPA
    acquired the connector on action one, retained attachment across four safe
    observations, and moved it `5.615 mm` at 0 N. It remains below the 20 mm
    retained-motion threshold with `insufficient_lift`. Eight-step follow-ups
    fingerprint-bound rollouts `rollout-20260824T124729Z-12400` and
    `rollout-20260824T132311Z-12401` then passed the task gate on both held-out
    seeds: `8/8` actions applied per seed, attachment retained for all eight
    observations, `55.258 mm`/`58.508 mm` retained displacement, 0 N contact,
    and no collision. Reset-identical zero trials did not attach and scripted
    trials passed the same task. The exact-proposal two-seed readiness artifact
    passes and permits filming this bounded task; production command authority
    remains false and insertion is still the next milestone.

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
15. [x] Demonstrate a meaningful JEPA-controlled reach-and-grasp before
    producing another video. Require a defined pre-grasp approach, verified
    rigid-connector attachment, retained lift/hold, safe force/collision and
    tracking evidence, and two whole held-out seeds compared with zero and
    scripted baselines. Free-space displacement and gripper closure are not
    task success. Cable insertion is the subsequent milestone. The data path is
    now active: scripted held-out artifact `grasp-20260824-held-11401-v1`
    validates 102 true-4-FPS frames, acquisition at frame 89, 13 retained
    observations, and `99.9997 mm` attached displacement. This is training
    evidence only; JEPA has not yet passed the live task gate. Live context 86
    now reconstructs the exact held-out prefix three actions before acquisition,
    advances the task target one reference frame per receding-horizon step, and
    binds connector reset state into baseline equivalence. Training variation is
    now 20 seeds while held-out seeds 12400/12401 remain fixed. A typed
    current-to-goal DROID delta improved the best task head to `90%` gate and
    `92.6%` active-direction passes, but contexts 76/83 and the attached hold
    still fail the strict offline gate. This checkpoint remains negative and has
    no live or filming authority. Explicit task progress reached held-out active
    cosines `0.844`/`0.871` but remained at `90%` gate on both seeds. Active
    cosine now excludes zero-vector stationary labels while their separate hold
    gate remains mandatory. A 20-recording grasp action adapter then exposed
    latent-objective exploitation: fixed bounded CEM reached only `0.434` cosine
    and `37.5%` gate despite lowering energy below recorded actions on all eight
    diagnostic contexts. Uniformly sampled mismatched rollout negatives improved the
    controlled comparison to `0.489`/`50%`, still a strict negative. Candidate-
    aware online mining over four bounded local candidates moved the same
    diagnostic to `0.497` active cosine and `62.5%` gate. The unrefined proposal
    itself scores `0.703`/`75%` there, so the remaining planner failure is weak
    proposal-centered regularization. A split-validated TRAIN calibration chose
    a `1e-2` proposal prior after it restored `0.986` cosine and `100%` passes;
    one frozen held-out run then preserved the proposal at `0.703`/`75%`. This
    removes CEM degradation but does not improve the proposal. The promoted
    gripper-timing head subsequently cleared the strict offline gate and its
    first strict live rerun acquired and retained the connector for four
    observations, moving it `5.615 mm` without force or collision. The task
    still fails only `insufficient_lift`; no live result yet clears the 20 mm
    retained-motion threshold or the required second held-out seed, so filming
    authority remained false at that checkpoint. The subsequent eight-step
    fingerprint-bound direct rollouts on held-out seeds 12400/12401 both passed
    reach-and-grasp at `55.258 mm`/`58.508 mm`, while reset-identical zero
    controls failed to attach and scripted controls passed. The persisted
    two-seed task readiness
    gate now passes; filming this grasp is authorized, but production command
    authority and cable-insertion capability remain false.
16. [x] Produce the readiness-bound reach-and-grasp visualization. Reload the
    persisted two-seed gate, replay one complete eight-action direct rollout,
    verify every rendered arm/gripper target plus contact and collision state,
    and encode the 1080p wrist-primary dashboard. Treat source task evidence and
    replay telemetry as separate contracts; the video remains visualization-only.
    `grasp-demo-20260824-seed12401-v1` contains 78 frames at 12 FPS and binds
    source rollout `rollout-20260824T132311Z-12401` to proposal fingerprint
    `6aa4b94b610bfd8fff07e9356e932574a11342533d55c69b06c1c2ab20e9fd2d`.
    Replay verification passed with `0.633 mrad` maximum arm error,
    `0.0011 mm` maximum gripper error, zero contact force, and no collision.
    The source task retained the connector for `58.508 mm`; insertion and
    production authority remain explicitly unproven.
17. [x] Correct the insertion grasp and validate a scripted geometry baseline.
    The Lula `right_gripper` frame now targets `40 mm` behind the connector tip,
    while the insertion endpoint compensates by the same offset so the tip
    terminates at the socket. Held-out artifact
    `insert-20260824-held-12402-v5` contains 124 true-4-FPS observations and
    independently reconstructs `40.005 mm` clasp clearance, `0.0022 mm` depth
    error, `0.1075 mm` lateral error, zero orientation error, and four seated
    observations. It is kinematic-only; v1/v2/v3 remain negative evidence.
18. [x] Establish contact-aware scripted insertion. Enable accurate connector
    and receptacle collision/contact evidence during the insertion segment,
    preserve at least `30 mm` exposed tip, and require bounded force,
    orientation, depth, lateral error, and retained seating. Only after this
    baseline passes should we collect 12 training and two disjoint held-out
    insertion recordings for JEPA-WM. Held-out artifact
    `contact-insert-20260824-held-12405-v9` passed with a physics-jointed
    connector, active body/contact colliders, a compliant latch exclusion,
    an explicit `1.05×` nominal socket allowance, 64 insertion increments,
    four seated observations, `0.0797 mm` lateral error, `0.0036 mm` depth
    error, `0.00159 rad` orientation error, bounded tracking, and `0 N`
    measured contact. The live interlock polls every intermediate physics
    update. The same sensor measured `142.4 N` on the rejected v3; v4/v5 then
    failed closed at `105.8 N`. V6 is superseded because its safety loop ran
    only at 4 FPS and its manifest did not bind the exact runtime contract. V7
    bound that contract but did not retain sub-limit transient force maxima;
    v8 does, while v9 additionally rejects non-finite/negative sensor values
    before aggregation.
19. [x] Complete insertion-domain JEPA-WM readiness. The exact contact-aware
    corpus now contains 12 unique TRAIN seeds and two disjoint HELD_OUT seeds
    under the v9 contract. Proposal fingerprint `ea7ce27c...ea255` passed all
    176 held-out post-attachment rollouts with `0.998534` mean active cosine
    and `4.41223e-8` sequence MSE. The resumable
    `ops/jepa_wm_insertion_corpus.sh` workflow owns
    the split/seed roster, strict per-recording validation, and exit backup. No
    current insertion artifact grants JEPA control or filming. The first exact
    adapter, fingerprint `4341ea85...be48`, completed 500 batch-one updates but
    failed both held-out seeds: `-2.31725e-5`/`12.5%` and
    `-4.12553e-5`/`47.7273%` mean improvement/win rate. Aggregate evidence was
    `-3.22139e-5`/`30.1136%`; a TRAIN diagnostic also failed, while adaptation
    still improved seed 12600 substantially over the base model. The result is
    preserved and recovery-backed up. The complete-epoch adapter, fingerprint
    `47969a0a...af9`, then visited all 1,056 TRAIN rollouts exactly once and
    passed both whole held-out seeds: `+6.13514e-5`/`82.9545%` and
    `+5.94737e-5`/`85.2273%`. Aggregate evidence over 176 rollouts is
    `+6.04125e-5`/`84.0909%`. This closes the offline insertion-energy
    checkpoint only. The first proposal-centered search then lowered energy
    below zero and recorded actions in all `16/16` sampled contexts, but failed
    insertion direction in all `16/16` (`0.119557`/`0.0687623` mean cosine).
    That exact negative is recovery-backed up. The next profile constrains
    search with the observable current-to-target DROID goal action; JEPA has
    not yet selected or executed an insertion. That aligned search passed goal
    direction `8/8` on seed `12600` but produced `-5.76454e-6` mean improvement
    over zero and beat recorded action only `1/8`; seed `12601` improved over
    zero by `+2.52775e-5` and beat recorded `7/8`, but passed goal direction
    only `6/8`. The planner therefore accepts a refinement only when it both
    meets the goal gate and improves proposal latent energy by at least `1e-6`;
    otherwise it retains only a goal-aligned proposal and explicitly blocks
    the context if both candidates fail. Exact v3 reports accepted `1/8` and
    `3/8` refinements, selected `7/8` and `6/8` contexts after goal-aligned
    proposal fallback, and blocked the rest. All selected actions passed goal
    alignment, but seed `12600` beat the recorded action only `1/7` times.
    Goal-aligned train-only negative mining and fresh whole held-out seeds
    remain required before any live insertion or filming checkpoint. The
    distinct `goal_aligned` adapter profile mines only bounded sequences.
    Active first actions must meet the same `0.95` observable-goal cone;
    first actions stay stationary when translation/rotation norms are at most
    `1e-5` and gripper delta is at most `0.005`, while future actions remain
    perturbed as hard negatives. The first run wrote no checkpoint: 72
    zero-action and 12 sub-`1.6 µm` transition frames exposed the missing
    stationary branch. That fail-closed result is recovery-backed up. The
    corrected exact-corpus preflight classifies 960 active and 96 stationary
    first actions; all active actions clear the cone with minimum cosine
    `0.988497`. A second fail-closed run found 24 context-36/37 active actions
    with `21.0–21.9 mm` demonstrated translation outside the planner's `20 mm`
    bound. Projecting those fallbacks through the canonical bounds preserves
    the goal cone (minimum `0.988497`); replacement now uses that bounded
    fallback and rechecks alignment. No checkpoint was written and the result
    is recovery-backed up. The completed bounded adapter, fingerprint
    `236514a7...d9dd1`, then remained fail-closed: seed `12600` scored
    `+6.11294e-5`/`60.2273%`, seed `12601` scored
    `+6.30901e-5`/`82.9545%`, and the aggregate was
    `+6.21097e-5`/`71.5909%`, below the `75%` win-rate gate. Losses cluster in
    the sub-millimetric insertion tail, where planner-bound noise averaged
    `5.34 mm` (95th percentile `13.22 mm`). The next separately named
    `goal_aligned_relative` profile scales noise from the demonstrated action,
    reducing that tail to `0.109 mm` mean/`0.361 mm` p95 while retaining the
    same bounds and goal cone. Its separately serialized floors are `10 µm`
    translation, `10 µrad` rotation, and `0.005` normalized gripper delta.
    Fingerprint `93735353...97320` completed at
    `+6.91447e-5`/`64.7727%` and `+7.02181e-5`/`84.0909%`; aggregate evidence
    was `+6.96814e-5`/`74.4318%` (131/176 wins). This recovered five wins but
    remains fail-closed because seed `12600` did not pass. Remaining losses
    versus the generic adapter cluster at contexts 80–106. The next checkpoint
    must fingerprint the generic adapter as a warm start and fine-tune the
    aligned local margin at pinned `1e-4` rate, preserving late-tail
    discrimination. The distinct profile is
    `goal_aligned_relative_finetune`; it derives the exact generic parent and
    rejects corpus, selection, source-revision, fingerprint, or rate drift.
    Child fingerprint `396496bf...fadc`, bound to generic parent
    `47969a0a...af9` and config `f66e1d46...f485`, visited all 1,056 TRAIN
    rollouts once at the pinned `1e-4` rate and passed both diagnostic seeds:
    `+7.05642e-5`/`87.5%` and `+7.06248e-5`/`87.5%`. Aggregate evidence was
    `+7.05945e-5`/`87.5%` (154/176 wins), and the exact artifacts are
    recovery-backed up. Existing held-out seeds are diagnostic only after
    informing these changes; freeze this artifact and evaluate it on fresh
    whole held-out recordings before planner or live insertion authority.
    The frozen child subsequently passed fresh whole held-out seeds `22600`
    and `22601` without adaptation: `+4.78526e-5`/`80.6818%` (71/88) and
    `+1.97309e-5`/`87.5%` (77/88), respectively. Aggregate evidence was
    `+3.37918e-5`/`84.0909%` (148/176), closing the strict 2/2 fresh offline
    energy gate. The typed fresh roster, raw 112-frame recordings, reports,
    and 12 GB recovery copy are preserved. This opens only the frozen
    task-aware planner benchmark; JEPA actuation and filming remain closed.
    That planner gate is currently `1/2`. With the exact frozen adapter,
    proposal `ea7ce27c...ea255`, and one fixed `4 × 64 / 8` search identity,
    seed `22601` selected `8/8`, kept direction/goal alignment at `100%`, and
    beat recorded energy `8/8` at `+2.02691e-5` mean. Seed `22600` selected
    `7/8`; context `44` blocked because initializer/search goal cosines were
    `-0.360`/`0.891` against the `0.95` cone, and only `4/7` selected actions
    beat recorded energy. The reconstructive readiness artifact therefore
    remains false, is recovery-backed up, and grants no live or production
    authority. Treat `22600–22601` as diagnostic from this point; improve the
    early insertion initializer from TRAIN-only evidence and retest a frozen
    revision on a new disjoint fresh pair.
    The TRAIN-only diagnostic localized the main initializer failure at context
    `44`: proposal `ea7ce27c...ea255` passed the goal cone in only `3/12`
    recordings (`0.863752` mean cosine). A separately named proposal adds a
    first-action-to-three-step-goal direction loss with weight `1.0`, retaining
    all prior supervision. Frozen fingerprint `efdf848c...7596f` passed context
    `44` in `12/12` TRAIN recordings (`0.998467` mean, `0.997284` minimum) and
    passed `1,019/1,020` active-goal TRAIN rollouts overall. Its ordinary
    diagnostic held-out proposal gate passed all `176/176` rollouts. With the
    adapter and planner identity unchanged, both diagnostic planner seeds then
    selected `8/8` contexts with `100%` goal/first-action passes: seed `12600`
    beat zero/recorded `8/8` and seed `12601` beat them `8/8` and `6/8`.
    Context `44` is selected on both with goal cosine above `0.998`. Freeze
    these exact bytes and evaluate the unchanged planner on a new disjoint
    two-seed recording pair; diagnostic success grants no live authority.
    Fresh seeds `32600–32601` then localized a separate late-tail search issue:
    the 256-candidate search passed all direction gates but seed `32601` beat
    zero only `6/8` and recorded actions `5/8`, so readiness remained `1/2`.
    A lower-weight initializer regressed the diagnostic planner and was not
    promoted. The bounded planner now also scores 12 exact-context TRAIN
    sequences, each projected into the existing proposal trust region, for
    268 total candidates. It beat zero and recorded actions `8/8` on both
    diagnostic seeds `12600–12601` and reconstructed `2/2` on the now-observed
    `32600–32601` pair, with all selection/direction gates passing. Typed
    readiness distinguishes the exact current policy from preserved historical
    reports. Since `32600–32601` informed the change, this remains diagnostic;
    a new disjoint two-seed pair is still required before live authority.
    The committed policy then passed untouched reserved seeds `42600–42601`.
    Both contact-aware v9 recordings validated all 112 frames with `0 N`
    measured contact. The frozen adapter passed both 88-rollout energy gates
    at `+6.19085e-5`/`81.8182%` and `+4.72146e-5`/`79.5455%`. The exact
    268-candidate planner selected `8/8` contexts on each seed, passed every
    direction/goal gate, and beat zero plus recorded actions `8/8` on both.
    Mean selected improvements over zero/recorded were
    `+5.84725e-5`/`+3.94733e-5` and
    `+2.33684e-5`/`+1.70866e-5`. Reconstructive readiness is `2/2`; all raw
    evidence and the verified 12 GB recovery copy are preserved. This closes
    frozen offline planner readiness and opens only the live simulator
    shadow/safety milestone. Actuation, filming, and production remain closed.
    The corrected dense command window is `43–106`; its committed frozen
    policy passed untouched reserved seeds `52600–52601` across all 64
    commands, so reconstructive offline readiness is now `2/2`. The full
    provenance, metrics, off-by-one negative, and preserved-artifact record are
    in the [dense insertion planner checkpoint](../README.md#dense-insertion-planner-checkpoint).
    At roughly 32 seconds per command, the search remains outside the live
    freshness budget. The resident proposal path has now passed no-actuation
    context-43 checks on both untouched seeds with fresh responses, passing
    live IK/contact/attachment evidence, and no result. The no-actuation policy
    rejects execution for both sessions. This closes `2/2` resident
    first-action freshness/no-actuation safety only. The separately authorized
    one-action trial then executed safely with zero contact and retained
    attachment, but regressed its three-frame translation target, so no
    multi-step authority opened. The controller now fails closed unless a
    projected insertion action reduces that target error by at least `25%`;
    fresh `2/2` no-actuation evidence under this stronger rule was initially
    blocked because wrapper reconstruction preceded Isaac's resumed
    physics-ready update. The corrected play, update, pre-refresh interlock,
    reconstruct, second interlock, read, continuity-check, and pause lifecycle
    now passes both seeds. Each
    independently rejects the overshooting full/half projections and accepts
    the quarter-scale projection with fresh timing, retained attachment, zero
    contact, no collision, and no execution result. This opens only the
    predeclared bounded one-action trial; multi-step authority remains closed.
    That trial executed safely with retained attachment and zero contact, but
    realized only `15.1%` translation-error reduction against the required
    `25%`, while rotation error increased. It stopped after one action and is
    preserved as a negative. Realized post-action progress must become the next
    fail-closed step boundary before another action can be authorized.
    Insertion filming and production authority remain false. Exact evidence is
    in the linked README checkpoint.
    The control-resolution checkpoint is paused for correction. Its prior
    runtime path combined drive targets with direct articulation state-setting,
    so the measurements establish neither Franka precision nor a defensible
    command deadband. The corrected benchmark uses drive-only runtime motion,
    first proves a stable observed baseline, and repeats zero, `0.5`, and
    `1.0 mm` probes at three representative poses with and without the attached
    load. It preserves the `3 mm` task tolerance and all existing force,
    collision, attachment, joint, and velocity limits; the longer `1.0 mm`
    period does not weaken the velocity gate. No further JEPA action is allowed
    until this benchmark establishes the deadband and orientation-hold bound.
    Its first full-roster attempt failed closed on cumulative zero-command
    drift before any nonzero probe; the follow-up requires an eight-interval
    globally stable baseline and distinguishes HOLD drift from active-target
    tracking error. Authenticated retries showed both a still-improving
    `10 s` window and a later stable endpoint that was not continuous with the
    unsettled capture. Capture now qualifies stability before its strict frame
    and persists typed early failures; measurement independently requalifies
    after pause/resume. The resulting three HOLD repetitions passed with
    micrometer-scale drift, while the first `0.5 mm` probe was rejected before
    actuation by the unchanged velocity gate. A retry with the safely longer
    persisted period was also rejected before actuation because IK selected an
    inadmissibly distant joint configuration. A bounded local-branch retry
    admitted the forward probe, but its rollback missed the unchanged tracking
    threshold at the 32-update cap. A longer bounded observation then reached
    command-relative settlement but missed exact reset equivalence by a small
    joint margin. Rollback now has a persisted tracking cap before the
    unchanged full reset check. A strict-cap retry then exposed that loaded
    rollback settlement was being measured against the drive target rather
    than the stable observed reset; those command and repeatability targets are
    now distinct. The cap derives from the unchanged reset joint tolerance,
    avoiding a second arbitrary limit. No acceptance or safety limit changed.
    The next retry completed two of three `0.5 mm` repetitions; the third
    rollback missed the same limit by `0.004404 mrad`, so only the bounded
    observation cap is extended. Exact negative evidence is retained in the
    linked README checkpoint.
    The frozen six-case pose/load roster is now terminal: attached contexts 43
    and 106 completed all probes, while attached context 74 and all three
    unloaded contexts failed closed at the first `0.5 mm` forward/rollback
    cycle. The request is distinguishable from zero drift but is not repeatable
    across pose and load, and observed orientation drift exceeds the prior hold
    value. Controller settlement/rollback must be corrected and the same roster
    rerun before choosing a deadband or orientation-hold tolerance. No JEPA
    action is authorized.
    The corrected drive-only rerun now supersedes that status. All three
    attached pose cases completed `9/9`; unloaded context 43 failed closed on
    a discontinuous final IK projection, while unloaded contexts 74 and 106
    each preserved a bounded second-`1 mm` motion/recovery timeout. The
    attached evidence establishes the existing `0.5 mm` minimum translation
    and `1.25 mrad` orientation-hold tolerance for exactly one context-43
    attached action. The earlier direct-state no-actuation sources did not
    survive the drive-only reset-equivalence check. Stable capture then exposed
    that action history incorrectly included the whole settling interval;
    conditioning it on the final stable interval produced fresh current-code
    `2/2` no-actuation passes. Qualified stable HOLD history is now canonical
    zero so independent resets bind exactly. The first exact-bound trial still
    stopped before motion because live wrapper reconstruction exhausted the
    unchanged freshness limits. Execution now requires typed live-state
    continuity evidence before refreshing only that exact bound action's timing;
    the limits are not extended. It does not cover context 74's larger target
    or authorize a second action. Realized progress remains a terminal
    fail-closed gate. The resulting one-action trial stopped before progress
    measurement when forward settlement missed its bounded threshold, and its
    interlocked rollback also timed out. It retained attachment at zero contact
    with no collision, but terminalized as `rollback_failed` with `0/1` applied
    steps. Controller settlement and verified reset rollback must now pass under
    the unchanged limits before another JEPA action is authorized. The
    insertion-specific observation cap now matches the validated 48-update
    drive-only checkpoint without changing any threshold or safety limit, and
    fresh current-code no-actuation checks pass `2/2`. This reopens only a
    separately invoked one-action retry, never an automatic next step.
    Multi-step, filming, and production authority remain false; exact metrics
    and evidence are in the
    [insertion control-resolution checkpoint](../README.md#insertion-control-resolution-checkpoint).
    A second 48-update retry proved that extending the observation cap alone
    did not solve loaded settlement. The corrected controller now separates
    active drive targets from stable realized reset targets, authenticates both
    before motion, and reuses the bounded drive-only load compensation without
    weakening any gate. Fresh exact-code no-actuation checks then passed `2/2`.
    One separately invoked context-43 action subsequently passed settlement,
    action tracking, realized progress, attachment, and the contact/collision
    interlock, and terminalized after exactly `1/1` applied step. No automatic
    second action, multi-step insertion, filming, or production authority is
    granted. Exact negative and successful evidence is in the linked README
    checkpoint.
    The bounded follow-up implementation subsequently stopped twice before a
    fresh observation or any command: first on legacy safety reconstruction,
    then on loss of the resident runtime across source reload. Both exits
    verified the recovery backup. Cross-generation runtime, plug, collision,
    and dual contact-sensor handoff is now regression-tested, but the old live
    generation cannot be recovered without an unobserved update. A follow-up
    therefore requires a separately authorized fresh two-action chain; no
    current multi-step, filming, or production authority is granted. Exact
    evidence remains in the linked README checkpoint.
    The authorized fresh two-action chain was then attempted. An abbreviated
    recording name first failed before capture; the corrected exact held-out
    source reached fresh action-1 inference, but every scale failed the
    unchanged observation/command freshness gates before execution authority.
    The terminal result is `0/1` applied, action 2 was never captured or
    attempted, attachment/contact/collision evidence remained safe, and both
    exits verified recovery backups. The same boundary must be requalified
    after reducing handoff latency; no retry, automatic follow-up, multi-step,
    filming, or production authority is granted. Exact timing and session
    evidence remains in the linked README checkpoint.
    Exact live-state reauthorization then cleared the unchanged freshness gate
    in a separately authorized fresh chain. Action 1 passed settlement,
    tracking, realized progress, attachment, and contact/collision checks, but
    the fresh action-2 camera capture failed closed before observation or
    inference because its render retries occurred after the synchronized
    timeline pause. The interlocked capture lifecycle is corrected locally;
    the terminal evidence remains `1/2` applied and does not establish a
    two-action loop. Another live retry, any third action, multi-step insertion,
    filming, and production remain unauthorized. Exact evidence is in the
    linked README checkpoint.
    The next explicitly authorized chain requalified the interlocked camera
    capture and again applied action 1 with verified progress. It produced a
    genuine fresh action-2 observation and inference, but the fixed recorded
    target horizon made every scale fail the unchanged progress gate before
    IK or motion. Follow-up target selection now derives and persists its
    bounded horizon from the synchronized live pose; initial target semantics
    and every control/safety threshold are unchanged. The terminal result
    remains `1/2`, and the live-pose correction still requires requalification
    before claiming a two-action loop. Exact evidence is in the linked README
    checkpoint.
20. [ ] Complete one end-to-end bounded unknown-start run in Isaac Sim. Freeze
    the training artifacts and a versioned reset-sampling contract first,
    including its workspace/geometry bounds and distribution, then draw one
    held-out reset from a reserved seed. The reset must not replay a recorded
    approach or post-attachment prefix, use manually staged task phases, or
    provide seed-specific waypoints. Ordinary synchronized camera observations
    and robot proprioception remain valid controller inputs.
    From that reset, JEPA-controlled, safety-gated actions must approach and
    acquire the connector, retain it through transport, align it with the
    socket, complete contact-aware insertion, and hold the validated seated
    state. Persist the reset-contract fingerprint and seed, initial reset,
    every observation/action, task-phase handoff, contact/force/collision and
    tracking evidence, and the terminal task decision so the run can be
    reconstructed independently. One valid held-out success satisfies the
    requested lab stopping point for this stage.
    Multi-seed repeatability, autonomous failure recovery, real-hardware
    transfer, and production authority are explicitly deferred and must not be
    added as prerequisites without new direction.
