# Runtime command routing versus published JEPA world models

Research date: 2026-08-30

## Short answer

The proposed three-way router is a small, task-specific mixture-of-experts
adapter around the action embedding. It does not change the visual encoder,
the latent predictor, the controller, or the safety gates.

For a three-action candidate horizon, let `x(a)` be its routing statistic in
base-frame X and let `epsilon` be a deadband fixed from TRAIN data before the
experiment. The intended action embedding is:

```text
x(a) < -epsilon  -> base(a) + negative_x_residual(a)
x(a) > +epsilon  -> base(a) + positive_x_residual(a)
otherwise        -> base(a)
```

`base(a)` is the existing seven-dimensional DROID-action-to-1024-dimensional
action map. The residual experts make small learned corrections to that map;
they do not directly command the robot. All seven action coordinates still
enter whichever map is selected.

In this corpus, the names have a concrete empirical meaning:

- **Negative-X motion:** every intentional attached retreat rollout in all 12
  TRAIN recordings has negative mean base-frame X. This route is therefore a
  runtime-observable proxy for the retreat dynamics that the global adapter
  failed to retain.
- **Positive-X motion:** every intentional alignment and insertion rollout in
  TRAIN has positive mean base-frame X. This route isolates the forward
  dynamics that the retreat-corrected global adapters reversed.
- **Neutral/hold through the unchanged base map:** X motion within the frozen
  deadband receives no learned motion residual. This avoids treating tiny
  measured hold drift as intentional retreat or insertion, which caused the
  diagnostic oracle treatment's only segment failure.

These are command modes, not semantic phase labels: the router must not read a
scripted `retreat`, `align`, or `insert` index. They are also not joint-space
directions and do not mean that the arm itself has a special “negative-X
motor.” A remaining design requirement is to specify the horizon statistic
and ensure that a command with near-zero X but material Y/Z/rotation/gripper
activity is not accidentally classified as a hold.

## Relationship to the papers

### What remains aligned

Negative/positive/neutral X are not concepts learned by original V-JEPA.
Original V-JEPA is action-free masked feature prediction: its predictor is
conditioned on masked video context and target locations, not robot commands
([V-JEPA method](https://arxiv.org/html/2404.08471#S3)). The three modes belong
entirely to Quantis's downstream action-conditioning layer.

The proposal keeps the core JEPA-WM recipe: encode observations into frozen
visual features, condition a predictor on candidate robot actions, predict
future features, and plan by minimizing distance to a goal feature. V-JEPA
2-AC freezes its visual encoder, trains an action-conditioned predictor on
robot trajectories, and executes only the first optimized action before
replanning in an MPC loop ([V-JEPA 2, Sections 3.1–3.2](https://arxiv.org/html/2506.09985#S3)).

Quantis also inherits choices supported by the later systematic JEPA-WM
study: its pinned upstream checkpoint uses DINOv3-L features, a 12-block
AdaLN/RoPE predictor, and L2 goal cost. The study found DINO features better
for fine object localization, AdaLN strongest on average (with
task-dependent differences), and L2 planning cost consistently stronger than
L1 in its experiments
([JEPA-WM encoder and predictor findings](https://arxiv.org/html/2512.24497v3#S5.SS2),
[pinned upstream Quantis evaluation config](https://github.com/facebookresearch/jepa-wms/blob/13cf1d9c7e476f53c17714d2e0f1dc239a883ce0/configs/evals/simu_env_planning/droid/jepa-wm/droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_r256_alpha0_ep64_decode.yaml)).

### What deviates

1. **Hard routing is not in the published architectures.** V-JEPA 2-AC uses
   one shared, continuous, roughly 300M-parameter block-causal transformer for
   all actions. Actions, end-effector states, and visual patches receive
   separate affine embeddings and are jointly processed; it does not select a
   predictor or action map from the sign of one coordinate
   ([V-JEPA 2-AC architecture](https://arxiv.org/html/2506.09985#S3.SS1)).
   The JEPA-WM study compares feature, token, and AdaLN conditioning, but not
   sign-routed experts
   ([JEPA-WM action-conditioning variants](https://arxiv.org/html/2512.24497v3#S11.SS2)).

2. **Quantis freezes much more of the model.** V-JEPA 2-AC freezes the visual
   encoder but trains the full nonlinear predictor. The JEPA-WM formulation
   jointly trains the action encoder, predictor, and optional proprioceptive
   encoder while freezing only the visual encoder
   ([JEPA-WM training definition](https://arxiv.org/html/2512.24497v3#S3)).
   The proposed Quantis experiment instead leaves the pretrained 12-block
   predictor fixed and learns only small residual action maps. This is a
   low-data, low-risk adaptation experiment, not a reproduction of the paper's
   post-training procedure.

3. **The router adds a task/coordinate-frame prior.** It encodes the corpus
   fact that retreat is negative base-X and alignment/insertion are positive
   base-X. That is deployable here because it uses the candidate command, not
   a hidden phase label, but it will not automatically transfer to a rotated
   task, another action convention, or a horizon containing direction changes.
   V-JEPA 2-AC instead attempts to learn action effects continuously and even
   reports a locally smooth action-energy landscape. The paper also documents
   camera/robot-axis ambiguity as a limitation
   ([energy landscape](https://arxiv.org/html/2506.09985#S4.SS2),
   [camera sensitivity](https://arxiv.org/html/2506.09985#S4.SS3)). A hard
   deadband can introduce an energy discontinuity, so the proposed bounded
   discontinuity test is important, especially for CEM search.

4. **Neutral actions deliberately retain the pretrained prior.** The papers
   train their shared predictor over the continuous action distribution; they
   do not reserve near-zero commands for an unchanged base map. Quantis does
   so because its canary showed that applying a motion residual to tiny hold
   drift damaged hold ranking. This is conservative, but it also means the
   experiment does not learn task-specific contact/settlement dynamics for
   neutral holds.

5. **The local objective is planning-ranking adaptation, not paper-style world
   model training.** Quantis directly teaches the recorded command to beat
   zero, mismatched, mined, and opposite-X commands in terminal latent energy.
   V-JEPA 2-AC uses next-feature teacher forcing plus a two-step autoregressive
   rollout loss; the JEPA-WM study uses embedding prediction and finds that
   multistep rollout training materially affects planning
   ([V-JEPA 2-AC loss](https://arxiv.org/html/2506.09985#S3.SS1),
   [JEPA-WM multistep result](https://arxiv.org/html/2512.24497v3#S5.SS2)).
   The ranking objective is defensible for this gate, but it can fit ordering
   without learning a generally calibrated transition function.

6. **The adaptation data is far narrower.** The router will be fit on 12
   scripted insertion recordings. V-JEPA 2-AC used roughly 62 hours of DROID,
   and the JEPA-WM study reports planning performance increasing with the
   quantity and diversity of dynamics data
   ([V-JEPA 2-AC data](https://arxiv.org/html/2506.09985#S3.SS1),
   [JEPA-WM data scaling](https://arxiv.org/html/2512.24497v3#S5.SS2)). Routing
   can resolve a local gradient conflict; it cannot substitute for broad
   causal coverage.

### An inherited precision caveat, not a router deviation

The selected upstream DROID checkpoint has `proprio_encoding: none` and a
visual-only planning cost. This actually matches the JEPA-WM study's proposed
DROID/Robocasa cross-domain model, which omitted proprioception because the
proprioceptive spaces were not aligned. However, the same study found
proprioception helpful in aligned environments, especially near a goal where
small metric motion may barely change frozen visual embeddings
([JEPA-WM proposed per-domain models](https://arxiv.org/html/2512.24497v3#S5.SS3),
[proprioception finding](https://arxiv.org/html/2512.24497v3#S5.SS2)). Quantis
uses robot state elsewhere in its proposal, controller, and safety stack, but
the router does not add it to the latent dynamics model. That remains relevant
for submillimetre insertion and hold discrimination.

## Interpretation

The router is consistent with JEPA as a *framework*, but it is not a result
claimed by the JEPA papers. It is a sharply bounded engineering hypothesis
suggested by Quantis's own ablation: one global adapter traded retreat against
forward motion, while two oracle-separated residuals modeled both dominant
regimes. Replacing the semantic oracle with an observable command-derived gate
tests whether that capacity can be made deployable.

Passing a fresh scripted canary would support this adapter for the current
coordinate convention. It would not establish that sign routing is generally
better than paper-style joint predictor training. Failure should not lead to
more router tuning on the canary; the cleaner paper-aligned alternatives are
same-reset counterfactual data and, if justified by a separately frozen
milestone, post-training more of the predictor with proprioceptive input and
multistep rollout loss.
