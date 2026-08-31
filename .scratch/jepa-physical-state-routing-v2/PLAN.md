# Physical-state routing probe v2

## Terminal result

Produce one authenticated, grouped TRAIN-only answer to a single question:
can task-relative physical observations predict the next semantic motion route
without visual latents, candidate actions, or scripted progress? Stop after the
route probe. Do not train residual experts, access held-out data, authorize live
JEPA action, or film.

## Frozen redesign

- Build the exact versioned 26-value physical observation declared by the
  router artifact: task-relative plug, end-effector, and gripper geometry;
  gripper and tracking telemetry; contact force; attachment; and the previous
  realized 7D action.
- Predict `hold`, `retreat`, `advance`, or `active_other` once per observation
  with a two-layer 64-unit nonlinear router.
- Never use visual latents, candidate or future actions, scripted phase,
  context index, or seed as runtime inputs.
- Label the declared `retreat_hold`, `align_hold`, and `seated_hold` segments
  as semantic holds even when their recorded telemetry contains tiny drift.
- Treat low confidence and `active_other` as fail-closed base routing.
- Reserve future residual experts for retreat and advance, with the existing
  hard `0.15` residual/base embedding trust region.

## Probe and stopping rule

Authenticate all exact 12 TRAIN recordings and bind the result to their
selected content digest. Use leave-one-recording-out folds. Report overall,
per-route, per-segment, per-fold, confidence, and fail-closed metrics. A pass
also requires zero retreat or advance activations in all semantic hold
segments.

One pass permits a separately frozen bounded-residual training experiment. A
failure ends this experiment; it does not permit in-probe retuning.
