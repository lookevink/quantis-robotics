# Observed-context routing experiment

## Terminal result

Produce one authenticated TRAIN-only result for a candidate-independent action
map routed by the previous realized 7D action. Stop after the frozen TRAIN gate.
Do not evaluate or capture a canary, access canonical model evidence, authorize
live JEPA action, or film.

## Frozen design

- Preserve the authenticated DINOv3 encoder, 12-block predictor, base
  checkpoint, frozen control action map, exact 12-TRAIN corpus, safety limits,
  and evidence contracts.
- Derive two routing weights once per observed context from
  `ControlObservation.previous_action`: negative-X and positive-X. Apply those
  same weights to recorded, zero, X-zero, X-opposed, mismatched, and mined
  candidate horizons.
- Candidate actions are forbidden router inputs. So are scripted phase,
  context index, seed, and recorded future action.
- Use the unchanged base map when observed X is inside the existing 0.1 mm
  deadband. Blend linearly from base to the relevant residual over the next
  0.1 mm, making the route continuous at both deadband boundaries.
- Train only the two zero-initialized linear residuals. Keep the base map
  bitwise frozen.
- Add the gate's missing pairwise objective directly:
  `recorded < X-zero < X-opposed`.

## Pre-frozen TRAIN audit

Using the frozen command deadbands, previous realized action and the recorded
future command choose the same hard route for 1,895/2,016 transitions (94.0%).
The previous-action roster is 576 negative-X, 1,283 positive-X, and 157 base.
Mismatches are concentrated at attachment, motion-to-hold, hold-to-motion, and
seated settling boundaries. This is a declared risk, not a result to tune
after training.

## Gate

Evaluate all 2,016 TRAIN transitions once. This is an optimization-contract
gate, not held-out generalization evidence. Require:

- recorded action win rates of at least 0.90 overall, 0.85 retained, and 0.95
  post;
- positive mean improvement over zero in every semantic segment;
- signed triple order on at least 0.75 of retreat, alignment, and insertion;
- candidate-invariant context weights for every comparison;
- unchanged base parameters; and
- maximum full-route residual/base embedding ratio at most 0.15 across the
  recorded, zero, X-zero, and X-opposed TRAIN evaluation candidates. Mismatched
  and mined actions must use the same context weights during training, but are
  not inputs to this frozen continuity gate.

## Stopping rule

Run one training artifact and one TRAIN evaluation. A pass may justify a
separately frozen fresh-canary milestone. A failure ends this experiment and
must be diagnosed without retraining inside this milestone.
