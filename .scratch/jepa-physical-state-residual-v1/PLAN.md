# Physical-state bounded residual training v1

## Terminal result

Produce one authenticated TRAIN-only physical-state action-conditioning
artifact and one terminal TRAIN optimization-contract evaluation. Stop after
that evaluation. Do not access held-out/canonical recordings, run Isaac, issue
a JEPA action, or film.

## Frozen training sequence

1. Authenticate the exact 12-recording TRAIN roster, selected content, base
   checkpoint, frozen control map, and passed physical-router probe.
2. Fit one final `26 -> 64 -> 64 -> 4` router on all authenticated TRAIN
   observations using the already selected hyperparameters. Fit normalization
   from those observations and require the same route gates as the grouped
   probe, including zero owned-route activations in semantic holds.
3. Freeze the router, its normalization, the base action map, visual encoder,
   and JEPA predictor.
4. Train only the two zero-initialized linear residuals. Alternate semantic
   retreat and advance examples. The frozen physical router selects one route
   per observation and that decision remains fixed across every candidate.
5. Enforce the `0.15` residual/base embedding bound inside the encoder for
   every action, not as a post-hoc loss.

## Objective and gate

Reuse the previously corrected ordered TRAIN objective: recorded action versus
zero, mismatched, locally mined, X-zero, and X-opposed candidates, including
`recorded < X-zero < X-opposed`. Evaluate recorded, zero, X-zero, and X-opposed
over all 2,016 TRAIN transitions.

The terminal gate requires the existing aggregate/retained/post win rates,
positive mean improvement in every semantic segment, signed ordering in
retreat/alignment/insertion, exact base-map behavior in semantic holds, the
hard applied-residual bound, and proof that the base and router remained
unchanged during residual training.

## Stopping rule

A pass creates an offline TRAIN candidate for a separately frozen disjoint
held-out gate. A failure ends this experiment and preserves one reconstructible
negative. Neither result automatically authorizes held-out access, retraining,
simulator action, live control, or filming.
