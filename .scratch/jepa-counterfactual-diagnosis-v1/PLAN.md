# Runtime-router counterfactual diagnosis

## Terminal result

Explain why the frozen runtime-command router achieved high recorded-versus-zero
win rates but failed signed counterfactual ordering. Produce one authenticated,
TRAIN-only diagnostic report. Do not train, tune, evaluate another canary, read
canonical model evidence, or authorize live action.

## Frozen inputs

- Router artifact SHA-256:
  `45326210f5a47f74a9008670e9bf0be03b3ef40955b3c9af79017588d9b79c30`.
- Base checkpoint, source revision, control map, experiment configuration, and
  exact 12-TRAIN roster are inherited unchanged from
  `../jepa-action-routing-v1/freeze-ledger.md`.
- The sealed seed-72601 router report is used only as the deterministic failure
  replay. New GPU probes use TRAIN recordings only.

## Frozen probes

For every TRAIN seed, select the same 24 motion contexts:

- retreat: `114, 120, 126, 132, 138, 144, 150, 156`
- alignment: `166, 172, 178, 184, 190, 196, 202, 208`
- insertion: `216, 224, 232, 240, 248, 256, 264, 272`

The authenticated TRAIN commands route as 96 negative-X, 180 positive-X, and
12 active base/non-X horizons. The 12 base horizons are context 216, whose mean
X is just inside the frozen signed-X deadband; retaining them is intentional.

Score the recorded, all-zero, X-zero, and X-opposed actions under three
read-only modes:

1. `candidate_routed`: each candidate chooses its own runtime-command route,
   matching the failed evaluation.
2. `recorded_route_locked`: every counterfactual uses the route chosen by the
   recorded command, changing only the candidate action.
3. `base_only`: every candidate uses the frozen base action map.

Report every pairwise ordering, not only the terminal triple. Also report
residual column norms, residual/base embedding ratios, discontinuity around the
X deadband, and deterministic gradient cosines between each expert's own-route
fit pressure and its use as the opposite route's negative.

## Stopping rule

Stop after the one diagnostic report. A confirmed cause may define a later
remediation experiment, but this milestone performs no optimizer step, writes
no model artifact, and consumes no held-out evidence beyond the already-sealed
failure trace.
