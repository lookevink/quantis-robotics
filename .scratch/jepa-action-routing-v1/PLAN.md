# Runtime-command action-routing experiment

## Terminal result

Produce one authenticated offline result for a frozen three-path action map:
negative-X motion, positive-X motion, and unchanged-base fallback. Stop after
the fresh-canary gate and, only if it passes, the already-declared canonical
offline gate. No live JEPA action or filming follows automatically.

## Frozen design

- Start from the existing authenticated control adapter and freeze its complete
  seven-to-1024 action map.
- Train one zero-initialized linear residual for active negative-X horizons and
  one for active positive-X horizons.
- Use only the candidate 7D DROID command at runtime. Scripted phase, context
  index, and seed are forbidden router inputs.
- Route inactive horizons through the exact base map. Active horizons whose
  mean X is inside the deadband also use the base map as a conservative
  non-X fallback; they are reported separately from neutral holds.
- Fit only the exact 12 TRAIN recordings. Seed 72600 is excluded from selection
  because it informed this design.
- Capture and evaluate fresh noncanonical seed 72601 once. Canonical seeds
  12600 and 12601 remain sealed unless that fresh canary passes unchanged
  gates.

## TRAIN-only deadband evidence

The previously used 1 mm candidate-mining activity threshold is not a routing
threshold: every slow insertion rollout is below it. The frozen 0.1 mm
translation/X deadband is the existing command-resolution boundary and yields
this exact 2,016-rollout TRAIN roster:

| Route | Rollouts |
| --- | ---: |
| negative-X residual | 564 |
| positive-X residual | 1,296 |
| inactive base | 106 |
| active non-X base fallback | 50 |

The two motion routes receive 1,008 deterministic balanced updates each. Base
examples receive no update because the base map is frozen and must remain
bitwise identical.
