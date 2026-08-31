# Causal context routing probe result

Run date: 2026-08-31 (America/Los_Angeles)

## Terminal result

The authenticated TRAIN-only preflight passed. The frozen grouped causal-route
probe failed, so bounded residual action-model training is not authorized.

- Preflight SHA-256:
  `4b25b36070e56cbc09860764b54ed7b3f0a6ec152de8524e61c6cda6bfcbc436`
- Route-probe SHA-256:
  `0d6a98f6b0a9f19cec0fa0dfb58298f2f42f6a12c718efca6ec67de78200f8a1`
- Exact TRAIN selection: 12 recordings, 2,016 windows, contexts `113..280`.
- Selected-input SHA-256:
  `576404f64ac55f47490ef8358eb2121f4dd044f5ab72e396a2817f439fe3d839`
- Held-out, canonical, simulator, and live-action access: none.
- Recovery: live and `/mnt/quantis-assets/quantis-state` copies match
  byte-for-byte; backup verified at `2026-08-31T15:32:38Z`.

## Preflight

Structural ownership was feasible: all owned failing slices had a trainable
future route. The frozen baseline reproduced the known failure boundary:

- `grasp_attach`: 0% recorded wins, mean improvement `-0.0000914811`.
- `retreat`: 0% recorded wins, mean improvement `-0.0003185012`.
- `align`: 100% recorded wins, mean improvement `0.0004122895`.
- `insert`: 100% recorded wins, mean improvement `0.0000754560`.

All 12 `grasp_attach` contexts had a retreat future label. The complete route
roster was 564 retreat, 1,296 advance, 106 hold, and 50 active-other examples.

## Probe gate

- Overall accuracy: `0.472718` (required `0.95`).
- Retreat recall: `0.624113` (required `0.98`).
- Advance recall: `0.433642` (required `0.98`).
- `grasp_attach` accuracy: `0.166667` (required `1.0`).
- Failed-closed fraction: `0.424107` (maximum `0.05`).
- Passthrough residual-route activations: retreat-hold `11`, align-hold `12`,
  seated-hold `2` (maximum `0` for each).
- Per-recording fold accuracy ranged from `0.023810` to `0.708333`.

## Interpretation and boundary

This negative does not prove the frozen visual representation is insufficient.
It proves that the frozen combination of globally pooled visual latent,
absolute context pose, previous realized action, and the specified 64-unit MLP
does not generalize the next-motion route across recordings. The large fold
spread is consistent with recording-specific geometry or feature scaling, and
the near-absent confident hold predictions show that the current observation
and calibration contract cannot safely protect passthrough states.

Per the frozen stopping rule, do not lower thresholds, train residual experts,
retune this probe, access held-out data, or launch live action inside this
milestone. A separate milestone may test task-relative geometry, normalized
causal state, and spatial rather than globally pooled visual features.
