# Physical-state bounded residual result

## Terminal outcome

The one permitted authenticated TRAIN-only router fit and two-residual training
run completed. The artifact passed its router, parameter-ownership, task-win,
signed-order, semantic-hold, and effective residual-bound checks, but the
literal terminal report is `physical_state_residual_train_failed` because an
inherited shared gate compared a floating-point value of
`0.15000002086162567` directly with the configured `0.15` cap.

This is a harness numeric-boundary negative, not a model or representation
negative. Per the stopping rule, the experiment ended without a patch, retry,
held-out access, JEPA action, or filming.

## Authenticated identities

- Experiment configuration:
  `b296b7fc064627f13ed87c1baeaf84d4961f1b04db115f9afcc689bf05dda78d`.
- Exact TRAIN selection:
  `f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`.
- Selected corpus contents:
  `576404f64ac55f47490ef8358eb2121f4dd044f5ab72e396a2817f439fe3d839`.
- Passed physical-router probe:
  `385305e7268e296702f1ecfd5e0104426894a119e15e73567acb03e129801ffa`.
- Preflight report:
  `8e6e52e412317fe61447fde39c30bf0fdd7b1ed209b59304b94a1cbf4c862088`.
- Trained artifact:
  `7f3cb2a99e749dd56e4f4a988b3b1c332ab13c4661c3413118ffb270f237db51`.
- Training report:
  `08f41c36f4e0987c4652244ee0011001de30c76a6e014bd94ad4966780d78aba`.
- Terminal TRAIN evaluation:
  `a5527c50eeb4a3223b74779584f7dad1f6edc29121534a778abf9147bf4d6bdd`.
- Completed run state:
  `bf41c643dc5b0e491be17fd534fc13b51bf2c27ce5efe46e963fc27250673c19`.

The initial wrong control-map filename was rejected before preflight and is a
nonterminal invocation record, fingerprint
`cc0a6fd213662819da5f59c76e7a2a9cf4b3a9992e0622231357903f1e247e9b`.
It fit no router and performed no residual update.

## Training evidence

The final `26 -> 64 -> 64 -> 4` router reached `0.992560` TRAIN accuracy,
`0.026290` fail-closed fraction, `0.995012` mean confidence, and zero owned
route activations in `retreat_hold`, `align_hold`, and `seated_hold`. Only the
two residual matrices trained: 14,336 parameters, with exactly 1,008 retreat
and 1,008 advance updates. The base map and serialized router remained bitwise
unchanged. Loss fell from `0.0205321` to `0.0115631`, with minimum
`0.00801351`.

## Terminal TRAIN metrics

| Slice | Recorded win | Signed order | Mean improvement |
| --- | ---: | ---: | ---: |
| all TRAIN | 0.982143 | 0.982143 | 0.001363744 |
| retained | 0.943396 | 0.943396 | 0.001509427 |
| post | 1.000000 | 1.000000 | 0.001296603 |
| grasp attach | 1.000000 | 1.000000 | 0.000571491 |
| retreat | 0.958333 | 0.958333 | 0.001653299 |
| retreat hold | 0.750000 | 0.750000 | 0.000017449 |
| align | 1.000000 | 1.000000 | 0.002164590 |
| align hold | 1.000000 | 1.000000 | 0.000007720 |
| insert | 1.000000 | 1.000000 | 0.000706141 |
| seated hold | 1.000000 | 1.000000 | 0.000000565 |

Every frozen task threshold passed. Semantic holds used the exact base map for
recorded, zero, X-zero, and X-opposed candidates. The maximum observed applied
residual/base ratio was `0.15000002086162567`; the encoder's explicit bounded
scaling caused the approximately `2.09e-8` overshoot through floating-point
arithmetic. The physical experiment's direct check allowed `1e-6`, but the
reused `_gate_for_context_indices` helper also imposed raw `<= 0.15`, making
the conjunction false.

## Recovery and authority

All six live evidence files match their recovery copies byte-for-byte at
`/mnt/quantis-assets/quantis-state`. The dedicated recovery filesystem backup
completed at `2026-08-31T18:47:21Z` and contains 16 GiB of Quantis state.

The artifact is not promoted by this literal failed report. The next milestone
is a separately frozen harness remediation: make the shared ratio predicate
use the same explicit numerical tolerance, regression-test the exact boundary,
and adjudicate this immutable report without retraining or rescoring. Held-out
access, another training run, Isaac action, live JEPA control, filming,
hardware, and production authority remain closed.
