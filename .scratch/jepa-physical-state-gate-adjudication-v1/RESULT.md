# Physical-state residual gate adjudication result

## Terminal outcome

The one permitted report-only adjudication passed. It changes only the
duplicated residual-ratio predicate from raw `<= 0.15` to
`<= 0.15 + 1e-6`. The immutable source report's observed ratio
`0.15000002086162567` passes that numerical tolerance, while the regression
suite proves `0.15001` and every non-finite gate metric still fail closed.

The existing artifact is therefore an offline
`physical_state_residual_train_candidate`. This establishes eligibility to
propose a separately frozen disjoint held-out gate; it does not authorize that
gate or any live action.

## Authenticated evidence

- Frozen adjudication configuration:
  `92986c48b5c05bbf0e93c5d7bc265d904eaa23daa56bd4ce13611d4c3a1a437e`.
- Immutable source TRAIN evaluation:
  `a5527c50eeb4a3223b74779584f7dad1f6edc29121534a778abf9147bf4d6bdd`.
- Existing trained artifact:
  `7f3cb2a99e749dd56e4f4a988b3b1c332ab13c4661c3413118ffb270f237db51`.
- Existing training report:
  `08f41c36f4e0987c4652244ee0011001de30c76a6e014bd94ad4966780d78aba`.
- Exact TRAIN selection:
  `f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`.
- Selected corpus contents:
  `576404f64ac55f47490ef8358eb2121f4dd044f5ab72e396a2817f439fe3d839`.
- Terminal adjudication sidecar:
  `f9a395cf0926cf965a52c7a498a15b9bf03bc22032da4234bae23ca5dcc3412f`.

The sidecar records `model_loaded: false`, `recordings_loaded: false`,
`rescored: false`, `trained: false`, `isaac_run: false`, and false access or
authority for held-out, canonical, live action, filming, and production.

## Verification and recovery

Focused verification passed locally and on the authenticated Quantis host;
the full local suite passed 793 tests with 138 expected skips. Independent
standards and spec reviews found no remaining pre-execution issue after
non-finite metric handling was restored.

The live sidecar and recovery copy match byte-for-byte at
`/mnt/quantis-assets/quantis-state`. The dedicated recovery filesystem backup
completed at `2026-08-31T19:05:20Z` and contains 16 GiB of Quantis state.

## Authority boundary

The TRAIN candidate may be named in a new proposal for a separately frozen,
disjoint held-out offline gate. No held-out or canonical recording has been
opened by this milestone. Retraining, rescoring, Isaac, JEPA action, filming,
hardware, production, and automatic expansion into the next gate remain
closed.
