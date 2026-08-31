# Physical-state residual disjoint held-out gate v1

## Terminal result

Evaluate the one authenticated physical-state residual artifact exactly once on
the two sealed canonical HELD_OUT recordings. Produce one immutable terminal
report and stop. Do not train, run Isaac, issue a JEPA action, or film.

## Frozen inputs

- Artifact SHA-256: `7f3cb2a99e749dd56e4f4a988b3b1c332ab13c4661c3413118ffb270f237db51`.
- TRAIN report SHA-256: `08f41c36f4e0987c4652244ee0011001de30c76a6e014bd94ad4966780d78aba`.
- Immutable TRAIN evaluation SHA-256: `a5527c50eeb4a3223b74779584f7dad1f6edc29121534a778abf9147bf4d6bdd`.
- Report-only adjudication SHA-256: `f9a395cf0926cf965a52c7a498a15b9bf03bc22032da4234bae23ca5dcc3412f`.
- Canonical seeds: `held-00` seed 12600 and `held-01` seed 12601, with
  the exact manifest identities frozen in `experiment-config.json`.
- Wrist-camera contexts `113..280`, stride one, action horizon three.

## Gate

Authenticate every input before model loading. Score recorded, zero, X-zero,
and X-opposed actions once. The combined population and each seed must
independently satisfy the existing aggregate, retained, post, per-segment, and
signed-order predicates. The serialized router must independently satisfy its
frozen accuracy/recall/fail-closed/hold predicates on the combined population
and each seed. Every candidate must respect the residual ratio bound within
the frozen numerical tolerance and all semantic holds must use the exact base
map.

Before the first canonical read, atomically claim the one permitted evaluation.
Pre-claim invocation failures remain retryable; once the claim exists, process
loss or any exception is terminal. Persist both the claim file and its parent
directory before the first read. Authenticate the exact evaluator module and
its explicit frozen implementation revision even on the gitless deployment
checkout. Assert the global rollout protocol still has action horizon three.

After the terminal evaluation report or terminal failure report, run the repository recovery workflow onto the
dedicated `/mnt/quantis-assets` filesystem. Independently hash the live report
and access claim, then require byte-identical copies of the claim and exactly
one terminal-report alternative below
`/mnt/quantis-assets/quantis-state/jepa-wm/checkpoints` before documenting the
result.

## Stopping rule

A pass establishes disjoint offline generalization evidence only. A failure
preserves one reconstructible negative. Neither result authorizes retraining,
a second evaluation, Isaac, live control, filming, hardware, or production.
