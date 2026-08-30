# JEPA action-conditioning experiment: outcome-gated plan

The 48-hour window is a coordination and review horizon, not a predicted
schedule. Each phase begins only after its predecessor meets the declared exit
condition. A slow phase does not weaken, skip, or time out an evidence gate.

## Terminal result

At the end of the window, report exactly one of:

1. `balanced_linear_candidate` — the simpler balanced linear treatment passed;
2. `nonlinear_residual_candidate` — only the small nonlinear treatment passed;
3. `regime_conflict_confirmed` — only oracle-separated maps passed;
4. `frozen_dynamics_or_representation_blocker` — no treatment passed; or
5. `experiment_invalidated` — a contract-changing code defect or evidence
   failure prevented an interpretable result.

No terminal result authorizes a simulator action or filming.

## Frozen experiment contract

- Official DROID JEPA-WM base checkpoint and pinned source revision.
- Frozen DINOv3 encoder and frozen 12-block nonlinear predictor.
- Authenticated 12-recording TRAIN roster only for fitting.
- Noncanonical seed-72600 canary for development selection.
- Canonical held-out seeds 12600 and 12601 are inaccessible to fitting and are
  evaluated only once if a promotion-eligible finalist survives the canary.
- Insertion contexts `113..280`, action horizon 3, and unchanged action format,
  normalization, terminal latent-L2 objective, and documented readiness gate.
- Fixed 2,016 updates, batch size 1, optimizer, learning rate, margins, seed,
  phase roster, signed-negative construction, and evaluation thresholds.
- A and B retain the existing global linear action-encoder shape. C adds only
  a zero-initialized `7 -> 32 -> 1024` SiLU residual. D adds two
  zero-initialized linear residual maps and uses recorded semantic phase only
  as an oracle diagnostic router.

## Treatment matrix

| ID | Treatment | Promotion eligible | Question |
| --- | --- | --- | --- |
| A | Existing global linear adapter and existing training contract | No; control | Does the known result reproduce? |
| B | Global linear adapter with balanced retained/post sampling and deterministic signed-X negatives | Yes | Was the failure optimization/negative imbalance? |
| C | B plus a zero-initialized nonlinear action residual | Yes | Is modest nonlinear action capacity sufficient? |
| D | B plus oracle-routed retained/post linear residuals | No; diagnostic only | Do the physical regimes require distinct maps? |

The retained sampling stratum is semantic contexts `113..165`; the post
stratum is `166..280`. Evaluation remains broken out into attachment/retreat,
retreat hold, alignment, alignment hold, insertion, and seated hold so the
sampling boundary cannot hide a local regression.

## Acceptance rules

Every report must pass the unchanged repository gate: finite metrics, positive
mean improvement over zero, and win rate at least 0.75.

For experimental selection, B or C must additionally achieve on the canary:

- overall recorded-vs-zero win rate at least 0.90;
- retained-stratum win rate at least 0.85;
- post-stratum win rate at least 0.95;
- positive mean improvement in every semantic segment;
- retreat ordering `E(x-negative) < E(x-zero) < E(x-positive)` in at least 75%
  of sampled contexts;
- alignment ordering `E(x-positive) < E(x-zero) < E(x-negative)` in at least
  75% of sampled contexts; and
- valid artifact, configuration, corpus-selection, and report fingerprints.

If both B and C pass, select B. D can classify the blocker but can never be
selected for promotion.

## Outcome-gated sequence

### Checkpoint and freeze

- Inventory the dirty working tree; preserve unrelated `error.log` and
  `supabase/` content.
- Review and test the new latent diagnostic and research note.
- Validate all 12 TRAIN identities, the canary identity, base checkpoint, and
  current adapter. Record fingerprints without evaluating canonical outcomes.
- Reproduce the known phase-locked baseline from existing authenticated
  reports.
- Write the immutable experiment configuration before implementation begins.

Exit: clean relevant checkpoint, complete ledger, no active training or
simulator recording job, and one frozen configuration fingerprint.

### Implement the bounded treatment seam

- Add versioned adapter-family serialization and strict load/apply identity.
- Add deterministic retained/post balanced sampling.
- Add X-zero and X-sign-flipped negatives that preserve the other six action
  coordinates exactly.
- Add B, C, and D behind one fixed training/evaluation interface.
- Make C and D residuals zero-initialized and prove they exactly match B's
  official-checkpoint initialization before an optimizer step.
- Make fitting commands reject canonical held-out and canary paths.
- Add unit, property, serialization, CLI, and synthetic energy-ordering tests.

Exit: focused tests pass; no GPU training has started.

### Independent preflight and experiment freeze

- Run code review against the experiment contract.
- Run the full non-PTY test suite once.
- Run one disposable one-batch smoke per new family using TRAIN data only.
- Delete/quarantine smoke artifacts and prove no canonical data was accessed.
- Freeze code revision, configurations, output names, and execution order.

Exit: reviewer has no blocking findings and every zero-step/one-step invariant
passes. Otherwise repair code, rerun this phase, and issue a new fingerprint.

### Serial training

- Preserve A as the immutable control.
- Train B, C, then D exactly once, serially, using the frozen configuration.
- Validate each artifact immediately after creation; do not evaluate canary
  outcomes until all treatments finish.
- Capture actual load, encoding, training, GPU-memory, and retry time.

The previous baseline measured 24 s load, 164 s encoding, and 2,660 s training.
These are historical measurements for diagnosing regressions, not forecasts or
deadlines for the new treatments.

Exit: four authenticated artifacts/reports or `experiment_invalidated`.

### One-pass canary gate

- Evaluate A, B, C, and D on seed-72600 once, in the frozen order.
- Run the signed-X/per-token probe and semantic-segment summary once per model.
- Apply the predeclared selection table without retuning or retraining.

Interpretation:

- B passes: sampling/negative construction was sufficient; select B.
- C passes and B fails: nonlinear action capacity was required; select C.
- only D passes: classify `regime_conflict_confirmed`; do not promote D.
- none pass: classify `frozen_dynamics_or_representation_blocker`.

### Conditional branch

If B or C was selected:

- freeze its identity;
- independently validate the artifact and complete canary evidence;
- evaluate the two canonical held-out recordings exactly once;
- require the unchanged documented gate plus the stronger segment diagnostics;
- stop after the offline result, pass or fail.

If only D passed:

- do no more training;
- specify the next context-derived soft-gating experiment using only runtime
  observable inputs; and
- preserve D strictly as diagnostic evidence.

If none passed:

- do no more training;
- prepare the exact same-reset negative-X/zero/positive-X capture protocol;
- execute no simulator motion unless separately authorized at that point.

### Audit and regression checks

- Revalidate artifact/report fingerprints and exact roster membership.
- Compare phase curves against A and the pre-experiment diagnostic.
- Run focused tests and one final full non-PTY suite at the stable checkpoint.
- Review all code changes and classify every mid-run repair.
- Create one recovery backup only for the stable milestone state.

### Checkpoint and handoff

- Commit only reviewed, relevant files; preserve unrelated user work.
- Update the milestone documentation with the terminal result and authority
  boundary.
- Report simulator time, training time, evaluation time, backup time, code-fix
  time, and invalidated/retry time separately.
- Hand off the one next milestone; do not begin it automatically. If this lies
  beyond the 48-hour review horizon, report the current evidence-backed gate
  rather than compressing or skipping work.

## Code-defect policy

Auto-fix without additional approval:

- syntax/import/type/shape defects;
- deterministic indexing, serialization, fingerprint, CLI, quoting, path,
  permission, idempotency, or recovery defects;
- test-harness errors where the written frozen contract already determines the
  correct behavior.

After any such repair, restart the affected command from a clean artifact. Do
not resume a partially trained checkpoint unless the frozen contract explicitly
declared resumability.

Automatically repair but invalidate and re-freeze the experiment if the defect
touches corpus membership, context windows, phase routing, normalization,
candidate construction, loss, thresholds, optimizer budget, model family,
evidence schema, or artifact identity. These changes need no additional coding
approval, but results produced under the old contract cannot be compared as if
nothing changed.

Never treat a physics/safety failure, insufficient model metric, or unexpected
experimental outcome as a programming defect. Preserve one minimal negative
and stop that branch.
