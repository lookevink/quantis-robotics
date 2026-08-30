## Destination

Produce one authenticated comparison that determines whether balanced linear
conditioning, a small nonlinear residual, or regime-separated conditioning
resolves the retained-retreat failure, and leave either one frozen promotion
candidate or one precisely scoped next blocker.

## Notes

- Execution is intentionally included in this map, not merely design.
- The detailed outcome sequence, gates, and change-control policy live in
  [PLAN.md](PLAN.md).
- The 48-hour window is a coordination/review horizon, not a deadline. Work
  advances only when the preceding outcome gate is satisfied; no duration is
  predicted for any milestone.
- Use `quantis-aws` for every AWS read or mutation, `diagnosing-bugs` for any
  failed invariant, `implement` for code changes, and `code-review` before the
  experiment freeze and final checkpoint.
- Code defects are auto-approved to repair. A repair that changes any frozen
  experimental variable invalidates affected results and requires a fresh
  fingerprint and restart; it never silently changes the running experiment.
- One GPU training or evaluation job at a time. No simulator action runs
  concurrently.

## Decisions so far

- [Freeze the experiment ledger](issues/01-freeze-experiment-ledger.md) — authenticated the exact corpus, artifacts, recovery copy, known baseline, and immutable experiment configuration; corrected the draft batch size from 2 to the proven value 1 before freezing.
- [Build the treatment seam](issues/02-build-treatment-seam.md) — added fingerprint-bound B/C/D action-conditioning families, balanced sampling, signed negatives, scoped oracle routing, and one-shot evaluation guards with 26 focused tests passing.
- [Pass preflight and review](issues/03-pass-preflight-and-review.md) — real TRAIN-only smokes proved identical initialization and finite updates, an inactive signed-margin defect was repaired and re-frozen, and the full 733-test suite plus standards/spec review passed.

## Not yet specified

- The deployable context-derived gate if only the oracle-separated treatment
  succeeds.
- The exact same-reset counterfactual capture protocol if every offline
  treatment fails.
- The live shadow/control and filming milestone after a successful canonical
  offline gate.

## Out of scope

- Weakening force, collision, tracking, attachment, task, freshness, or the
  documented 0.75 offline action-control gate.
- Training on either canonical held-out recording or the development canary.
- Automatically expanding a successful offline result into simulator control,
  a JEPA action, filming, hardware, or production authority.
