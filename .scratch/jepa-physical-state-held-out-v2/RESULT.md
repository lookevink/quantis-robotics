# Physical-state residual disjoint held-out gate v2 result

## Terminal outcome

The separately frozen v2 gate passed. The unchanged authenticated physical
state residual artifact scored both exact TRAIN-disjoint canonical HELD_OUT
recordings once, using one shared set of scores for the combined and per-seed
populations. All three populations passed every frozen energy, segment,
router, residual-bound, and exact-hold predicate with no failure reason.

This repairs only the v1 experimental-ordering defect. Before the irreversible
v2 claim, the evaluator authenticated the complete JEPA-WM/DINOv3 runtime,
base checkpoint and source revisions, residual artifact and training contract,
TRAIN evaluation, adjudication, the consumed v1 claim/failure, the absence of a
v1 score, and the successful runtime remediation evidence.

## Authenticated evidence

- Evaluator SHA-256: `46d4fdd448fc94a2049eaf5323e7c23b3bff98da6a19e724afd47c5c2450a836`.
- Evaluator revision: `3e06d7ece80078315b19371444c61c984430d3fc`.
- Frozen configuration SHA-256: `2cc04d39fa843850e0fe685e1349b508b563aeed07fc1b989ec76f1ffe65fd4e`.
- Terminal report SHA-256: `ca100086efaf394a65d893627142252db7107a87c76352aac6993841431cc2f4`.
- Exclusive access-claim SHA-256: `29536227542455ad066ad4a2eec8566f3df2ede45d6abb069b05732e30ae1520`.
- Exact recordings: `held-00` seed 12600 and `held-01` seed 12601,
  168 examples each, 336 combined.
- Model load: 11.490 s; encoding: 18.781 s; scoring: 94.806 s;
  peak CUDA allocation: 2.075 GiB.

The combined and both per-seed populations each achieved 98.2143% overall
recorded-action wins, 94.3396% retained wins, and 100% post-contact wins.
Combined router accuracy was 99.1071%; per-seed accuracy was 99.4048% and
98.8095%. Retreat recall was 100% and advance recall 99.0566% in every
population. Failed-closed fraction was 2.3810%, semantic holds activated no
owned route and used the exact base map, and the maximum applied residual ratio
was `0.15000002086162567`, inside the frozen `0.15 + 0.000001` tolerance.

Focused remote verification passed 15 tests, the full remote suite passed all
809 tests, and independent standards/spec reviews found no remaining pre-run
finding. The first standards review caught a forgeable boolean claim guard;
that was replaced with a private completed-authentication token and the final
evaluator identity was rebound and reverified before canonical access.

## Recovery and authority boundary

The repository recovery workflow completed successfully with a 16 GiB state
copy on the dedicated `/dev/nvme1n1` ext4 filesystem. Independent `cmp` and
SHA-256 checks proved byte-identical live and recovery copies of the report and
claim. No v2 failure artifact exists.

Stop. This checkpoint establishes disjoint offline generalization evidence for
the frozen residual world-model action encoder only. It did not train, run
Isaac, issue a JEPA action, film, operate hardware, or grant production
authority. Those remain separate future milestones.
