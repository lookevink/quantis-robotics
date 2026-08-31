Type: research
Status: resolved
Blocked by: 05

## Question

If no promotion-eligible treatment survives, does the evidence call for a
runtime-observable soft gate or same-reset causal captures, and what is the
smallest separately authorized next experiment?

## Answer

The evidence calls for a runtime-observable action-mode gate before any new
same-reset causal capture. D's independent-canary result proves that the
frozen encoder/predictor and two small residual maps can rank the demonstrated
retreat, alignment, and insertion transitions correctly at the same time.
That rules out representation insufficiency as the demonstrated blocker for
the dominant motion regimes.

The canary telemetry supplies a particularly small deployable seam: all 48
retreat rollouts have negative mean base-frame X, while all 48 alignment and
all 64 insertion rollouts have positive mean X. The failed hold windows contain
only measured drift. The exact same invariant holds split-safely on TRAIN:
`576/576` retreat rollouts are negative-X, while `576/576` alignment and
`768/768` insertion rollouts are positive-X across all 12 recordings. Hold
drift mixes signs, so sign alone is insufficient; routing drift through a
motion expert caused D's sole segment failure. The next bounded treatment
should therefore:

1. retain the frozen DINOv3 encoder, 12-block predictor, shared official base
   action map, safety gates, objective, and exact 12-TRAIN fitting roster;
2. route a proposed three-action horizon using only runtime-observable command
   values: negative-X motion, positive-X motion, or neutral/hold;
3. apply small residual experts only to the two motion routes and use the
   shared base path for neutral actions;
4. determine the neutral deadband from TRAIN data and declared command
   quantization before freezing, never from held-out outcomes;
5. prove route invariance, exact-zero neutrality, sign symmetry, bounded
   discontinuity at the deadband, serialization identity, and no semantic
   phase/context-index dependency in fast tests;
6. use a fresh noncanonical scripted canary because seed 72600 informed this
   design; do not reuse 72600 for selection and do not open canonical seeds
   until the fresh canary passes the unchanged gate; and
7. stop after that offline result. It grants no live JEPA or filming authority.

Only if the fresh action-routed treatment fails should the following milestone
capture same-reset negative-X/zero/positive-X futures. That capture would be a
new separately frozen scripted-data experiment, not an automatic continuation
of this one.
