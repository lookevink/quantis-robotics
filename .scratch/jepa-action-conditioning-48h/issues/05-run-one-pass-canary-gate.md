Type: task
Status: resolved
Blocked by: 04

## Question

Which predeclared diagnosis follows from one seed-72600 comparison of A, B, C,
and D under the fixed global, segment, signed-X, and fingerprint gates?

## Answer

No promotion-eligible treatment passed. The exact frozen terminal label is
`frozen_dynamics_or_representation_blocker`; the evidence narrows that coarse
label to a routing/capacity conflict rather than proving representation
insufficiency.

| Treatment | Overall win | Retained win | Post win | Mean improvement | Frozen experimental gate |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 0.702381 | 0.056604 | 1.000000 | 0.000052213 | fail |
| B | 0.297619 | 0.943396 | 0.000000 | -0.000084014 | fail |
| C | 0.267857 | 0.849057 | 0.000000 | -0.000001553 | fail |
| D | 0.982143 | 0.943396 | 1.000000 | 0.001082767 | fail; diagnostic only |

A reproduced the known phase split. B learned retreat but reversed every
post-retreat ranking. C reduced absolute energy and almost met the retained
threshold, but also reversed every post-retreat ranking. Oracle-routed D
passed the unchanged repository action-control gate and the main overall,
retained, post, retreat, and alignment thresholds. It failed only the stronger
every-segment requirement: four `retreat_hold` rollouts had negative mean
improvement and `0.25` wins. D is non-promotable regardless.

The one-pass summary selected no artifact. Canonical seeds 12600 and 12601
were not opened, and no threshold was retuned.
