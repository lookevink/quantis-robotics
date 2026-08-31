# Runtime-router counterfactual diagnosis result

## Terminal result

The failed signed counterfactual gate is explained by candidate-dependent
expert switching and the resulting training-objective conflict. It is not a
demonstrated DINOv3 representation failure, an Isaac failure, or a controller
regression.

- Report:
  `/home/ubuntu/docker/jepa-wm/checkpoints/quantis_action_routing_v1/counterfactual-diagnosis-v1.json`
- Report SHA-256:
  `7fac08d9e1efa56e0d1b88db77bc3776fdbc9835aac740c5731afba949263961`
- Scope: 288 authenticated TRAIN transitions; no optimizer step, held-out
  access, canonical access, simulator action, or model update.
- Route roster: 96 negative-X, 180 positive-X, and 12 active base/non-X.

## Decisive intervention

Signed triple order is `recorded < X-zero < X-opposed`.

| Scoring mode | Retreat | Alignment | Insertion | All |
| --- | ---: | ---: | ---: | ---: |
| candidate routed, as deployed | 0% | 0% | 0% | 0% |
| recorded route locked | 100% | 100% | 100% | 100% |
| frozen base only | 0% | 100% | 100% | 66.7% |

Nothing except the route assigned to each candidate changed between the first
two rows. Locking all four candidates to the recorded action's route recovered
all 288 orderings. The frozen base already ordered every forward alignment and
insertion transition, but ordered no retreat transition. The negative expert
therefore repaired retreat locally; the gate failed when an opposed action was
sent through the other expert.

Under candidate routing, `X-zero < X-opposed` was 0% in every segment. On
retreat, recorded still beat X-opposed in 95/96 cases, but X-opposed was always
scored below X-zero. On alignment and insertion, recorded beat X-opposed in
only 7/96 and 13/96 cases. These TRAIN failures reproduce the sealed canary's
shape, so this is not primarily a held-out generalization failure.

## Why training produced this

The training objective contains `recorded < X-zero` and
`recorded < X-opposed` margins. It does not contain the gate's missing edge,
`X-zero < X-opposed`. Candidate routing also makes each expert serve two
opposing purposes: fit its own recorded motions and make the opposite regime's
sign-flipped candidates look bad.

The measured gradients are nearly antiparallel:

| Expert | Opposite margin active | Own-fit vs cross-rejection cosine |
| --- | ---: | ---: |
| negative-X | 100% | -0.9654 |
| positive-X | 91.7% | -0.9675 |

The cross-rejection gradient norms are comparable to the own-fit norms, so
this is not a negligible auxiliary effect. The two learned residual weight
matrices themselves have cosine `-0.8015`, consistent with the optimizer
building opposing maps.

## Architectural warning

The residual coefficients are not small relative to the frozen base map.
Negative-X column norms are 1.54-3.50 times their base columns; positive-X
columns are 0.88-1.83 times base. Small commands keep the resulting residual
embeddings to a mean 8.0% and 3.1% of the base embedding, respectively, but
crossing the signed-X deadband creates an embedding jump 4.97 times the frozen
base jump for the same X change. The implementation is therefore a pair of
regime maps with a discontinuous switch, not merely two gentle directional
corrections.

## Representation conclusion

This experiment does not prove that the representation is sufficient for the
entire physical task. It does rule out representation insufficiency as the
cause of this gate failure: with the same frozen DINOv3 encoder and predictor,
route locking recovered every TRAIN ordering, and the frozen base recovered
every forward ordering. Same-reset physical clip separability remains an
independent question, but it is not needed to explain the present failure.

## Smallest next remediation experiment

Do not retrain this candidate-routed architecture. Freeze a new experiment in
which routing is determined from the observed physical context, independently
of the candidate action, so every candidate in one comparison uses the same
dynamics expert. Attachment/contact/proprioceptive state is a legitimate
runtime input; recorded action, scripted phase, context index, and seed are
not. Add the exact `X-zero < X-opposed` pairwise margin, enforce a bounded
deadband discontinuity, and require the full TRAIN signed gate before spending
a fresh canary.

Recorded-route locking is diagnostic evidence only; it cannot be promoted
because the recorded future action is unavailable at runtime.

## Harness execution notes

Two direct launches were rejected before data encoding because the scratch
entrypoint initially omitted the repository import path and then the standard
DINOv3 runtime environment. A third run reached a mistaken hard-coded route
roster assertion and wrote no report. The roster invariant was corrected from
authenticated TRAIN actions and moved before model loading/encoding. No
partial result was treated as evidence, and the successful run wrote the one
terminal report allowed by the plan.
