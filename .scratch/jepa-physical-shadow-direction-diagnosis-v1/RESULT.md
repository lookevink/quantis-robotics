# Physical shadow direction mismatch diagnosis v1

## Terminal result

The known-start canary failure was reproduced offline with the exact direct and
planned three-action sequences. A deterministic unit loop failed in `0.003 s`
and reproduced `3/3`: CEM improved its saved objective while returning a first
action that failed the frozen `0.9` direction gate.

The owning defect was planner orchestration, not the safety threshold or Isaac.
`plan_shadow_candidates` optimized latent, prior, and optional task penalties;
it evaluated `FirstActionGate` only after CEM had selected a winner. Therefore a
candidate that was structurally ineligible for promotion could still win the
search. The physical residual exposed the defect by assigning that candidate a
lower latent energy, but did not create the missing constraint.

The correction wraps the existing proposal-centered bounds with the unchanged
first-action gate. A sampled first action that fails is projected to the bounded,
gate-validated direct first action; the remaining horizon stays searchable. No
activity, cosine, trust-region, safety, or task threshold changed. The exact
regression loop is green and the full remote suite passes `821` tests.

No model training, canonical evaluation, Isaac execution, live action, filming,
hardware, or production authority was exercised. The consumed canary remains
terminal and was not retried.
