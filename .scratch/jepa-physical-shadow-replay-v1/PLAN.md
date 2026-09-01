# Physical shadow corrected-planner offline replay v1

## Terminal result

Run the corrected planner exactly once against the immutable observation,
direct proposal, and physical residual from failed session
`physical-shadow-canary-12601`. Require the old evidence to fail only its
direction gate and the corrected replay to preserve objective improvement while
passing that unchanged gate. Recover the evaluation byte-identically and stop.

## Frozen non-authority

This is offline inference over saved frames. Do not start Isaac, capture a new
observation, evaluate simulator safety again, apply an action, train, film, or
create hardware/production authority. Never modify the source session.

## Stopping rule

One replay only. Any failed invariant is terminal. Do not tune and rerun inside
this experiment.
