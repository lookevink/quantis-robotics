# Physical residual known-start shadow canary v1

## Terminal result

Run one zero-actuation deployment canary from the authenticated contact-grasp
known start on HELD_OUT seed 12601. Bind the held-out-qualified contact-grasp
proposal and the held-out-qualified physical residual artifact, produce one
direct proposal, score one shadow search under the observed 26-value physical
route, evaluate its counterfactual Isaac safety projections, persist a terminal
report, recover it, and stop.

## Frozen non-authority

The workflow must never call `apply_control_response`, create execution-started
evidence, write a control result, or bind an experimental candidate. Direct and
planned actions are evidence only. A successful terminal report may open a
separately frozen milestone-20 unknown-start proposal; it does not authorize
motion, filming, hardware, or production.

## Stopping rule

The exclusive claim is consumed before capture. Any failure after the claim is
terminal for this milestone. Preserve its evidence, back it up, and do not patch
and rerun inside the same experiment.
