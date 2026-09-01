# Corrected physical residual known-start shadow canary v2

Run one zero-actuation canary on the other canonical held-out start: seed 12600,
context 110. Bind the unchanged proposal/residual and corrected planner. Require
capture safety, the shadow direction gate, and counterfactual Isaac safety.
Recover terminal evidence and stop. Any failure is terminal; never apply an
action, tune, retry, train, or film inside this experiment.
