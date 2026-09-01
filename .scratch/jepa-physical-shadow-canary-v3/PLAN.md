# Unknown-start physical shadow canary v1

Freeze one zero-actuation handoff from passed reset
`unknown-start-reset-v5-62604` into the already authenticated contact-grasp
proposal and physical residual planner. The run may create exactly one fresh
wrist observation, one direct model response, one shadow-only CEM search, and
one counterfactual safety evaluation. It must not apply an action, train, film,
touch hardware, or grant production authority.

The experiment ends on its first terminal result. Implementation, harness,
environment, or stale-live-state failures do not adjudicate representation
sufficiency; they are diagnosed and corrected in a separately versioned run.
Only a fully authenticated response/shadow/safety failure may be reported as a
model checkpoint.
