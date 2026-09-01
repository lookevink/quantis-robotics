# Corrected physical residual known-start shadow canary v2 result

## Terminal outcome

The single authenticated zero-actuation evaluation passed. Its exclusive claim
bound config fingerprint
`329b771fb1cf700371230cfda6e94240e31f250015834ebc4257e04bff8aa7b3`
to session `physical-shadow-canary-v2-12600` before capture. It used the other
canonical held-out start, seed 12600 at context 110.

The observation completed at `0 N` contact force with no collision. The
corrected shadow planner scored the frozen 256-candidate budget, reduced the
objective from `0.1484913528` to `0.1404376250`, and passed the unchanged first
action direction gate with cosine `0.9925016733` against the `0.9` minimum. The
separate Isaac counterfactual safety projection passed at translation/gripper
scale `1.0` and rotation scale `0.25`. Authority remained `shadow_only`.

No action was applied and execution never started. No control result, execution
evidence, experimental candidate binding, insertion binding, training,
filming, hardware, or production authority was created. The worker stopped and
Isaac was stopped after verification.

The terminal result and recovery copy are byte-identical with SHA-256
`d38104495cc1a26465221b67ef76a8441a1868f5be98cc108493e3d40f9c0a56`.
The evaluation and recovery copy are byte-identical with SHA-256
`808ec1457ac50e92026c7ce86441e3d631965eda1e0d4878d7f00ed7af75bfa1`.
The full remote suite passed all 827 tests before the canary.

## Boundary

This closes the corrected planner's known-start zero-actuation integration
gate. It does not itself authorize motion or filming. The next milestone is a
separately frozen milestone-20 reset and end-to-end execution contract; no
unknown-start reset or action was performed here.
