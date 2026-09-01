# Milestone 20 unknown-start reset authentication v4

Run the unchanged reset-only contract with recording identity
`unknown-start-reset-v4-62603` and unused held-out seed `62603`. Preserve all
terminal v1/v2/v3 ledgers, negatives, recovery copies, and spent seeds.

V4 adds observability only. The validator names every rejected invariant and
persists `UNKNOWN_START_RESET_NEGATIVE.json` with the complete pre-validation
evidence before recorder cleanup. The negative embeds every frozen run identity
and the captured-frame hash; terminal failure validates those bindings and
records the negative artifact hash. No distribution, tolerance, coordinate
frame, physics, force, collision, attachment, action, training, or filming gate
changes.

Known implementation defects continue to be repaired autonomously. Pause for
the user only if reproducible evidence proves a representation or model
insufficiency.
