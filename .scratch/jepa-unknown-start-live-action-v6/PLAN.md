# Milestone 20 unknown-start live action v6

Reuse the exact, authenticated v4 rollback recovery and the unchanged passed
unknown-start shadow source. V5 claimed but executed zero model actions because
its atomic capture encountered a playing timeline immediately after the runtime
reload. V6 corrects only that pre-execution lifecycle defect: capture acquires
and freezes the paused boundary before reauthentication.

Repeat the same single frozen model action under a fresh identity. Preserve all
model, source, controller, projection, tracking, time, force, collision, and
attachment thresholds. Apply at most one model action and pause afterward.

Require a byte-identical 16 GiB recovery copy before terminal success. Do not
train, film, attach new authority, or continue automatically to a second model
action.
