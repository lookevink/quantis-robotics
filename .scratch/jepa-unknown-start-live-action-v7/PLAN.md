# Milestone 20 unknown-start live action v7

Reuse the exact, authenticated v4 rollback recovery and unchanged passed shadow
source. V6 claimed but executed zero model actions because runtime reload drifted
the transient reset before atomic candidate capture acquired control.

V7 changes only reset-trial initialization: after capture acquires and freezes
the paused boundary, it reconstructs the authenticated reset command from the
frozen seed/sample and waypoint solver, initializes the exact observed joints
with zero velocity, and validates before and after camera capture.

Apply at most one unchanged model action, then pause. Preserve every model,
controller, projection, tracking, time, force, collision, and attachment gate.
Require a byte-identical 16 GiB recovery copy before terminal success. Do not
train, film, add authority, or continue to a second model action.
