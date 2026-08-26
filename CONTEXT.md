# Domain language

## Bounded insertion rollout

A sequence of JEPA-WM insertion actions that begins from one fresh reset and
has a persisted maximum number of steps. Every next step requires the previous
step to have applied with passing settlement, realized progress, attachment,
contact, and collision evidence. Reaching the maximum never creates authority
for another action.

## Rollout position

The persisted one-based step index and maximum step count for one bounded
insertion rollout. A follow-up advances the position by exactly one and cannot
change the maximum.

## Demo-suitable autonomous drive

A four-action bounded attached-insertion rollout that independently
reconstructs as fully applied on both predeclared held-out seeds under the
unchanged control and safety gates. It demonstrates repeatable receding-horizon
control, not full seating, unknown-start insertion, filming authority, or
production readiness.
