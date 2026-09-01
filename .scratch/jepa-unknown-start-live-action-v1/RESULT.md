# Result: failed before motion

- Claim: `a43c91e3...19257`
- Phase: candidate capture
- Applied model actions: `0`
- Recovery: verified 16 GiB backup

Reset and source authentication passed, but capture rejected the nonexistent
proposal alias `experimental_shadow_candidate`. No observation was persisted,
execution was never claimed, and the robot did not move. The owning layer is
the workflow's proposal identity binding, not the model, controller, or Isaac.
