# Result: safely blocked before motion

- Session: `unknown-start-live-action-v2-62605`
- Evaluation: `a025b95b...3e629`
- Applied model actions: `0`
- Gate reason: `stale_observation`
- Observed age: `3.242441 s`
- Recovery: verified 16 GiB backup

Reset/source authentication, capture, and candidate binding passed. The
separate capture, binding, and apply server calls exceeded the unchanged
freshness limit, so every projection was rejected before IK execution and the
robot did not move. This is a harness cadence failure, not a model rejection.
