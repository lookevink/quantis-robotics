# Result: safely blocked before motion

- Session: `unknown-start-live-action-v3-62605`
- Evaluation: `92cad7b4...7d56c`
- Applied model actions: `0`
- Gate reasons: `stale_observation`, `command_time_invalid`
- Observed age: `3.177046 s`
- Recovery: verified 16 GiB backup

The atomic in-process harness removed remote-call gaps, but the required final
live reauthentication/synchronization itself exceeded both unchanged time
limits. Candidate execution omitted the existing post-synchronization
`InsertionEvaluationRefresh` used by insertion and contact-grasp policies, so
the gate correctly rejected pre-reauthentication timestamps. No IK execution
or robot motion occurred.
