# Result: action applied, tracking rollback failed

- Session: `unknown-start-live-action-v4-62605`
- Evaluation: `1f3d8c1f...0e1d0`
- Applied model actions: `1`
- Contact force: `0 N`
- Collision: none
- Recovery backup: verified 16 GiB

The strict gate selected translation scale `1.0`, rotation `0.25`, and gripper
`0.25`; freshness was `0.008923 s` and IK errors were `0.000490 mm` and
`0.568 mrad`. Directional realization was excellent (translation cosine
`0.999935`, rotation cosine `0.999395`), but settlement stopped with
`3.486 mm` translation, `3.642 mrad` rotation, and `0.010854` normalized
gripper errors. Tracking rollback was commanded but did not reach its reset
tolerance within the generic budget. The stage remained paused, unattached,
collision-free, and at `0 N`.
