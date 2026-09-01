# Result: passed

- Session: `unknown-start-shadow-canary-v5-62605`
- Reset result: `70a8fba8...a6d58`
- Experiment configuration: `d39cade2...1d7a3`
- Claim: `4cb42ddc...258d`
- Evaluation: `cf083d1c...22bef`
- Terminal result: `4f1ba5c5...3c542`
- Recovery verified: yes, byte-identical 16 GiB copy

The fresh authenticated 512x512 wrist observation was collision-free and at
exactly `0 N`. The unchanged frozen proposal/residual model produced a bounded
three-action plan. CEM scored `256` candidates, improved the objective by
`0.002837`, and passed the first-action gate at cosine `0.999805`. Independent
Isaac counterfactual safety passed at translation scale `1.0`, rotation scale
`0.25`, and gripper scale `0.25`.

No action was applied, execution never started, and no training or filming
occurred. This closes the unknown-start zero-actuation model-readiness gate
only. Live motion requires its own separately frozen authority boundary.
