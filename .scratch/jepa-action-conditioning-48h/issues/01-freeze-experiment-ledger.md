Type: task
Status: resolved
Blocked by:

## Question

Can the corpus, artifacts, code revision, configurations, authority boundary,
and known baseline be frozen into one reproducible ledger before new code or
training begins?

## Answer

Yes. [`freeze-ledger.md`](../freeze-ledger.md) authenticates the local source
state, AWS account, base checkpoint, control adapter, evidence reports, all 12
TRAIN recordings, both canonical held-out recordings, the development canary,
and the verified recovery boundary. Every exact recording independently passed
the 284-frame contact-aware contract with 0 N maximum connector force and four
seated observations. No recording job was active, and 16 focused diagnostic
and evaluation tests passed.

The freeze caught and corrected one draft regression before creating the
experiment identity: the proven baseline used batch size 1, while the first
plan draft said 2. The frozen canonical configuration is
[`experiment-config.json`](../experiment-config.json), SHA-256
`a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14`.
The original preflight fingerprint was explicitly superseded before training
when review proved inactive signed-X examples needed masking.
No model training, evaluation, or simulator action ran while resolving this
gate.
