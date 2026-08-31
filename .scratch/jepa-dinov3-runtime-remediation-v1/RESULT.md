# DINOv3 runtime remediation v1 result

## Terminal outcome

The bounded deployment remediation passed. One real CUDA preflight
authenticated the complete frozen runtime and loaded `EncPredWM` without
opening any recording or frame.

The root cause was the experimental invocation, not DINO or the model. The
consumed held-out attempt called the Python module directly and bypassed
`ops/jepa_wm.sh`, so `JEPAWM_HOME` was absent and upstream used its
home-directory fallback `/home/ubuntu/dinov3`. A model-load-only differential
reproduced that exact failure in 17 seconds; adding the wrapper environment
loaded the same frozen model successfully. The DINO checkout, weights, cache,
and base checkpoint had remained intact.

The investigation also found that the old smoke command executed `smoke.py`
outside the repository module boundary and failed with
`ModuleNotFoundError: jepa_wm`. That cheap invariant could not have caught the
deployment regression.

## Remediation

- `load_headless_model` now authenticates the runtime before construction.
- The base checkpoint SHA-256 is frozen as `daa69198...f4aa`.
- The approved DINO checkpoint SHA-256 is frozen as `8aa4cbdd...6035`.
- The stale upstream checkpoint name and Torch cache must resolve to that same
  approved artifact.
- JEPA-WM and DINO source trees must match their exact revisions and have no
  tracked or untracked changes.
- `JEPAWM_HOME`, `JEPAWM_OSSCKPT`, and `TORCH_HOME` must share the exact
  installed runtime root.
- Smoke, model-load preflight, and future physical held-out evaluation now have
  explicit repository module/runtime-wrapper entry points.
- The preflight reserves an exclusive, fsynced claim before touching CUDA.

## Authenticated evidence

Commit `a04a30b` passed seven focused remote tests and the full 804-test remote
suite. Independent standards and spec reviews found no remaining deployment
blocker.

The one model-load preflight passed on the NVIDIA L4:

- load time: `11.729 s`;
- peak CUDA allocation: `1.983 GiB`;
- action dimensions: `7`;
- JEPA-WM source revision: `13cf1d9c...83ce0`;
- DINO source revision: `6876159a...751`;
- report: `55f6786b96bfefd95c4d8f6fa450324c364e7a04baac4665e382458f83cc0179`;
- exclusive claim: `782afe7d20d5a090591b4dbd27c275713d5c894578800152dbe2b0615f14b38f`.

The report records `recordings_loaded: false`, `canonical_accessed: false`,
`trained: false`, and `live_action_authorized: false`. The prior canonical
claim and terminal-failure fingerprints remained exactly unchanged.

Both new artifacts match byte-for-byte on the dedicated recovery filesystem.
The verified 16 GiB backup completed at `2026-08-31T21:21:35Z`.

## Authority boundary

Stop. This checkpoint proves deployment readiness only. It does not authorize
a new canonical evaluation, training, Isaac, JEPA action, filming, hardware,
or production. Any new disjoint gate must be separately frozen and authorized.
