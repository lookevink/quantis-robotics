# DINOv3 runtime remediation v1

## Terminal result

Repair and authenticate the JEPA-WM/DINOv3 deployment boundary without opening
any recording. Produce one new model-load preflight report, verify its recovery
copy, and stop before proposing another canonical gate.

## Confirmed failure

The consumed held-out attempt invoked `python -m` directly with only
`JEPA_WM_REVISION`. Upstream therefore used its home-directory fallback and
looked for `/home/ubuntu/dinov3/hubconf.py`. The same model-load-only command
reproduced that failure in 17 seconds without recordings. Adding the existing
wrapper exports loaded `EncPredWM` successfully; the pinned DINO source,
revision, weights, and base checkpoint were intact.

The prior `ops/jepa_wm.sh smoke` path also failed before model loading because
it executed `smoke.py` as a file outside the repository import boundary.

## Frozen remediation

1. Make `load_headless_model` fail closed before construction unless
   `JEPAWM_HOME`, `JEPAWM_OSSCKPT`, `TORCH_HOME`, the expected JEPA source,
   DINO `hubconf.py`, base checkpoint, and stale-name DINO checkpoint are
   present and mutually consistent.
2. Run smoke and model-load preflight through `python -m` from the repository
   root.
3. Add an explicit runtime-wrapper entry for the physical residual held-out
   evaluator; bare direct Python is not an authenticated deployment workflow.
4. Persist exactly one preflight at
   `/home/ubuntu/docker/jepa-wm/checkpoints/quantis_physical_state_residual_v1/runtime-preflight-v1.json`.

## Gate and stopping rule

Require focused and full tests, two-axis review, a real CUDA model load with
zero recordings, a report stating `recordings_loaded: false` and
`canonical_accessed: false`, and a byte-identical dedicated recovery copy.

Do not open TRAIN, held-out, or canonical recordings. Do not train, run Isaac,
issue a JEPA action, film, or create a new canonical access claim. A pass only
establishes deployment readiness for a separately frozen future proposal.
