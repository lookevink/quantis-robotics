# Milestone 20 unknown-start reset live failure

## Terminal outcome

The one-shot reset-authentication experiment terminally failed on seed `62600`
before recorder initialization, direct state setting, inference, or motion. It
applied zero actions. The canonical failure has `retry_authorized: false`, and
the claim and failure were independently matched on the recovery volume.

- Source revision: `9765506dfd2bd91538ee7c3ccc7b9fc38d7c8c63`
- Runtime source fingerprint:
  `9b650dfc1f953a69fce1ced0199779d0b4d4293d819e0ff224c085e817a1d8a3`
- Contract fingerprint:
  `bd745c96700f544aa93cbf1262a3bbde4bb8d550ddebc21a5c89917e762a13aa`
- Sample fingerprint:
  `47098a1d49eab2ed93e723486eadba10a88c31e9fdb4c857585960a97d7efc58`
- Claim fingerprint:
  `43872081936d8e33cc5f92bf8dacf464e8f196240dd8457f3eca40850fa31089`
- Failure fingerprint:
  `95152c8ab5c47e575c1668c3f782ac9bef30a9d5c0d07e891d4dc2555a03008f`
- Recovery backup: verified, 16 GB

The runtime stopped at `unknown-start variant did not realize its plan`.
Read-only inspection of that same authored session showed:

- Wrist translation: `[0.16019, 0.077121, -0.189619]`, exactly the base mount
  plus the sampled camera offset.
- Socket scale: `[1.05, 1.05, 1.05]`, exactly the sampled scale.
- Requested light-exposure delta: `0.370041`.
- Authored float exposure delta: `0.37004101276397705`.
- Absolute exposure error: approximately `1.2764e-8`.

The frozen contract permits `1e-6` light-realization error. The reset-specific
USD readback adapter duplicated that check with a stricter `1e-9` threshold,
so it rejected a realization the authoritative contract would accept. The
owning layer is the reset adapter's prevalidation, not Isaac, the robot, the
world model, or the frozen contract.

## Boundary

Do not repair and retry seed `62600` in this milestone. A separate follow-up
milestone may remove the duplicated tighter precheck (or delegate tolerance
checking exclusively to `UnknownStartResetEvidence.validate`) and must use a
new exclusive recording identity and unused reserved seed. No model action,
training, filming, hardware, or production authority was earned here.
