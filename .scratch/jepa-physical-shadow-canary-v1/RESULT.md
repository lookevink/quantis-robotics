# Physical residual known-start shadow canary v1 result

## Terminal outcome

The one authorized zero-actuation evaluation is terminally **failed**. The
exclusive claim bound config fingerprint
`16f2b88f35f04eeace7de429b81945d3c5d50d29d1c4038424418ba7acf9da25`
to session `physical-shadow-canary-12601` before capture.

The known-start capture completed with `0 N` contact force and no collision.
The direct response used proposal fingerprint
`fa17063e36b687a9f2696133e85850bae19e552fc4266ea660f2e0e3274f87aa`.
The physical residual shadow search scored 256 candidates and reduced latent
energy from `0.1497276723` to `0.1413725317`, but its first action failed the
frozen direction gate: cosine `0.6170790099` was below `0.9`. The separate
Isaac counterfactual safety projection passed at translation/gripper scale
`1.0` and rotation scale `0.25`.

No action was applied. No control result, execution evidence, experimental
candidate binding, training, filming, hardware, or production authority was
created. The worker stopped. The terminal failure report and its recovery copy
are byte-identical with SHA-256
`4d5d6305c05f9cd4a3c62798ae18bbf20b9905c896c7efffb8c487dbbe8eccbf`.
Its exclusive claim fingerprint is
`a72c27373fb4593d030f8b86dc0d2ebe80c674b8a8626a9f31674c23096eef19`.

## Boundary

This exposes a new deployment-planning failure class: held-out physical-model
action wins and counterfactual safety did not guarantee that latent search
would preserve the active grasp direction. The canary cannot be retried.
Diagnosis and any remediation belong to a separately frozen offline milestone;
live action and filming remain closed.
