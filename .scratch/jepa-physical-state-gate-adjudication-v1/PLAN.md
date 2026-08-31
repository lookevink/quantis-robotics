# Physical-state residual gate adjudication v1

## Terminal result

Produce one authenticated adjudication of the immutable physical-state
residual TRAIN evaluation. Correct only the duplicated floating-point ratio
predicate. Do not load the JEPA model, recordings, or simulator; do not train,
rescore, access held-out data, or authorize live control.

## Frozen correction

The source report is identified by SHA-256
`a5527c50eeb4a3223b74779584f7dad1f6edc29121534a778abf9147bf4d6bdd`.
Every stored task, router, hold, artifact, corpus, and authority field remains
immutable. Replace raw `maximum_residual_ratio <= 0.15` with
`maximum_residual_ratio <= 0.15 + 1e-6`, matching the experiment's existing
explicit bound check. The tolerance covers numerical roundoff only; a ratio of
`0.15001` must still fail.

## Stopping rule

Write one adjudication sidecar and recovery-back it, then stop. A pass makes
the existing artifact eligible to be proposed for a separately frozen
disjoint held-out gate. It does not itself authorize that gate, retraining,
Isaac, a JEPA action, filming, hardware, or production use.
