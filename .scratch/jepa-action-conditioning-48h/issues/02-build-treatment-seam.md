Type: task
Status: resolved
Blocked by: 01

## Question

Can one versioned adapter-family seam express balanced linear, zero-initialized
nonlinear residual, and oracle-separated diagnostic treatments while retaining
strict artifact identity and baseline equivalence?

## Answer

Yes. `jepa_wm/action_conditioning.py` defines strict versioned B/C/D families,
exact zero-residual installation, scoped oracle routes, fingerprint-bound
serialization, and trainable-parameter boundaries. The frozen experiment
runner in `jepa_wm/action_conditioning_experiment.py` rejects non-TRAIN fitting
inputs, changed configuration bytes, reused outputs, incomplete windows, and
treatment/artifact disagreement. Balanced sampling and signed-X negatives live
in `jepa_wm/action_conditioning_training.py`.

Twenty-six focused tests pass remotely with PyTorch, including artifact
round-trip, identity change detection, route cleanup, stratum coverage,
six-coordinate preservation, frozen-config identity, and non-promotion of the
oracle diagnostic. No training or held-out evaluation ran while resolving this
gate.
