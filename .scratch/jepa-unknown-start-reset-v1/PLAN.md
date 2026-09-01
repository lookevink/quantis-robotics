# Milestone 20 unknown-start reset contract v1

Freeze a deterministic reset distribution before any new live evaluation. The
distribution owns reserved seeds `62600..62699`, uses the existing held-out
exploration sampler with the contact-insertion socket scale, permits direct
state setting once during reset only, and permits only drive-based motion after
initialization. A valid realization is unattached, collision-free, at zero
contact force, and has replayed zero recorded prefix frames.

This contract is not motion authority. After source review and full regression
tests, a separate one-shot claim may select one unused seed. The first live use
must be zero-actuation reset/observation authentication. If that exposes a new
failure class, preserve one reconstructible negative and stop the experiment.
Do not modify the system and recursively retry inside the same milestone.
