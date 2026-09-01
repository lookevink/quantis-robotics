# Milestone 20 unknown-start reset authentication v4 terminal negative

Recording `unknown-start-reset-v4-62603`, held-out seed `62603`, terminated
safely on the single named invariant `workspace_bounds`. It performed exactly
one reset state-set and applied zero actions; no model was loaded or evaluated.

The authenticated negative proves every other evidence invariant passed. The
connector physics readback was `x = -0.025600001215934753 m` against the
zero-width bound `-0.0256 m`, a difference of approximately 1.22 nanometers.
The contract already carried a `1e-5 m` (10 micrometer) position-realization
tolerance, but `contains()` did not apply it to physical workspace readback.

Primary and recovery copies match exactly:

- claim SHA-256: `ae0ce03d96bfdd989f2627f2234848ac8edcaebaeee8ff41617ec31a785232a0`
- failure SHA-256: `0ba8d22a3f2e5392dcdf354148a01accd9317ef5e207449e62d5f6f3c1cd4341`
- negative SHA-256: `efa47031eae3ecc35089dc2ff75b525cc0660599199168ee0c255c35dfe4d2c2`
- wrist frame SHA-256: `0685a8ce9f98597705b917e77f8a8e58586e94a60d5477ff50b094b46c57ae7e`

V5 applies the already frozen realization tolerance consistently to observed
workspace containment. It does not enlarge the sampled distribution. The
contract advances to schema v3 and fingerprints this containment semantics;
seed 62603 remains spent and forbidden.
