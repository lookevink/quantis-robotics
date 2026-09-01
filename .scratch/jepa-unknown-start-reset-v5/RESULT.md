# Milestone 20 unknown-start reset authentication v5 result

Status: **passed**, independently recovery-verified.

- recording: `unknown-start-reset-v5-62604`
- held-out seed: `62604`
- source revision: `3479d857123daff3ce355555292d6fa3936e0078`
- contract: `quantis.unknown_start_reset_contract.v3`
- contract fingerprint: `3e216b58ad9543e876a3ac3ee2624add614c02c1860fcc2af7d03d436461489b`
- runtime-source fingerprint: `ce131abd5a67507f42d5aff7256046798065939950af55755a5584b580364c33`
- sample fingerprint: `5f4a001ca2304be10d01789b90e0564e397996384d73d07f2efdb0d8072a4441`
- evidence fingerprint: `43c1ac4c81515d4921e1070dc866313256183eb54a85a1de3896674810903ccf`
- claim fingerprint: `893f7ee5d86ac7deee763520079bf26d7e5987ebecf16d0566624958e46c882b`
- recovery marker fingerprint: `19c1a49ea6d9200338f0d16f8e8368a1260af7c86626c48ccef5ba3995a89235`

The run used exactly one reset state-set, zero prefix replay frames, zero
applied actions, and no model inference. The plug was unattached, collision
was false, and contact force was exactly `0.0 N`. The explicit
`right_gripper_control_frame` position was
`[0.2513500578, -0.2409477374, 1.4725119471] m`, inside the frozen workspace.
The largest authored arm-offset readback error was approximately
`1.16e-7 rad`, and the connector physics readback differed from its sampled
position by at most approximately `4.53e-8 m`; both are well inside the
unchanged realization tolerances.

The independent post-workflow audit compared the claim and all five canonical
artifacts byte-for-byte between primary and recovery, verified the primary
terminal result, verified that recovery carries only the non-passing
`RECOVERY_VERIFIED.json` marker, and confirmed that no v5 failure or negative
artifact exists.

This result authenticates reset only. Its terminal payload explicitly leaves
training and filming unauthorized and earns no live model-action authority.
