# Milestone 20 unknown-start reset v2 live failure

V2 terminally failed after one reset state-set, 16 settling updates, and one
camera capture, but before recording finalization. It applied zero actions and
never loaded or queried a model. The claim and failure match the verified
16 GB recovery backup; the 281,541-byte partial wrist frame is preserved.

- Source revision: `6d28d34771198d06a9a3f3cdba811c168630bb95`
- Runtime fingerprint:
  `bde9597a5dd6efd30b89966a7db9496182019b694c28f7ffcd9909943c654747`
- Sample fingerprint:
  `280c80d2e8a0f313e83989609835bebb77a1d769a8ccdb65857d623c56e4b65e`
- Claim fingerprint:
  `71709339921138cc105859280b041c4f6277fdc4b2dfdd2fe74f96165bbaf111`
- Failure fingerprint:
  `1163de6da16b2073a4952569c8bf91f78798f62d0120a415dce261396e3d1c84`

Read-only diagnosis of the stopped state found two evidence-binding defects:

1. The sampled arm offset was authored exactly as the reset target, but the
   evidence compared it to physical joints after settling. Gravity/drive
   settling produced up to `0.003148335` rad of observed difference versus a
   realization tolerance of `1e-5` rad. Authored realization and observed
   settled joints are distinct facts; the evidence schema already records the
   latter separately.
2. The workspace bound describes the gripper control frame near X `0.25` m,
   but runtime supplied the Panda hand-link origin at X `0.349665` m. The
   synchronized recording separately provides `gripper_frame_world_position`,
   which is the coordinate the contract intended.

Connector and socket positions matched the sampled scene offset, the camera
offset matched, both contact sensors read zero/no-contact, and the grasp joint
was disabled. These are runtime evidence-binding bugs, not representation or
model failures. Seed `62601` must not be reused.
