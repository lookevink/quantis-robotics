# Milestone 20 unknown-start reset authentication v3 terminal negative

Recording `unknown-start-reset-v3-62602`, held-out seed `62602`, terminated
safely during semantic evidence validation. It performed exactly one reset
state-set and applied zero actions; no model was loaded or evaluated.

The terminal claim and failure are immutable in
`unknown_start_reset_v3_claims`. Primary and recovery copies match exactly:

- claim SHA-256: `9ba4d700e247a6214c282bddb46887afbd547709752773a5684675146159b74e`
- failure SHA-256: `ddc670ad494eaab27c0d8ad428f388cc2cecaf73ee19905f4928a94047c0217d`
- wrist frame SHA-256: `754c58b7d6022a4bff94daa326641aa3f7d689683047ce786d29faf746509317`

The old aggregate validator discarded the rejected field name and value.
A read-only inspection of the simulator state left by v3 authenticated the
sample/workspace mapping, control-frame bounds, authored arm target, camera,
socket scale, finite settled joints, gripper width, and current 0 N contact.
The only unpreserved state was the peak safety reading accumulated during the
16 settling updates.

V4 does not weaken or change any gate. It makes each evidence invariant
individually observable and writes the complete rejected evidence plus named
failures before recorder cleanup. Seed 62602 remains spent and forbidden.
