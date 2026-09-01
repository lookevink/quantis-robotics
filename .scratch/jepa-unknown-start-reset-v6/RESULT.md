# Result: passed

- Recording: `unknown-start-reset-v6-62605`
- Held-out seed: `62605`
- Source revision: `04d3435622721e746c0415a302885ace4daa4802`
- Contract: `3e216b58...1489b`
- Sample: `67d09973...3427c`
- Terminal result: `70a8fba8...a6d58`
- Evidence: `3cea7cc3...61e38`
- Recovery verified: yes, 16 GiB

The reset used one initialization state set, zero prefix replay, and zero
actions. Evidence was read and captured after pausing Isaac, and the timeline
remained paused for the shadow handoff. The realized state was unattached,
collision-free, and at exactly `0 N`; arm tracking error was `0.564 mrad` and
gripper tracking error was `0.011 mm`.

This passes only the reset/continuity prerequisite. It does not authorize
training, live JEPA action, filming, hardware, or production use.
