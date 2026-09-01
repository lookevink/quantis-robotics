# Milestone 20 unknown-start reset authentication v6

Run the unchanged reset-only contract with recording identity
`unknown-start-reset-v6-62605` and unused held-out seed `62605`. Preserve all
terminal v1-v5 ledgers, evidence, recovery copies, and spent seeds.

V6 corrects the reset-to-shadow continuity boundary. After the single
reset-time state set and bounded safety-observed settling, pause the Isaac
timeline before reading the authenticated joint, gripper, connector, socket,
camera, scale, light, attachment, collision, and force state. Capture the
observation from that paused state and leave the timeline paused. The later
shadow preflight must independently reauthenticate those exact values before
it may claim an experiment.

The sampling distribution, workspace, model artifacts, controller, force,
collision, tracking, attachment, task, and evidence gates remain unchanged.
The reset applies no action, runs no inference or training, and grants no live
JEPA action or filming authority. Known code, workflow, simulator, and AWS
defects are repaired autonomously; pause only for authenticated
representation/model insufficiency.
