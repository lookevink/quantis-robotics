# Milestone 20 unknown-start live action v7 result

Status: terminal `passed` on held-out reset seed `62605`.

V7 applied exactly one frozen JEPA-WM proposal action from the authenticated
unknown-start reset and passed the unchanged projection, freshness, IK,
tracking, force, collision, and attachment gates. It did not grasp, insert,
film, train, or grant production authority.

## Frozen identity

- Source revision: `76c1b199d568a3c55f9107e4e3f4789818dd8890`
- Execution session: `unknown-start-live-action-v7-62605`
- Shadow source: `unknown-start-shadow-canary-v5-62605`
- Reset: `unknown-start-reset-v6-62605`
- Proposal: `contact-grasp-v10-drive-slow-2600_task12_h256_s3000`
- Maximum model actions: `1`
- Applied model actions: `1`

## Live result

- Selected scale: translation `1.0`, rotation `0.25`, gripper `0.25`
- Observation age: `0.008479 s`
- IK position error: `0.000492 mm`
- IK orientation error: `0.564 mrad`
- Translation tracking error: `0.228 mm`
- Rotation tracking error: `0.738 mrad`
- Normalized gripper tracking error: `0.001139`
- Maximum joint tracking error: `0.771 mrad`
- Translation cosine: `0.999943`
- Rotation cosine: `0.999347`
- Maximum contact force: `0 N`
- Collision detected: `false`
- Plug attached: `false`

V5 and V6 are retained terminal pre-execution negatives. Each claimed but
executed zero model actions; their failures exposed reload-to-pause lifecycle
gaps. V7 moved exact reset initialization into atomic candidate capture and
validated it before and after camera capture.

## Authentication and recovery

- Claim: `67159b4237a9e324b07c70fa0f65f3967a5b37cee9134a16e9844a10be1d6274`
- Evaluation: `181e742c3b7c2785dd8eb86690a9a1e3b3f54bc4a54981638bb2e8b6cadd56e2`
- Terminal result: `14dcf231b035a6481e1c79b1e538358383d3c2f0246d7e983ad0e7eae92efeb2`
- Recovery: verified byte-identical `16 GiB` copy
- Remote verification before V7: `873/873` tests passed

This closes only the first unknown-start live-action realization gate. A second
action, multi-step grasp-to-insertion rollout, training, filming, hardware, and
production authority remain outside this experiment.
