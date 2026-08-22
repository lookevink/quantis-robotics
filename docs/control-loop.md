# JEPA control integration

## What can be loaded immediately

The AWS workflow now loads the official Hugging Face
`facebook/vjepa2-vitl-fpc64-256` encoder and converts either recorded wrist or
presentation-camera observations into a normalized latent vector. Demo capture
guarantees the model's 64-frame input window, keeps both views synchronized with
robot state, and stores the embedding beside the persistent recording. The live
CUDA proof produced a 1,024-dimensional wrist-camera vector from
`demo-20260822T194745Z`.

The control worker below is the next milestone. It will use that vector to select
a stage and measure progress toward prerecorded goal-image windows:

1. `approach cable`
2. `grasped cable`
3. `aligned with socket`
4. `plug seated`

The intended resulting stage/subgoal is executed by Isaac Sim's motion generation controller as an online visual feedback loop, not open-loop playback.

## Why V-JEPA 2-AC is not the first step

The released action-conditioned checkpoint is a useful experiment but is not a drop-in robot policy. It predicts future latents conditioned on Franka/DROID actions. Control additionally requires:

- matching its action and proprioception normalization;
- sampling candidate actions;
- rolling the predictor forward;
- scoring rollouts against a goal latent;
- cross-entropy-method refinement;
- applying only the first action and replanning;
- enforcing workspace, velocity, collision, and force limits.

The official entry point is:

```python
import torch

encoder, predictor = torch.hub.load(
    "facebookresearch/vjepa2",
    "vjepa2_ac_vit_giant",
)
```

Use Meta's `energy_landscape_example.ipynb` as the executable reference for preprocessing and energy scoring. Do not send its raw sampled actions directly to a robot or simulator articulation without a safety/controller layer.

## Recommended process boundary

Keep Isaac Sim and JEPA in different processes:

```text
Isaac Sim                          JEPA worker
---------                          -----------
RGB frame window  ───────────────► encoder
robot state                       stage/goal score
candidate action metadata         optional AC rollout
                  ◄────────────── subgoal or bounded 7D delta
motion controller
safety limits
```

The first interface should return a discrete subgoal and confidence:

```json
{
  "observation_id": 1234,
  "subgoal": "module_extracted",
  "confidence": 0.87,
  "goal_similarity": 0.74
}
```

Later, the action-conditioned interface can return a bounded 7D command:

```json
{
  "observation_id": 1234,
  "action": {
    "delta_xyz": [0.002, 0.0, 0.0],
    "delta_rpy": [0.0, 0.0, 0.0],
    "gripper": 1.0
  },
  "predicted_goal_energy": 0.18
}
```

## Dataset contract

Each episode contains:

- ordered RGB frames at a fixed capture rate;
- one 7D action per transition;
- joint/end-effector state sampled at the same point;
- task and camera metadata;
- success/failure outcome.

WebRTC is only for viewing. Capture directly from Replicator and the controller because WebRTC compression and network latency break action/frame alignment.

## Milestones

1. [x] Capture smoke test succeeds on AWS EC2.
2. [x] Scripted Franka success run produces 64 synchronized wrist and
   presentation-camera observations with robot state.
3. [x] Frozen V-JEPA encodes a real wrist-camera run on the AWS GPU and persists
   its normalized latent vector.
4. [ ] Record goal windows and failed/perturbed runs for the four task stages.
5. [ ] Train and validate a stage/progress head over frozen V-JEPA embeddings.
6. [ ] Close stage predictions around Isaac's bounded motion controller.
7. [ ] Evaluate V-JEPA 2-AC offline against held-out simulated trajectories.
8. [ ] Only then allow the action-conditioned planner to propose bounded
   simulated actions.
