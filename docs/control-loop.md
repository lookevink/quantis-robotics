# JEPA control integration

## What can be loaded immediately

The bootstrap loads the official Hugging Face `facebook/vjepa2-vitl-fpc64-256` encoder and verifies that it converts a 64-frame observation window into a normalized latent vector. The control worker below is the next milestone, not part of the current embedding smoke test. It will use that vector to select a stage and measure progress toward prerecorded goal-image windows:

1. `approach handle`
2. `grasped handle`
3. `module extracted`
4. `module in service bin`

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

1. Capture smoke test succeeds on AWS EC2.
2. Scripted Franka task produces synchronized successful and failed episodes.
3. Frozen V-JEPA embeddings distinguish the four task stages.
4. Stage predictions close the loop with Isaac's motion controller.
5. V-JEPA 2-AC is evaluated offline against held-out simulated trajectories.
6. Only then allow the action-conditioned planner to propose bounded simulated actions.
