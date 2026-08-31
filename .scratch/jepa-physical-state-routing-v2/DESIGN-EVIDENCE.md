# Physical router design evidence

This is architecture-selection evidence, not the frozen one-shot probe result
and not action-model evidence. It used all 2,016 exact TRAIN transitions, the
semantic hold labels in this milestone, leave-one-recording-out grouping,
TRAIN-fold-only normalization, confidence `0.75`, residual ratio `0.15`,
weight decay `0.0001`, and seed `237`. It did not load DINO/JEPA, access
held-out data, train residuals, or run the simulator.

The original architecture comparison exposed quaternion-component subtraction
as a feature-definition bug during review. Those scores are superseded. After
replacing it with a normalized, sign-canonical relative quaternion, the
selected smallest passing architecture was rerun without retuning:

| Hidden/steps/lr | Accuracy | Retreat recall | Advance recall | Attach | Fail-closed | Worst fold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64 / 1000 / .003 | .987103 | .992908 | .989780 | 1.0 | .028274 | .934524 |

The corrected candidate produced zero retreat/advance activations in
`retreat_hold`, `align_hold`, and `seated_hold` and still meets every frozen
gate. The exact 64x64 design remains frozen in `experiment-config.json`.
