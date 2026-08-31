# Observed-context routing result

## Terminal outcome

The frozen TRAIN-only experiment failed. Per the stopping rule, no retraining,
fresh canary, canonical evaluation, live JEPA action, or filming followed.

The candidate-independent previous-action router did resolve the earlier
directional counterfactual conflict. It did not satisfy the complete frozen
gate because the attachment segment lost to zero and the positive-X residual
exceeded the declared magnitude ceiling.

## Authenticated evidence

- Experiment configuration:
  `2b57e748a1bf3e60af1e6ad0ec946a1ed502923240ea7b03765c7e808bb3abf6`.
- Frozen control artifact:
  `e2fea116de2aca46bb9a3e72e3d971e49dfc64936f8fc27469353da102ffa0ed`.
- Trained observed-context artifact:
  `27d00e1290fdc14cb5e183bbc4e4c5ce4797ced0abc79c69c1b4626da6b30912`.
- Training report:
  `ebba58816ce08653ef6ba3883117f19f69030d2955ea48170736bda6fd892abf`.
- Terminal TRAIN evaluation:
  `493f42e4246e5141cd85e8854fd7707203a99bba9d8e8b94af619ef9755e5965`.
- Exact TRAIN selection:
  `f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`.

The artifact trained on all 2,016 authenticated TRAIN transitions. Only the
14,336 parameters in the two residual matrices were trainable; the base map
remained bitwise unchanged. Candidate routing used only the previous realized
7D action and remained invariant while scoring every counterfactual candidate.
Training loss fell from `0.0185640` to `0.0106822`, with minimum `0.00719185`.

## Frozen gate result

| Slice | Rollouts | Recorded win rate | Signed order | Mean over-zero improvement |
| --- | ---: | ---: | ---: | ---: |
| all TRAIN | 2,016 | 0.988095 | 0.988095 | 0.001372884 |
| retained | 636 | 0.962264 | 0.962264 | 0.001469931 |
| post | 1,380 | 1.000000 | 1.000000 | 0.001328158 |
| grasp attach | 12 | 0.000000 | 0.000000 | -0.000091481 |
| retreat | 576 | 1.000000 | 1.000000 | 0.001623500 |
| retreat hold | 48 | 0.750000 | 0.750000 | 0.000017449 |
| align | 576 | 1.000000 | 1.000000 | 0.002226017 |
| align hold | 24 | 1.000000 | 1.000000 | 0.000007720 |
| insert | 768 | 1.000000 | 1.000000 | 0.000716770 |
| seated hold | 12 | 1.000000 | 1.000000 | 0.000000565 |

Two conjunctive requirements failed:

1. Every semantic segment required positive mean improvement over zero.
   `grasp_attach` was negative and the recorded action lost in all 12 cases.
2. Every full-route residual/base embedding ratio had to be at most `0.15`.
   The positive-X route reached `0.175160` on a recorded candidate and
   `0.174430` on an X-opposed candidate. The negative-X maximum was `0.119134`.

All aggregate, retained, post, retreat, alignment, insertion, and main signed
ordering thresholds otherwise passed. This is materially different from the
candidate-dependent router failure: retreat, alignment, and insertion each
achieved perfect signed ordering because the same observed-context route was
used for every candidate comparison.

## Interpretation and boundary

This result does not demonstrate a DINOv3 or JEPA-WM representation
insufficiency. It shows that observable, candidate-independent regime routing
can represent the main backward and forward command distinctions. The
remaining negative is localized to the attachment transition and to excessive
positive-route residual magnitude. Any constrained residual parameterization,
attachment-boundary treatment, regularization change, or new gate belongs to
a separately frozen milestone.

Model loading took `14.788 s`, corpus encoding `212.114 s`, and training
`4,436.430 s`. Terminal evaluation separately took `17.119 s` to load,
`210.150 s` to encode, and `982.358 s` to score. The repository recovery
workflow verified a 16 GiB copy at `/mnt/quantis-assets/quantis-state` on
`2026-08-31T08:20:59Z`; the trained artifact, training report, and terminal
evaluation hashes match their recovery copies exactly.
