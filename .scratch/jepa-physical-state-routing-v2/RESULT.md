# Physical-state routing probe result

## Terminal outcome

The one permitted authenticated grouped TRAIN-only probe passed. Per the
stopping rule, no residual expert trained, no held-out or canonical recording
was accessed, and no live JEPA action or filming followed.

## Authenticated evidence

- Experiment configuration:
  `f889fcd39704f9d242e6f4e45965fc5d857e6cf2727109aad86b3cc34361f2d5`.
- Base checkpoint:
  `daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa`.
- Frozen control map:
  `e2fea116de2aca46bb9a3e72e3d971e49dfc64936f8fc27469353da102ffa0ed`.
- Exact TRAIN selection:
  `f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`.
- Selected corpus contents:
  `576404f64ac55f47490ef8358eb2121f4dd044f5ab72e396a2817f439fe3d839`.
- Terminal route-probe report:
  `385305e7268e296702f1ecfd5e0104426894a119e15e73567acb03e129801ffa`.

The report is 18,919 bytes. Its live copy and the recovery copy at
`/mnt/quantis-assets/quantis-state` match byte-for-byte. The repository backup
completed and verified at `2026-08-31T16:22:46Z`.

## Frozen gate

| Requirement | Result | Gate |
| --- | ---: | ---: |
| Overall grouped accuracy | 0.987103 | >= 0.95 |
| Retreat recall | 0.992908 | >= 0.98 |
| Advance recall | 0.989780 | >= 0.98 |
| Grasp-attachment accuracy | 1.0 | >= 1.0 |
| Fail-closed fraction | 0.028274 | <= 0.05 |
| Retreat/advance activations in semantic holds | 0 | <= 0 |

All 2,016 TRAIN examples were evaluated in leave-one-recording-out folds. The
semantic roster was 142 hold, 564 retreat, 1,272 advance, and 38 active-other.
`retreat_hold`, `align_hold`, and `seated_hold` each achieved 1.0 accuracy and
zero owned-residual-route activation. The router used only the versioned
physical observation; candidate actions and visual latents were not inputs.

## Authority boundary

This is route-selection feasibility evidence, not action-model evidence. It
permits proposing a separately frozen experiment that fits one final router on
authenticated TRAIN data and trains the two hard-bounded residual experts. It
does not authorize that training automatically, access held-out data, operate
the simulator or robot, promote a JEPA action, or film.
