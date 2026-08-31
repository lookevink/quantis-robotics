# Causal context routing probe

## Terminal result

Produce one authenticated, grouped TRAIN-only answer to a single question:
can the frozen context representation plus causal robot state predict the next
motion route without candidate actions? Stop after the route probe. Do not
train residual experts, evaluate a canary or canonical split, authorize live
JEPA action, or film.

## Redesign under test

- Pool the frozen visual context latent and combine it with the observed pose
  and previous realized 7D action.
- Predict `hold`, `retreat`, `advance`, or `active_other` once per context.
- Never use candidate actions, recorded future actions, scripted phase,
  context index, or seed as router inputs. The recorded future horizon is a
  TRAIN label only.
- Treat low confidence and `active_other` as fail-closed base routing.
- Reserve residual experts for retreat and advance. Holds and active-other do
  not silently become owned motion routes.
- Enforce the `0.15` residual/base embedding trust region inside the action
  conditioner by construction; it is not a terminal-only optimizer target.

## Ownership

The post-attachment manipulation adapter owns `grasp_attach` as the causal
attachment-to-retreat handoff plus retreat, alignment, and insertion. Hold
segments are passthrough and must preserve the frozen base result exactly.
Preflight must reject any owned failing slice that has no trainable future
route before encoding or training.

## Probe

Authenticate every exact TRAIN recording against the strict contact-insertion
evidence contract, then bind preflight and probe to a content digest covering
the manifests, telemetry, and selected context/target frame bytes. Reject a
structurally untrainable owned slice before loading or encoding the model.

Use leave-one-recording-out folds over the exact 12 TRAIN recordings. Train the
small route classifier only on eleven recordings and score the twelfth. Report
overall, per-route, per-segment, per-fold, confidence, and fail-closed metrics.
No candidate action may enter the router. A pass also requires zero retreat or
advance residual-route activations in every passthrough hold segment.

## Stopping rule

A pass permits a separately frozen bounded-residual training experiment. A
failure ends this probe and requires either richer causal observations or more
transition evidence. It does not permit in-probe retuning.
