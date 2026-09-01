# Classifier-corrected unknown-start physical shadow canary v5

Run the unchanged reset-v6 zero-actuation model gate under a fresh terminal
identity. The preceding run produced a complete authenticated observation,
passed its model shadow gate and counterfactual safety gate, then failed only
because terminal evaluation's duplicated schema list omitted v6.

Use one shared unknown-start schema classifier for configuration and terminal
evaluation. Preserve reset seed `62605`, planner seed `237`, all model and
residual artifacts, routing, thresholds, and safety gates. Apply zero actions;
do not train or film.
