# Unknown-start physical shadow canary v1 result

Status: **terminal apparatus negative; not model evidence**.

- session: `unknown-start-shadow-canary-v1-62604`
- config fingerprint: `c691b5efeadcd773df82050f24e388e0134b941ed5f100cb12fff498064e0a53`
- claim fingerprint: `604f408847c2a8985baf44e46a62ba8df1112d4ee4d9824ec6f0aca0cead9ff7`
- failure fingerprint: `b89691ae62534042474b5d19e1371a346935d6079aa08b0d5ac20492ae0e2026`
- terminal phase: `capture`
- error: `unknown-start shadow capture requires a paused timeline`

The run stopped at its first live precondition, before creating a control
session or capturing an image. It performed no model request, shadow search,
counterfactual safety evaluation, action, training, or filming. Primary and
16 GiB recovery copies of the claim and failure match byte-for-byte, and no
session directory exists. The immutable identity is not reusable.

The owning layer is the experiment/app lifecycle: reset v5 stopped the
timeline, but that process state did not remain frozen until this later
milestone. Version 2 moves an explicit pause plus complete reset-state
reauthentication before the exclusive experiment claim.
