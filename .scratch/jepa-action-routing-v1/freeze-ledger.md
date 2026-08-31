# Runtime-command routing freeze ledger

Freeze date: 2026-08-30 (America/Los_Angeles)

- Authority: offline training/evaluation and one fresh scripted canary capture.
- Local starting HEAD: `4a2facbfadc73255021d8275b6b0007a55ac902a`.
- AWS account: `686410906008`, verified through profile `quantis`.
- Instance: `i-0ee3209a8972f008b`, running `g6.2xlarge`.
- Existing TRAIN/canonical identities: inherited unchanged from
  `../jepa-action-conditioning-48h/freeze-ledger.md`.
- Fresh canary: `contact-insertion-v10-drive-slow-72601-held-00`, seed 72601;
  name confirmed absent from live and incomplete recording roots before freeze.
- Seed 72600 is design evidence only and may not be reused for selection.
- Base checkpoint SHA-256: `daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa`.
- Frozen control map SHA-256: `e2fea116de2aca46bb9a3e72e3d971e49dfc64936f8fc27469353da102ffa0ed`.
- Experiment configuration SHA-256: `98fc2af503919d52a3853d3181bf007d56360136e5c1d27cd1a08a4db18bf66d`.
- Safety, controller, recording, and task thresholds: unchanged.
- Live JEPA action, filming, hardware, and production authority: closed.

The experiment configuration fingerprint and produced artifact/evidence hashes
are appended only after their files exist and authenticate.

## Terminal result

- Software checkpoint: `cd8a439cf39e8dcece8d641ae37df3f96d769758`,
  with 749 tests passing on the remote JEPA environment.
- Fresh canary seed 72601: valid 284-frame contact-aware held-out recording;
  manifest SHA-256
  `3bc0f2e796916012a8094d2d8026fe165ae939d8a8b0a6fb8b804326092ef153`,
  telemetry SHA-256
  `371fa92b8df939e9119742922af47594e038582057a5e324d28469d9c163f246`,
  selected-input SHA-256
  `056b08827e26b1925a3ca4d1cd96f6ed6ea0a879d6231f9cab17c1ad29b8505e`.
- Router artifact SHA-256:
  `45326210f5a47f74a9008670e9bf0be03b3ef40955b3c9af79017588d9b79c30`.
- Training report SHA-256:
  `0a8452a1440e8448b3223d6dc6cfec6f9dacf4aaebcea226ebc1e9ed269d7c3b`.
- Control evaluation SHA-256:
  `19c1a04f74b5ae9da90bf4b6cfc8572200952ab0590be954ca702639ccc66f00`.
- Router evaluation SHA-256:
  `c12ce02a0ac6e6c274d6965650b40c5acf618f7f81abb42d1a582f0f03d7d72c`.
- Canary summary SHA-256:
  `3469d4e73cbed26d63aad4751bf0ebf88b31d651b34feb7a5a678d02a2dc93a4`.
- Terminal outcome: `router_failed`. Router overall/retained/post win rates
  were `0.988095`/`0.962264`/`1.0`, but retreat and alignment signed-order
  fractions were `0` and `0.020833`; no artifact was selected.
- Canonical offline authority: false. Live action authority: false. Canonical
  seeds 12600 and 12601 were not evaluated.
- Recovery: live and recovery copies of the router, training report, both
  canary reports, and summary match byte-for-byte in the verified 16 GB state
  backup.
