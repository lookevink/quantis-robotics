# Frozen experiment ledger

Freeze date: 2026-08-30 (America/Los_Angeles)

## Source and authority

- Local branch: `main`
- Local HEAD: `dcb27bae9b383d1dd4fb3fbcc8c839d2f520bd26`
- Upstream JEPA-WM source revision:
  `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- AWS account: `686410906008`, verified through the `quantis` profile
- EC2 instance: `i-0ee3209a8972f008b`, `g6.2xlarge`, running
- Authority: offline adapter training and offline evaluation only
- Running recording jobs at freeze: none
- Remote source is a synchronized deployment tree without `.git`; synchronized
  experiment file hashes below match the local files.

## Frozen artifacts and evidence

- Base checkpoint SHA-256:
  `daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa`
- Control adapter SHA-256:
  `e2fea116de2aca46bb9a3e72e3d971e49dfc64936f8fc27469353da102ffa0ed`
- Control training-selection fingerprint:
  `f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`
- Control training-config fingerprint:
  `c3f42b98570237fc39127b2a784877b4b10d577d623de8130788074e7083a8d3`
- Corpus roster report SHA-256:
  `8f4036569e3c392e634d3fb7188a092bce9246d49563ea83a2a63e890cffa84b`
- Control readiness report SHA-256:
  `ceb74a568060e9ae491e96f29f7efe865c222ff503798d7d93b8acea03abd6da`
- Control training report SHA-256:
  `0c600d4a1e95a1e43b0d257770353a5ef179c98cebeed4ed95065c49fedfb21d`
- Latent diagnostic report SHA-256:
  `69186e22c109d728bf119dd1ed75f5a84eccc3374719bea731cc1a50eb53ae36`
- Known control result: 336 rollouts, positive mean improvement
  `0.00006202256329180229`, win rate `0.7023809523809523`, failed only the
  unchanged 0.75 minimum-win-rate gate.

## Synchronized uncommitted diagnostic inputs

- `jepa_wm/insertion_latent_diagnostic.py`:
  `8a95a4f6b5626a28a8ee0ed07cad33b37d7692372e01903c62acfc18d9e81121`
- `tests/test_jepa_wm_insertion_latent_diagnostic.py`:
  `78a9868e75d186224e0008e0d5c9ba0caff39a32d577c35789b7eb450512b34e`
- `docs/research/jepa-wm-retreat-failure.md`:
  `a3fd07626a06d86c173f256b694d2209af159d32ae77241c390ed3772477c1bb`
- Unrelated pre-existing `error.log` and `supabase/` content is excluded and
  must remain untouched.

## Authenticated recording roster

Every entry independently passed the contact-aware 284-frame contract with
0 N maximum connector force and four seated observations. SHA-256 values are
for each recording's `manifest.json`.

| Recording | Split | Seed | Manifest SHA-256 |
| --- | --- | ---: | --- |
| contact-insertion-v10-drive-slow-2600-train-00 | train | 2600 | `f68dc64f813aaba7cdf549353bc8a182a3e25d2ee2527ed8c734a0a147255142` |
| contact-insertion-v10-drive-slow-2600-train-01 | train | 2601 | `b07bdbc2f99dfb12027469b72f66106553c427fed4557f176b4b3dfe3db979c0` |
| contact-insertion-v10-drive-slow-2600-train-02 | train | 2602 | `90d4626758da3a4f76752d48de118070fe02e97c360ec235a25746b297406a33` |
| contact-insertion-v10-drive-slow-2600-train-03 | train | 2603 | `605a0591c4a4b63d13f3f5b514548222a7dfc74534938e460bea7bcb648de184` |
| contact-insertion-v10-drive-slow-2600-train-04 | train | 2604 | `668f55e402f057760d21bc9488d8a52cedded4ef21fec65766f4d57f8a85eb01` |
| contact-insertion-v10-drive-slow-2600-train-05 | train | 2605 | `953474217600cca07985510bcf3faee5c9993cd260d89bd03115d05e2053855c` |
| contact-insertion-v10-drive-slow-2600-train-06 | train | 2606 | `96bd4d5f9ad9ddca53d9273d7bedc5119dc82365662ccdc092a601d75c3b247c` |
| contact-insertion-v10-drive-slow-2600-train-07 | train | 2607 | `629a4424e99c1e26978f0153c4fa7ae475b8c2c7be60895378f3ab987fe92b88` |
| contact-insertion-v10-drive-slow-2600-train-08 | train | 2608 | `061771dbdb45714151914f9a17a2ff7469c9c6b77cdcf9d3a28716d44a95a890` |
| contact-insertion-v10-drive-slow-2600-train-09 | train | 2609 | `4dc9a8b3f9f7b0b5567552939dcd6fcbcd9b46bc8e3cfeb302db4faa8d81ca24` |
| contact-insertion-v10-drive-slow-2600-train-10 | train | 2610 | `217c3e46542eee6c2648e427d25cdbcbb6209b5b8e5c75db962c77b4de816f42` |
| contact-insertion-v10-drive-slow-2600-train-11 | train | 2611 | `74ce25284a81423f985c81a5059ff4204a4f7bae50f625f58e8cfd61935021a8` |
| contact-insertion-v10-drive-slow-2600-held-00 | canonical held_out | 12600 | `dd504100969f706e22795f6c7dc85578f11b0fc53dd5050560a8976c725a5879` |
| contact-insertion-v10-drive-slow-2600-held-01 | canonical held_out | 12601 | `3d5d64eb6efa8a3b18331ad016f1b502be074decd40f411a6aadf72bf4e070f5` |
| contact-insertion-v10-drive-slow-72600-held-00 | development canary | 72600 | `9f9fa0b9fb22fd49c1ba6e045100f6f74588c2655e0b3830bc4b68dc0eccfa86` |

## Recovery boundary

- Dedicated recovery backup timestamp: `2026-08-30T03:04:31Z`
- Source and recovery SHA-256 match for the base checkpoint, control adapter,
  and all 15 manifest files above.
- Control reports and corpus roster exist in the recovery tree with matching
  sizes from the verified backup.

## Frozen configuration

The canonical experiment configuration is
[`experiment-config.json`](experiment-config.json). Its SHA-256 is recorded
after canonical JSON validation and becomes the immutable comparison identity.

- Experiment configuration SHA-256:
  `a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14`
- Superseded preflight fingerprint:
  `b82627965542f0f0ee64a9b493ca6418cfe7553e8641899d20b4817a43cbde7e`.
  It was invalidated before training because it applied signed-X margins to
  inactive hold actions, where recorded and negative actions are identical.

## Terminal experiment evidence

- Frozen terminal outcome: `frozen_dynamics_or_representation_blocker`
- Selected treatment/artifact: none
- B artifact SHA-256:
  `7bb9b54682c245f580f8e4922ceed53e33e21ffcc35ab80e01b11e90378a0754`
- C artifact SHA-256:
  `65272ab9a4b1f5369486dd829585739f0119f3f186dcd8d07072512d94ca9908`
- D diagnostic artifact SHA-256:
  `243c80ea0c2cd56ff0ce8c76426fb90cdc79aa898d816abd400b0d14b420610a`
- Canary summary SHA-256:
  `6324665b9d22f370851bbd3d2b4fa9930df8a141f08497279d89f68d5e0e4650`
- Stable recovery backup timestamp: `2026-08-31T00:10:14Z`
- Source and recovery hashes match for all three artifacts and sidecars, all
  four seed-72600 reports, and the terminal summary.
- Canonical held-out seeds 12600 and 12601 were not evaluated by any model.
