# Physical shadow corrected-planner offline replay v1 result

## Terminal outcome

The single authenticated replay passed. It used the immutable request, state,
direct response, old shadow request, old shadow result, and shadow-safety files
from `physical-shadow-canary-12601`; all six retained their frozen hashes.

With the same proposal, physical residual, observation, CEM seed, and
256-candidate budget, the corrected planner retained objective improvement
`0.0080960924` and raised first-action cosine from the failed replay's
`0.6170790099` to `0.9961398888`. The unchanged `0.9` direction gate and the
complete shadow gate passed. Authority remained `shadow_only`.

Evaluation SHA-256 `adb88ea025f30706b4a53da39e17c27b67a623d55023827340718c492fe65c7e`
matches its recovery copy. Terminal result SHA-256
`e541ba373dd78138ace00a1968b89e831456d0dcd80159cdc7d051e89cec7cce`
also matches its recovery copy. The worker stopped and Isaac remained stopped.

No capture, simulator safety evaluation, action application, training, filming,
hardware, or production authority was exercised. This opens only a separately
frozen proposal for a new zero-actuation canary on a different held-out start;
the consumed canary remains terminal and cannot be retried.
