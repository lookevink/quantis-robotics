Type: task
Status: resolved
Blocked by: 03

## Question

Can treatments B, C, and D complete once under identical frozen sampling,
optimization, evidence, and resource contracts, yielding authenticated
artifacts without mid-run experimental mutation?

## Answer

Yes. B, C, and D each completed exactly once, serially, against the exact 12
TRAIN roster and frozen configuration
`a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14`.
Every artifact was independently loaded after its atomic write and matched its
sidecar fingerprint, treatment specification, training-selection fingerprint
`f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74`,
and balanced `1008/1008` retained/post sampler counts.

| Treatment | Artifact SHA-256 | Parameters | Initial loss | Final loss | Minimum loss | Training seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B | `7bb9b54682c245f580f8e4922ceed53e33e21ffcc35ab80e01b11e90378a0754` | 7,168 | 0.017280983 | 0.013490250 | 0.010219557 | 3,628.797 |
| C | `65272ab9a4b1f5369486dd829585739f0119f3f186dcd8d07072512d94ca9908` | 40,192 | 0.017280983 | 0.007791738 | 0.005516719 | 3,632.968 |
| D | `243c80ea0c2cd56ff0ce8c76426fb90cdc79aa898d816abd400b0d14b420610a` | 21,504 | 0.017280983 | 0.011364294 | 0.006990391 | 3,747.505 |

There was no mid-run repair, retry, held-out access, concurrent training, or
simulator action. A malformed post-run validator import and one unsupported
summary CLI argument both failed before inspecting or writing experimental
state; neither invalidated an artifact or report.
