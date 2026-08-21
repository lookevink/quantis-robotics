---
name: quantis-aws
description: Use AWS CLI and infrastructure tools for this repository with the quantis profile, authenticated to AWS account 686410906008. Apply whenever work in this project reads or changes AWS resources.
---

# Quantis AWS

Use the local AWS CLI profile `quantis`. The expected AWS account ID is
`686410906008`.

## AWS CLI

Run AWS CLI commands through `scripts/quantis-aws` in this skill directory. The
wrapper fixes `AWS_PROFILE=quantis` and verifies the account with STS before
executing the requested command.

Do not use the default profile, infer the account from an environment variable,
or fall back to another profile. Do not modify `~/.aws`, credentials, or the
profile configuration unless the user explicitly asks.

If authentication is unavailable or STS returns any other account, stop and
report the failure. Never continue an AWS mutation under an unverified identity.

## Other AWS-aware tools

For Terraform, CDK, SDK scripts, and similar tools that do not run through the
wrapper:

1. Verify identity with `scripts/quantis-aws sts get-caller-identity`.
2. Set `AWS_PROFILE=quantis` for the tool invocation.
3. Keep the tool's normal approval and deployment boundaries; selecting this
   account does not itself authorize creating, changing, or deleting resources.

Do not commit access keys, session tokens, or credential files.
