# 99designs__aws-vault-1178

- repo: 99designs/aws-vault
- language: go
- difficulty: easy

## Rewritten Prompt

After upgrading AWS Vault to 7.0.0, a profile that uses both a role and web identity credentials no longer validates. Instead of working, it fails with an error like: `aws-vault: error: exec: Error getting temporary credentials: profile 'XXXX' has more than one source of credentials`.

Update the profile validation so web identity profiles can still use a role together with a web identity token file or process, since that combination is required for this credential flow. The behavior should still reject genuinely ambiguous configurations with multiple independent credential sources, but it must not treat the required role + web identity pairing as an error.

## Preserved Requirements

- Profiles using web identity with a role must continue to work.
- The observed error is that a profile is rejected as having more than one source of credentials.
- Validation should still reject truly conflicting multiple credential sources.
- A web identity credential flow requires both a role and a web identity token file/process to be set.

## Removed Noise

- Issue checklist boilerplate about latest release, config attachment, and debug output.
- External source URL and PR/reference links.
- Implementation diagnosis pointing to a specific validation loop and line range.
- Speculation about the motivation for the new validation check.
- Repository path and file-location hints.

## Risk Notes

- The exact precedence rules for when a role should or should not count as an additional source are only described at a high level here.
- The prompt preserves the failure symptom and required compatibility, but not the internal validation structure.
- If the code distinguishes between web_identity_token_file and web_identity_token_process, both should be checked for the same exception behavior.

## Original Prompt

Web identity + role fails to validate in the 7.0.0 release
- [X] I am using the latest release of AWS Vault
- [X] I have provided my `.aws/config` (redacted if necessary)
- [ ] I have provided the debug output using `aws-vault --debug` (redacted if necessary)

After upgrading to 7.0.0, we're seeing errors:

```
aws-vault: error: exec: Error getting temporary credentials: profile 'XXXX' has more than one source of credentials
```

This appears to be related to some new validation code(https://github.com/99designs/aws-vault/blob/ec5e53c91b9990c39c0af69de45f21e436abaa23/vault/config.go#L684-L709), which counts both a `role` and a `web_identity_token_process` as two independent sources. However, the web identity provider _requires_ both of these to be set: https://github.com/99designs/aws-vault/pull/587#issue-616928047

I'm not sure what the motivation with the new validation check was, but I'm guessing we need to only increment the count when a role exists when the web identity file/process is not set.

## Original Interface

No new interfaces are introduced.
