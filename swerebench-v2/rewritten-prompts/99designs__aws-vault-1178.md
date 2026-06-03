# 99designs__aws-vault-1178

- repo: 99designs/aws-vault
- language: go
- difficulty: easy

## Rewritten Prompt

After upgrading to version 7.0.0, a profile that uses both a role and web identity credentials is rejected with an error saying the profile has more than one source of credentials.

The expected behavior is that this combination should be allowed when web identity authentication is configured, since the role and web identity settings are both required for that flow.

Update the credential-source validation so it does not incorrectly treat that supported configuration as conflicting sources.

## Preserved Requirements

- Profiles using both a role and web identity credentials must be accepted.
- The validation must not report this supported configuration as having more than one source of credentials.
- The change should address the regression introduced in version 7.0.0.

## Removed Noise

- Issue checklist markdown about using the latest release and providing config/debug output.
- The specific error stack text and quoted command output.
- The GitHub code link and line reference.
- The PR/issue reference and discussion link.
- Speculation about the exact implementation detail of the validation logic.

## Risk Notes

- The exact set of web identity configuration forms that should count as credential sources is only implied; keep behavior aligned with the intended role + web identity flow.
- Avoid broadening acceptance beyond the supported role/web identity case.

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
