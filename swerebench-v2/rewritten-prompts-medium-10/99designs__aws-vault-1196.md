# 99designs__aws-vault-1196

- repo: 99designs/aws-vault
- language: go
- difficulty: medium

## Rewritten Prompt

`source_profile` should work when a profile assumes a role from another profile, including when the referenced profile uses SSO credentials. At the moment, running a command with an assumed-role profile appears to ignore `source_profile` and falls back to the default profile’s credentials instead.

Make profile resolution honor `source_profile` consistently so the selected source profile is used for credential lookup and temporary credential creation. The behavior should match AWS CLI expectations for the same config setup.

The profile data should be fully resolved from the config, including inherited settings from the source profile, and the resulting temporary credentials provider should be appropriate for the resolved source profile.

## Preserved Requirements

- Using an assumed-role profile should not always fall back to the `default` profile.
- `source_profile` must be honored when resolving credentials for a profile.
- A profile can use `source_profile` to point to an SSO-based profile.
- Profile resolution should include inherited settings from the source profile.
- Temporary credential provider selection should reflect the resolved source profile, including SSO-based sources.
- Behavior should align with AWS CLI expectations for equivalent config.

## Removed Noise

- Issue template checkboxes about using the latest release, providing config, and providing debug output.
- Version-specific complaint about `7.1.2-Homebrew`.
- Narrative thanks and subjective wording.
- Example command outputs and redacted account details.
- Full `.aws/config` sample with specific account IDs, URLs, and regions.
- Full debug log output.
- References to `aws-vault --debug` and `get-caller-identity` as diagnostic evidence.
- Generated interface notes and implementation hints.
- Method, function, file, and type names from internal notes.
- References to tests, assertions, and hidden validation.

## Risk Notes

- The original report suggests `default` is used instead of the intended source profile, but the exact failing branch is not fully specified.
- The config example mixes SSO source credentials with an assume-role target; preserving that interaction is important.
- The task likely depends on correct resolution of chained profile settings, not just credential provider selection.

## Original Prompt

source_profile is not working correctly.
- [v] I am using the latest release of AWS Vault
- [v] I have provided my `.aws/config` (redacted if necessary)
- [v] I have provided the debug output using `aws-vault --debug` (redacted if necessary)

It seems that the `source_profile` option is not working in 7.1.2-Homebrew version.
When I use with the `source_profile` with `dev`, the `get-caller-identity` always uses the `default` profile.

AWS CLI is working well, so I hope this feature is working same with the cli in the future.
Thanks.

```
> aws-vault exec test -- aws sts get-caller-identity
{
    "UserId": "ARxxxx:xxxx,
    "Account": "3701xxxx",
    "Arn": "arn:aws:sts::3701xxxx:assumed-role/xxxx"
}

> aws sts get-caller-identity --profile test
{
    "UserId": "ARxxxx:test",
    "Account": "3604xxxx",
    "Arn": "arn:aws:sts::3604xxxx:assumed-role/xxxx"
}

> vi ~/.aws/config
[profile test]
external_id=1d88xxxx
role_arn=arn:aws:iam::3604xxxx:role/xxxx
role_session_name=test
source_profile=dev
region=ap-northeast-2

[profile dev]
sso_session=common
sso_account_id=2160xxxx
sso_role_name=AdministratorAccess
region=ap-northeast-2
output=json

[default]
sso_session=common
sso_account_id=3701xxxx
sso_role_name=AdministratorAccess
region=ap-northeast-2
output=json

[sso-session common]
sso_start_url=https://xxxx.awsapps.com/start
sso_region=ap-northeast-2
sso_registration_scopes=sso:account:access
```

And here is the debug log.
```
aws-vault exec test --debug -- aws sts get-caller-identity
2023/03/20 09:16:32 aws-vault 7.1.2-Homebrew
2023/03/20 09:16:32 Using prompt driver: terminal
2023/03/20 09:16:32 Loading config file /Users/lamanus/.aws/config
2023/03/20 09:16:32 Parsing config file /Users/lamanus/.aws/config
2023/03/20 09:16:32 [keyring] Considering backends: [keychain]
2023/03/20 09:16:32 [keyring] Querying keychain for service="aws-vault", keychain="aws-vault.keychain"
2023/03/20 09:16:32 [keyring] Found 7 results
2023/03/20 09:16:32 profile test: using SSO role credentials
2023/03/20 09:16:32 Setting subprocess env: AWS_REGION=ap-northeast-2, AWS_DEFAULT_REGION=ap-northeast-2
2023/03/20 09:16:32 [keyring] Querying keychain for service="aws-vault", keychain="aws-vault.keychain"
2023/03/20 09:16:32 [keyring] Found 7 results
2023/03/20 09:16:32 [keyring] Querying keychain for service="aws-vault", keychain="aws-vault.keychain"
2023/03/20 09:16:32 [keyring] Found 7 results
2023/03/20 09:16:32 [keyring] Querying keychain for service="aws-vault", account="sso.GetRoleCredentials,dGVzdA,aHR0cHM6Ly9vcHNub3cuYXdzYXBwcy5jb20vc3RhcnQ,-62135596800", keychain="aws-vault.keychain"
2023/03/20 09:16:32 [keyring] No results found
2023/03/20 09:16:32 [keyring] Querying keychain for service="aws-vault", account="oidc:https://xxxx.awsapps.com/start", keychain="aws-vault.keychain"
2023/03/20 09:16:32 [keyring] Found item "aws-vault oidc token for https://xxxx.awsapps.com/start (expires 2023-03-20T17:12:17+09:00)"
2023/03/20 09:16:33 Got credentials ****************U7QJ for SSO role AdministratorAccess (account: 3701xxxx), expires in 11h59m58.598194s
2023/03/20 09:16:33 [keyring] Querying keychain for service="aws-vault", keychain="aws-vault.keychain"
2023/03/20 09:16:33 [keyring] Found 7 results
2023/03/20 09:16:33 [keyring] Querying keychain for service="aws-vault", keychain="aws-vault.keychain"
2023/03/20 09:16:33 [keyring] Found 7 results
2023/03/20 09:16:33 [keyring] Checking keychain status
2023/03/20 09:16:33 [keyring] Keychain status returned nil, keychain exists
2023/03/20 09:16:33 [keyring] Keychain item trusts keyring
2023/03/20 09:16:33 [keyring] Adding service="aws-vault", label="aws-vault session for test (expires 2023-03-20T21:16:32+09:00)", account="sso.GetRoleCredentials,dGVzdA,aHR0cHM6Ly9vcHNub3cuYXdzYXBwcy5jb20vc3RhcnQ,1679314592", trusted=true to osx keychain "aws-vault.keychain"
2023/03/20 09:16:33 Setting subprocess env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
2023/03/20 09:16:33 Setting subprocess env: AWS_SESSION_TOKEN
2023/03/20 09:16:33 Setting subprocess env: AWS_CREDENTIAL_EXPIRATION
2023/03/20 09:16:33 Exec command aws sts get-caller-identity
2023/03/20 09:16:33 Found executable /opt/homebrew/bin/aws
{
    "UserId": "ARxxxx:xxxx,
    "Account": "3701xxxx",
    "Arn": "arn:aws:sts::3701xxxx:assumed-role/xxxx"
}
```

## Original Interface

Method: ConfigLoader.GetProfileConfig(self, profileName string) 
Location: vault/config_loader.go (method of type *ConfigLoader) 
Inputs: profileName – name of the profile to resolve; uses the loader’s File (parsed config) and ActiveProfile fields. Returns the resolved *ProfileConfig and an error if the profile cannot be found or is malformed. 
Outputs: (*ProfileConfig, error) – the concrete profile configuration or an error describing missing/invalid profile. 
Description: Retrieves a fully‑resolved profile configuration (including inheritance via source_profile) from the loaded config file. Used by callers to obtain the profile data needed for credential providers.

Function: LoadConfig(configPath string) 
Location: vault/config.go (top‑level function) 
Inputs: configPath – file system path to an AWS config file. Parses the INI‑style file and returns an internal representation. 
Outputs: (*ConfigFile, error) – the parsed configuration structure or an error if the file cannot be read or parsed. 
Description: Reads and parses an AWS configuration file into a ConfigFile object that can be queried by ConfigLoader. Called directly by tests to load the temporary config file.

Function: NewTempCredentialsProvider(cfg *ProfileConfig, keyring *CredentialKeyring, allowRefresh bool, requireMfa bool) 
Location: vault/provider.go (factory function) 
Inputs: cfg – the profile configuration to base the provider on; keyring – wrapper around a keyring implementation for storing/retrieving master credentials; allowRefresh – whether the provider may refresh credentials automatically; requireMfa – whether MFA handling should be enforced. 
Outputs: (aws.CredentialsProvider, error) – a concrete credentials provider (e.g., SSORoleCredentialsProvider, AssumeRoleProvider, etc.) or an error if the profile cannot be satisfied. 
Description: Constructs the appropriate temporary credentials provider for a given profile, handling SSO, AssumeRole, GetSessionToken, web‑identity, and credential‑process flows. The test asserts that with a source_profile pointing to an SSO profile the returned provider is an *SSORoleCredentialsProvider.
