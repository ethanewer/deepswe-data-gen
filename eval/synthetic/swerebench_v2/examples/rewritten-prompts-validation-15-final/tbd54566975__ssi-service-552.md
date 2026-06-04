# tbd54566975__ssi-service-552

- repo: TBD54566975/ssi-service
- language: go
- difficulty: hard

## Rewritten Prompt

When creating or fetching credentials through the Credential API, the returned credential object should expose only the credential’s own ID. Do not prefix the ID with `http://localhost`; the response should contain the credential identifier itself.

Keep the credential container behavior consistent with that contract: `Container.IsValid()` should consider a container valid when it has a non-nil credential with an `ID` and contains either a data-integrity credential or a JWT credential. The container’s own `ID` field should not be required for validity.

Also preserve the constructor behavior for `NewCredentialContainerFromJWT` and `NewCredentialContainerFromMap`: both should build containers with an empty top-level `ID`, while the embedded `Credential.ID` holds the credential identifier.

## Preserved Requirements

- Credential API responses must return the credential ID without an `http://localhost` prefix.
- The `id` field in create and GET credential responses should contain only the credential identifier.
- `Container.IsValid() bool` must return true only when the container has a non-nil credential whose `ID` is set and the container contains either a data-integrity credential or a JWT credential.
- `Container.ID` should not be required for validity.
- `NewCredentialContainerFromJWT(credentialJWT string) (*Container, error)` must return a container with an empty top-level `ID`, store the JWT in `Container.CredentialJWT`, and place the credential identifier in `Container.Credential.ID`.
- `NewCredentialContainerFromMap(credMap map[string]any) (*Container, error)` must return a container with an empty top-level `ID` and a populated `Credential` whose `ID` holds the credential identifier.

## Removed Noise

- Issue template sections such as "Describe the bug", "To Reproduce", and "Expected behavior".
- External screenshot/attachment reference.
- Repository-specific source location notes.
- Implementation diagnosis phrased as internal container ID duplication details beyond the observable behavior.
- References to bug labels and supporting material metadata.

## Risk Notes

- The request concerns both API output formatting and internal container validity/constructor semantics; keep both aspects aligned so the top-level container ID is not accidentally reintroduced.
- The exact credential identifier format is not further specified beyond removing the `http://localhost` prefix, so avoid introducing any new normalization rules.

## Original Prompt

[Bug] ID includes `localhost`
**Describe the bug**
When using the Credential API, the service returns the `id` field with `http://localhost`

**To Reproduce**
1. Create a credential with the service.
2. Observe the `id` field in the response. It should start with `localhost`
3. Call GET to get credentials from the service.
4. Observe the `id` field in the response. It should start with `localhost`

**Expected behavior**
The `id` field should only return the credential ID

**Supporting Material**
<img width="970" alt="Screenshot 2023-06-15 at 3 22 18 AM" src="https://github.com/TBD54566975/ssi-service/assets/102400653/61a8fb4b-057c-454b-9d87-e71c05bedbf8">

## Original Interface

Method: Container.IsValid() bool
Location: internal/credential/model.go
Inputs: None (operates on the Container instance)
Outputs: Returns true only when the container holds a non‑nil credential whose `ID` field is set and the container contains either a data‑integrity credential or a JWT credential. This change ensures that the container’s own `ID` field is no longer required for validity and that the credential’s internal ID is used instead.

Function: NewCredentialContainerFromJWT(credentialJWT string) (*Container, error)
Location: internal/credential/model.go
Inputs:
- credentialJWT (string): a signed JWT representing a Verifiable Credential.
Outputs:
- *Container: a new Container where `Container.ID` is left empty (previously populated with `cred.ID`) and `Container.Credential.ID` contains the credential’s identifier. The JWT string is stored in `Container.CredentialJWT`.
Description: Parses a JWT credential and returns a Container that relies on the embedded credential’s ID rather than duplicating it on the container.

Function: NewCredentialContainerFromMap(credMap map[string]any) (*Container, error)
Location: internal/credential/model.go
Inputs:
- credMap (map[string]any): a map representation of a Verifiable Credential.
Outputs:
- *Container: a new Container with an empty `ID` field (previously set to `cred.ID`) and a fully populated `Credential` whose `ID` holds the credential identifier.
Description: Constructs a Container from a credential map, decoupling the container’s top‑level ID from the credential’s own ID and ensuring the internal credential ID is the source of truth.
