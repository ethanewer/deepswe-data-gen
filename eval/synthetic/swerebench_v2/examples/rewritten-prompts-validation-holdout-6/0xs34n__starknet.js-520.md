# 0xs34n__starknet.js-520

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

`Account.getStarkName()` should return an empty string when the address has no associated Starknet name.

The method already supports an optional `address` argument and an optional `StarknetIdContract` override. If no address is passed, it should use the account’s own address. If the name cannot be found for the resolved address, the method should resolve to `""` rather than returning `"stark"` or another placeholder value. Keep the existing return type and behavior for successfully resolved names.

## Preserved Requirements

- `Account.getStarkName(address?: BigNumberish, StarknetIdContract?: string)` remains available.
- If `address` is omitted, the account’s own address (`this.address`) is used.
- If `StarknetIdContract` is omitted, the provider resolves the official Starknet.ID contract for the current network.
- The method returns `Promise<string>`.
- When a Starknet name is found, it should be returned as the resolved name string (for example, `vitalik.stark`).
- When no Starknet name exists for the address, the method should resolve to an empty string.
- The bug concerns the observable result of `account.getStarkName()` for an address with no associated Starknet name.

## Removed Noise

- Issue template sections like "Describe the bug", "To Reproduce", and "Desktop".
- Environment/version details such as Node version, package version, and network.
- The external GitHub URL and line reference.
- Internal implementation diagnosis about `useDecoded` and string concatenation.
- Explicit references to the provided reproduction snippet as a required implementation path.
- Mentions of bug-report metadata and prompt scaffolding.

## Risk Notes

- The original interface note says the method throws an Error if the name cannot be retrieved or is not found, but the bug report expects an empty string when no name exists. The rewritten prompt prioritizes the concrete expected behavior from the issue.
- It is preserved that successful lookups should still return the resolved Starknet name string; only the missing-name case changes.

## Original Prompt

getStarkName() should return empty string when no starkname found
**Describe the bug**

Currently `account.getStarkName()` returns "stark" when address has no stark name.
This is due to `useDecoded` function in starknetId utils which concat "stark" at the end of the result from the call to the the starknet.id naming contract, even if it's empty.

https://github.com/0xs34n/starknet.js/blob/b0f4b7690b471b1c8edbbebaa4f9b64feb124d00/src/utils/starknetId.ts#L22


**To Reproduce**

```
const account = new Account(provider, address, ec.genKeyPair())
const result = await account.getStarkName()
```

**Expected behavior**
`account.getStarkName()` should return an empty string when address has no associated stark name.

**Desktop (please complete the following information):**

- Node version `19.4.0`
- StarkNet.js version : `4.15.0`
- Network `devnet`

## Original Interface

Method: Account.getStarkName(address?: BigNumberish, StarknetIdContract?: string)
Location: src/account/default.ts
Inputs:
- address (BigNumberish) – optional; if omitted the account’s own address (`this.address`) is used.
- StarknetIdContract (string) – optional contract address; when undefined the provider resolves the official Starknet.ID contract for the current network.
Outputs: Promise&lt;string&gt; – resolves to the resolved Starknet name (e.g., “vitalik.stark”). Throws an Error if the name cannot be retrieved or is not found.
Description: Retrieves a Starknet ID name for a given address, defaulting to the account’s address. The method now accepts an explicit address parameter, enabling callers to query names for arbitrary addresses while retaining the optional contract override.
