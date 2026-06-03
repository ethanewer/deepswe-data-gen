# 0xs34n__starknet.js-520

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

`Account.getStarkName()` should return an empty string when the queried address has no Starknet name.

If a name exists, it should still return the resolved Starknet ID name as a string. The method should continue to support querying an explicit address and an optional Starknet.ID contract override, while defaulting to the account’s own address and the network’s official Starknet.ID contract when those arguments are omitted.

## Preserved Requirements

- When no Starknet name is associated with the address, `Account.getStarkName()` must return an empty string.
- When a Starknet name exists, `Account.getStarkName()` should return the resolved name string.
- The method may be called with an explicit address argument.
- The method may be called with an optional Starknet.ID contract override.
- When the address argument is omitted, the account’s own address should be used.
- When the contract override is omitted, the official Starknet.ID contract for the current network should be used.

## Removed Noise

- Bug report/template text such as "Describe the bug", "To Reproduce", and "Expected behavior".
- Node version, package version, and network environment details.
- Repository URL and line-specific source link.
- Implementation hint about `useDecoded` concatenating "stark".
- Example reproduction snippet.
- References to tests, PRs, or hidden validation.

## Risk Notes

- The original prompt says the method throws an error if the name cannot be retrieved or is not found, but the bug report explicitly expects an empty string when no name exists. This rewrite preserves the behavioral expectation from the bug report.
- The generated interface notes mention an explicit address parameter; the rewrite keeps that public behavior without over-specifying internal implementation details.

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
