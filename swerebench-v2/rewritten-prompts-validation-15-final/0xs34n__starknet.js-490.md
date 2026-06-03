# 0xs34n__starknet.js-490

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

Add `simulateTransaction` to `Account` as a public account method, matching the account-facing behavior of fee estimation. It should accept one call or an array of calls, plus optional fee-estimation details such as an explicit nonce and a block identifier, and return a transaction simulation containing both the execution trace and a fee estimation with a suggested max fee.

The method must use the account’s signer to simulate the call(s), so it is only available where signature-based simulation is supported. If simulation is not enabled on the RPC, it should fail with the same kind of explicit error used for unsupported fee-estimation flows, rather than silently proceeding.

Keep the public API name `Account.simulateTransaction` and preserve its input/output shape and compatibility with existing account behavior.

## Preserved Requirements

- Expose `Account.simulateTransaction` as a public method on Account.
- Support a single `Call` or an array of `Call` objects.
- Each call must include `contractAddress`, `entrypoint`, and optional `calldata` defaulting to an empty array.
- Accept optional `estimateFeeDetails` with `nonce` and `blockIdentifier`.
- If `nonce` is omitted, use the account’s current nonce.
- Return a `Promise<TransactionSimulation>`.
- The result must include `trace` and `fee_estimation`.
- `fee_estimation` must be an `EstimateFee` object extended with `suggestedMaxFee` derived from the simulated fee.
- Simulation must be signed with the account’s signer.
- If the RPC does not support simulation, throw an explicit error instead of attempting the call.
- Preserve the `Account.simulateTransaction` public symbol.

## Removed Noise

- Repository and language metadata.
- GitHub PR reference and external URL.
- Issue/template-style introductory phrasing.
- Implementation hint about using a specific endpoint name.
- Exact throw-string example quoted in the prompt.
- Location/path details from generated interface notes.
- References to tests, PRs, or hidden validation.

## Risk Notes

- The original text mixes a desired placement on `Account` with an implementation detail about RPC support; the rewritten prompt keeps the observable behavior but avoids over-specifying the internal mechanism.
- The phrase 'just like estimateFee' is preserved only as a behavioral comparison; if the repository already has nuanced account/RPC support patterns, the agent may need to align with existing method conventions.

## Original Prompt

Refactor `simulateTransaction` to Account
Regarding:
https://github.com/0xs34n/starknet.js/pull/466

`simulateTransaction` should be defined on Account as signature is mandatory for simulating a transaction, just like `estimateFee`

We can throw error for RPC just like estimateFeeBulk

`
throw Error(“Simulate Transaction is not enabled on RPC for now”)
`

## Original Interface

Method: Account.simulateTransaction(self, calls: AllowArray<Call>, estimateFeeDetails?: EstimateFeeDetails) → Promise<TransactionSimulation>
Location: src/account/default.ts (added in Account class)
Inputs:
- **calls** – a single Call object or an array of Calls. Each Call must include:
  - `contractAddress` (string) – address of the contract to invoke.
  - `entrypoint` (string) – name of the contract entrypoint.
  - `calldata` (string[]; optional, defaults to empty) – calldata for the call.
- **estimateFeeDetails** (optional) – object allowing:
  - `nonce` (number | string | BN) – explicit nonce to use; if omitted the account’s current nonce is fetched.
  - `blockIdentifier` – block identifier passed through to the provider.
Outputs:
- **Promise\<TransactionSimulation\>** – resolves to an object containing:
  - `trace` – the transaction trace (`TransactionTraceResponse`).
  - `fee_estimation` – an `EstimateFee` object extended with `suggestedMaxFee` (BN) derived from the simulated fee.
Description: Simulates the provided contract call(s) using the Sequencer’s `simulate_transaction` endpoint, automatically signs the invocation with the account’s signer, and returns both the execution trace and a fee estimation (including a suggested max fee). Use this when you need to preview the outcome and cost of a transaction without actually sending it on‑chain.
