# 0xs34n__starknet.js-490

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

Refactor the account API so transaction simulation is available on `Account`, not as a standalone behavior. Simulating a transaction requires an account signature, similar to fee estimation.

The account should expose a `simulateTransaction` method that accepts a single call or an array of calls, plus optional simulation details such as a nonce and block identifier. It should simulate the call(s) through the RPC and return both the execution trace and a fee estimation, including a suggested maximum fee derived from the simulation.

If the connected RPC does not support transaction simulation, the method should fail with an error indicating that simulation is not enabled on the RPC yet.

## Preserved Requirements

- `simulateTransaction` should be defined on `Account` because signature is mandatory for simulating a transaction, like `estimateFee`.
- The method accepts a single `Call` or an array of `Call`s.
- The method accepts optional simulation details including nonce and block identifier.
- The method returns a transaction simulation result containing both a trace and fee estimation.
- The fee estimation includes a `suggestedMaxFee` derived from the simulated fee.
- The account should automatically sign the invocation when simulating.
- If RPC support is unavailable, the method should throw an error indicating simulation is not enabled on the RPC yet.

## Removed Noise

- GitHub PR reference and external URL.
- Repository metadata and language metadata.
- Issue/template phrasing and the "Regarding" section.
- Direct implementation hint about matching `estimateFeeBulk`.
- Exact throw statement text formatted as code.
- Generated interface notes with specific type signatures and location details.
- References to hidden tests or patch/PR context.

## Risk Notes

- The original prompt is somewhat ambiguous about whether the public method should be added, moved, or renamed; the rewrite preserves the behavior requirement without asserting a specific internal implementation.
- The exact shape of the simulation and fee objects is not restated in full to avoid over-specifying beyond the user-visible requirement.

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
