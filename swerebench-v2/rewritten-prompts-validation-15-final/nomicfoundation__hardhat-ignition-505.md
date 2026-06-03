# nomicfoundation__hardhat-ignition-505

- repo: NomicFoundation/hardhat-ignition
- language: ts
- difficulty: hard

## Rewritten Prompt

Before executing a module, ensure nonce state is synchronized for every account that may be used. The nonce sync step must consider all senders already present in the deployment state, plus every future in the module that has a `from` address, and also the default sender when a future omits `from`.

The behavior should still use `getNonceSyncMessages(jsonRpcClient, deploymentState, ignitionModule, accounts, defaultSender, requiredConfirmations)` and return a `Promise` of journal messages. It should detect when a pending transaction’s nonce is lower than the network’s latest nonce and report it as replaced by the user, detect when a pending transaction’s nonce is higher than the network’s latest nonce and report it as dropped, and prevent execution when any sender still has unconfirmed pending transactions beyond the allowed confirmation threshold.

## Preserved Requirements

- The public function name `getNonceSyncMessages` must remain available.
- The callable signature must accept `jsonRpcClient`, `deploymentState`, `ignitionModule`, `accounts`, `defaultSender`, and `requiredConfirmations` in that order.
- The function must return a `Promise<Array<OnchainInteractionReplacedByUserMessage | OnchainInteractionDroppedMessage>>`.
- Nonce synchronization must consider every sender already used in the deployment state.
- Nonce synchronization must also consider every future in the module whose `from` address has not yet been reflected in the deployment state.
- Nonce synchronization must include the default sender for futures whose `from` is undefined.
- User-provided `accounts` are used to resolve `from` values that are indexes.
- The logic compares pending transaction nonces against the chain’s latest and pending transaction counts.
- A pending transaction with a lower nonce than the network’s latest nonce is treated as replaced by the user.
- A pending transaction with a higher nonce than the network’s latest nonce is treated as dropped.
- Execution must not proceed if any sender has unconfirmed pending transactions exceeding the required confirmation count.

## Removed Noise

- Issue-template wording and summary framing.
- The external GitHub URL and line references.
- Implementation diagnosis about checking only previously used accounts.
- References to PRs, tests, files, and internal source locations.
- Comments about hidden execution flow or solution hints.

## Risk Notes

- The exact distinction between 'latest nonce' and 'pending transaction count' should remain consistent with existing chain-query behavior.
- The prompt preserves the observable requirement to include futures' `from` addresses and the default sender, but does not specify internal traversal details.
- The return type is narrowed to the two journal message variants noted in the interface notes; if the implementation currently emits other message shapes, that behavior would need careful reconciliation.

## Original Prompt

Not all accounts' nonces are sync before execution
Before starting to execute a module we run [`getNonceSyncMessages`](https://github.com/NomicFoundation/ignition/blob/316ec19b1df431a665e523cefd88ef4db7f0bf31/packages/core/src/new-api/internal/new-execution/nonce-management.ts#L42C23-L42C43) to sync the nonces from the previous with the actual ones, and to prevent us from execution if there are user transactions without enough confirmations.

We only check the accounts that we have already used, but we have to check every transaction that's either a `from` in a future, or the default sender.

## Original Interface

Function: getNonceSyncMessages(jsonRpcClient: JsonRpcClient, deploymentState: DeploymentState, ignitionModule: IgnitionModule<string, string, IgnitionModuleResult<string>>, accounts: string[], defaultSender: string, requiredConfirmations: number) → Promise<Array<OnchainInteractionReplacedByUserMessage | OnchainInteractionDroppedMessage>>
Location: packages/core/src/internal/execution/nonce-management/get-nonce-sync-messages.ts
Inputs:
- jsonRpcClient – a JsonRpcClient used to query latest block number and transaction counts.
- deploymentState – the current DeploymentState containing pending on‑chain interactions.
- ignitionModule – the IgnitionModule being executed; its futures are scanned for any future “from” addresses that have not yet been reflected in the deployment state.
- accounts – list of user‑provided accounts (used to resolve “from” values that are indexes).
- defaultSender – the fallback sender address when a future’s `from` is undefined.
- requiredConfirmations – number of block confirmations required before a transaction is considered final.
Outputs:
- A Promise that resolves to an array of journal messages, each being either:
  * OnchainInteractionReplacedByUserMessage – when a pending transaction’s nonce is lower than the network’s latest nonce (user sent a replacement transaction).
  * OnchainInteractionDroppedMessage – when a pending transaction’s nonce is higher than the network’s latest nonce (transaction was dropped).
Description:
Computes nonce‑synchronisation messages before execution. It gathers pending nonces per sender from the deployment state, augments the set with all senders required by futures (including the default sender), compares them against the chain’s latest and pending transaction counts, and returns messages indicating replaced or dropped interactions, or throws if any sender has unconfirmed pending transactions that exceed the allowed confirmations. This ensures the execution engine only proceeds with a nonce‑consistent state.
