# 0xpolygonhermez__zkevm-node-1321

- repo: 0xPolygonHermez/zkevm-node
- language: go
- difficulty: medium

## Rewritten Prompt

Make the node work without an account when it is not sending sequences or proofs. Today, an account/.keystore is required even for node types that only synchronize data and redirect transactions to the trusted sequencer, which creates unnecessary setup friction.

Allow the Etherman account to be optional so read-only nodes can operate without a signing key. In read-only mode, operations that genuinely need the account should fail cleanly rather than forcing account configuration up front.

Preserve the public behavior of `etherman.GetPublicAddress`: it returns the node’s public Ethereum address as `(common.Address, error)`, and it may return an error when the client is in read-only mode. Also preserve `Client.IsReadOnly`, which reports whether the Etherman client was created without a signing key.

## Preserved Requirements

- Nodes that only synchronize data and forward transactions must work without a configured account or .keystore file.
- The Etherman account must be optional instead of mandatory for every node type.
- Read-only operation must be supported for nodes that do not send sequences or proofs.
- Account-dependent operations must be guarded so they fail cleanly in read-only mode rather than requiring configuration upfront.
- `etherman.GetPublicAddress` must remain available and return `(common.Address, error)`; it may error in read-only mode.
- `Client.IsReadOnly` must remain available and return a boolean indicating whether the client was created without a signing key.
- A client created without a signing key is read-only.
- Higher-level components may use the read-only indicator to avoid private-key-dependent operations.

## Removed Noise

- Issue-template style discussion and casual wording.
- The implementation suggestion about adding checks at the beginning of methods.
- The speculative rationale phrased as a proposal rather than a requirement.
- The informal mention of `wdyt @arnaubennassar?`.
- References to source locations in the generated notes.

## Risk Notes

- The task implies behavior changes around account-dependent operations; care is needed to keep existing signatures and error handling consistent.
- `GetPublicAddress` failing in read-only mode must remain explicit so callers can distinguish missing-account cases from other errors.
- `IsReadOnly` should reflect how the client was constructed, not transient runtime state.

## Original Prompt

Remove account requirement for nodes that are not sending sequences or proofs
The current implementation requires an account to be configured(aka requires a .keystore file) regardless of the node type.

This will cause unnecessary friction for users.

We could have a Read Only Etherman for nodes that are only synchronizing data and redirecting TXs to the trusted sequencer.

In order to achieve this, we basically need to add a check at the beginning of the methods that really depend on the account and make the account to be optional.

wdyt @arnaubennassar?

## Original Interface

Method: etherman.GetPublicAddress(self)
Location: aggregator/interfaces.go & etherman/etherman.go
Inputs: none
Outputs: (common.Address, error) – returns the node’s public Ethereum address; may return an error (e.g., when the Etherman client is in read‑only mode).
Description: Retrieves the address used for L1 transactions. In read‑only mode the call fails, allowing callers to handle the missing account explicitly.

Method: Client.IsReadOnly(self)
Location: etherman/etherman.go (type *Client)
Inputs: none
Outputs: bool – true if the Etherman client was created without a signing key (read‑only), false otherwise.
Description: Indicates whether the Etherman instance can send L1 transactions. Used by higher‑level components (e.g., EthTxManager) to guard operations that require a private key and to surface ErrIsReadOnlyMode when attempts are made in read‑only mode.
