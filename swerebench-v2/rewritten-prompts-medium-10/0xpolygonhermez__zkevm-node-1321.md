# 0xpolygonhermez__zkevm-node-1321

- repo: 0xPolygonHermez/zkevm-node
- language: go
- difficulty: medium

## Rewritten Prompt

Make the node work without a configured account when it is not sending sequences or proofs. In that mode, the node should still be able to synchronize data and forward transactions, but it must not require a keystore or signing key at startup.

When the node is configured without a private key, expose that it is in read-only mode and make operations that need an account fail explicitly instead of assuming one exists. The node should also provide its public Ethereum address when available, and return an error if that address cannot be obtained because the client is read-only.

Ensure the behavior is consistent across the components that interact with Etherman so that account-dependent actions are guarded, while non-signing node modes continue to operate normally.

## Preserved Requirements

- Nodes that are not sending sequences or proofs must not require a configured account/keystore.
- Read-only nodes must still support synchronizing data and redirecting transactions to the trusted sequencer.
- Account-dependent operations must be guarded and fail explicitly when no signing key is configured.
- The system must expose whether Etherman is in read-only mode.
- The system must provide the node’s public Ethereum address when available, and return an error when it is unavailable in read-only mode.

## Removed Noise

- Issue-template phrasing and conversational aside.
- Reference to a specific user mention.
- Speculation about implementation details such as adding checks at the beginning of methods.
- Explicit mention of .keystore file as a requirement artifact.
- PR/test references and benchmark metadata.
- Repository and language header.

## Risk Notes

- The prompt implies a read-only Etherman mode; implementations should preserve existing signing behavior for nodes that do send sequences or proofs.
- The public-address API behavior should remain compatible with callers that need to handle the missing-account case explicitly.
- Avoid relaxing account checks for operations that still require signing, especially L1 transaction submission.

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
