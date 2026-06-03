# 0xpolygonhermez__zkevm-node-1044

- repo: 0xPolygonHermez/zkevm-node
- language: go
- difficulty: medium

## Rewritten Prompt

Unsigned transaction handling needs to use the correct nonce and the network’s configured L2 chain ID. Right now the transaction encoder relies on a hardcoded chain ID, but it should use the chain ID from configuration instead. The nonce is already being handled by the JSON-RPC flow, so the remaining requirement is that unsigned transactions are encoded with the correct chain ID from the active network config.

Update the relevant transaction encoding and configuration flow so that unsigned transactions produced by the node carry the network-specific L2 chain ID rather than a constant value. The public config surfaces should expose the L2 chain ID for use by JSON-RPC and state logic, and network config should distinguish between L1 and L2 chain identifiers as separate values.

## Preserved Requirements

- Unsigned transactions must be encoded with the correct nonce and L2 chain ID.
- The chain ID must come from network configuration, not a hardcoded constant.
- The nonce for unsigned transactions is obtained through the JSON-RPC flow.
- Network configuration should distinguish between L1 and L2 chain IDs as separate values.
- JSON-RPC and state logic should be able to access the configured L2 chain ID.

## Removed Noise

- Issue template / PR-style wording and conversational context.
- Mention of a specific PR number.
- Reference to an external URL or protocol discussion.
- Generated interface notes listing file locations and method signatures.
- Test-oriented wording and assertions.
- Implementation hint about a specific method name.
- Metadata about confidence, difficulty, files changed, and similar benchmark annotations.

## Risk Notes

- The original prompt implies a transaction-encoding API change, but the exact public function signature may need to be inferred from the repository.
- There is some ambiguity between L1 and L2 chain ID naming; preserve the distinction exactly as the codebase expects.
- The nonce requirement is described as already handled elsewhere, so avoid re-implementing unrelated nonce-fetching logic.

## Original Prompt

Fix executor input for unsigned transactions
After talking with @0xPolygonHermez/protocol we've decided that the executor will require correct nonce and chainID when using unsigned transactions. Since the jRPC method behind this functionality doesn't receive this data, we need to artificially inject this, probably in this method `func EncodeUnsignedTransaction(tx types.Transaction) ([]byte, error)`:

- [ ] ChainID it's already being added, but it's a hardcoded constant, instead of something that depends on network config
- [x] The nonce should be fetched from DB. Handled by the jRPC, implemented in this PR #1034

## Original Interface

Function: EncodeUnsignedTransaction(tx types.Transaction, chainID uint64) ([]byte, error)
Location: package state, file helper.go
Inputs: 
- tx: an unsigned Ethereum transaction (types.Transaction) whose fields (nonce, gasPrice, gas, to, value, data) are used for encoding.
- chainID: the L2 chain identifier (uint64) that must match the network configuration; previously hard‑coded.
Outputs: 
- []byte: RLP‑encoded unsigned transaction ready for inclusion in a batch.
- error: non‑nil if RLP encoding fails.
Description: Encodes an unsigned transaction with the provided L2 chain ID, replacing the previous constant chain ID. Used by State.EstimateGas and State.ProcessUnsignedTransaction.

Method: NetworkConfig.L1ChainID uint64
Location: package config, file network.go (struct NetworkConfig)
Inputs: None (field of the struct)
Outputs: uint64 value representing the L1 (Ethereum main) chain identifier.
Description: Replaces the former ChainID field; exposed via JSON key “l1ChainID”. Used by tests to verify correct loading and merging of network configuration.

Method: NetworkConfig.L2ChainID uint64
Location: package config, file network.go (struct NetworkConfig)
Inputs: None
Outputs: uint64 value representing the L2 (zkEVM) chain identifier.
Description: New field paired with L1ChainID; exposed via JSON key “l2ChainID”. Tests assert its proper population after loading custom network configs and during config merges.

Method: Config.ChainID uint64 (jsonrpc.Config)
Location: package jsonrpc, file config.go
Inputs: None
Outputs: uint64 L2 chain identifier used by the JSON‑RPC server.
Description: Added to propagate the L2 chain ID to RPC methods (e.g., net_version, eth_chainId). Tests construct a default Config with ChainID set and verify that it is returned by the RPC.

Method: Config.ChainID uint64 (state.Config)
Location: package state, file config.go
Inputs: None
Outputs: uint64 L2 chain identifier required for transaction encoding.
Description: New field mirrors the L2 chain ID from the network config; state.State uses it when calling EncodeUnsignedTransaction. Tests set this field in the state configuration and rely on it for correct behavior.
