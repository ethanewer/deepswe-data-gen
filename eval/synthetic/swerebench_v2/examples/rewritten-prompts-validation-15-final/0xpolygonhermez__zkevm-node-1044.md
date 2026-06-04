# 0xpolygonhermez__zkevm-node-1044

- repo: 0xPolygonHermez/zkevm-node
- language: go
- difficulty: medium

## Rewritten Prompt

Unsigned transactions need to be encoded with the correct chain ID and nonce before they reach the executor. Update the behavior so the unsigned-transaction encoding uses the network’s configured L2 chain ID instead of a hardcoded value, and make sure the nonce is included from the available transaction state.

The chain ID must come from configuration and remain compatible with the existing public interfaces. The network config now exposes separate L1 and L2 chain ID fields, and the RPC/state configs each carry a `ChainID` value that should be used to propagate the L2 chain ID where needed. Keep the unsigned transaction encoding API returning RLP-encoded bytes and an error on failure.

## Preserved Requirements

- Unsigned transactions must be encoded with the correct nonce and chain ID before executor use.
- The unsigned transaction encoder must use the network-configured L2 chain ID rather than a hardcoded constant.
- The encoding API remains `EncodeUnsignedTransaction(tx types.Transaction, chainID uint64) ([]byte, error)` and still returns RLP-encoded bytes plus an error on encoding failure.
- `NetworkConfig.L1ChainID` remains a public `uint64` field exposed via JSON key `l1ChainID`.
- `NetworkConfig.L2ChainID` remains a public `uint64` field exposed via JSON key `l2ChainID`.
- `jsonrpc.Config` includes a public `ChainID uint64` field used to propagate the L2 chain ID to RPC methods.
- `state.Config` includes a public `ChainID uint64` field used by state transaction encoding.
- The public symbols `EncodeUnsignedTransaction`, `NetworkConfig.L1ChainID`, `NetworkConfig.L2ChainID`, and `Config.ChainID` must remain available with compatible names and types.

## Removed Noise

- Issue template and conversational context.
- Reference to a specific protocol discussion.
- Mention of a pull request number.
- Implementation hint about using a specific method name as the place to inject data.
- Test references and descriptions of how tests verify the change.
- Source file locations and package/file path details.
- Explicit mention that the nonce handling was already implemented elsewhere.

## Risk Notes

- The prompt keeps the need for nonce and chain ID correctness, but does not spell out exactly where nonce should be sourced from; the repository may need to preserve existing caller behavior.
- `Config.ChainID` appears in both JSON-RPC and state configs; the task assumes both should continue representing the L2 chain ID, which could require careful wiring to avoid mismatch with any L1 naming changes.
- The original notes mention `NetworkConfig.L1ChainID` and `L2ChainID` while also saying `Config.ChainID` is the L2 chain ID; this naming split may be easy to confuse during implementation.

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
