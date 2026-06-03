# burntsushi__toml-418

- repo: BurntSushi/toml
- language: go
- difficulty: easy

## Rewritten Prompt

`Metadata.Keys()` can return incorrect key paths when decoding nested TOML tables and values. Make it return the correct fully qualified key for every decoded key/value in document order, without duplicating or overwriting earlier entries.

For example, decoding a document like a nested table with two keys inside it should produce three distinct keys: the table path itself, then the first value key, then the second value key. The metadata for decoded documents should also remain consistent for `Undecoded`, since it depends on the same recorded key paths.

## Preserved Requirements

- `Metadata.Keys()` must return fully qualified key paths for every decoded key/value in a TOML document.
- The returned keys must be in document order.
- The returned keys must not contain duplicate entries caused by aliasing or mutation.
- Nested tables and their values must be recorded correctly.
- `Metadata.Undecoded` must remain consistent with the recorded key paths.

## Removed Noise

- Issue template and sample code.
- Playground link.
- Exact observed output and expected output blocks.
- Link to a specific line and implementation hint about `append` aliasing.
- Go playground example about slice capacity behavior.
- Repository/file path reference presented as a fix hint.

## Risk Notes

- The visible bug involves slice aliasing, so the fix likely needs to copy key data before later mutation.
- `Undecoded` is mentioned indirectly; preserve its behavior even though the report centers on `Keys()`.
- The prompt does not specify where the bug is located, so exploration is required.

## Original Prompt

metadata.Keys (and thus metadata.Undecoded) is invalid in some cases
**Sample code:**
```
package main

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

func main() {
	content := `[table.subtable.subsubtable]
a = 1
b = 2
`
	var v interface{}
	metadata, err := toml.Decode(content, &v)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%#v\n", metadata.Keys())
}
```

Playground: https://go.dev/play/p/Rr5bm5DbEql

**Output:**
```
[]toml.Key{toml.Key{"table", "subtable", "subsubtable"}, toml.Key{"table", "subtable", "subsubtable", "b"}, toml.Key{"table", "subtable", "subsubtable", "b"}}
```

Notice how the second and third value in that slice both point to `b`.

**Expected output:**
```
[]toml.Key{toml.Key{"table", "subtable", "subsubtable"}, toml.Key{"table", "subtable", "subsubtable", "a"}, toml.Key{"table", "subtable", "subsubtable", "b"}}
```

The root cause seems to be somewhere near this [append call](https://github.com/BurntSushi/toml/blob/eb727477b3f7e4aa878635d39257753f0840811b/meta.go#L139). The result of `append` can return a slice backed by the same array if the capacity is available. See Go example code: https://go.dev/play/p/2FGKuCvCUKz.

## Original Interface

Method: Metadata.Keys(self) 
Location: meta.go 
Inputs: (receiver) Metadata – the metadata object returned by toml.Decode or toml.DecodeFile. No additional parameters. 
Outputs: []Key – a slice of Key values, each representing a fully‑qualified key path (as a slice of strings) for every key/value decoded in the TOML document. The slice reflects the order of appearance and contains no duplicate entries for the same key. 
Description: Returns all decoded key paths from the TOML document, used by the test suite to verify that the parser correctly records keys—including nested tables and values—without producing duplicate entries.
