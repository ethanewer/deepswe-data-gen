# burntsushi__toml-339

- repo: BurntSushi/toml
- language: go
- difficulty: easy

## Rewritten Prompt

Decoding an empty TOML array into a `map[string]interface{}` should produce a non-nil empty slice, not a nil value. For example, `x = []` should decode so that `d["x"]` behaves like an empty array when later encoded or inspected, and nested empty arrays should preserve that shape as well.

Please keep the decoder behavior consistent for interface-typed values so that TOML arrays remain distinguishable from `null`-like values in Go.

## Preserved Requirements

- Decoding `x = []` into `map[string]interface{}` must yield a non-nil empty slice.
- The decoded value should behave like an empty array when later encoded or inspected.
- Nested empty arrays such as `x = [[]]` should preserve their array structure and not collapse to nil values.
- Interface-typed decoded TOML arrays should remain distinguishable from null-like Go values.

## Removed Noise

- Version comparison between 0.3.1 and 0.4.1.
- The example Go program and its logging/reflect output.
- The JSON encoding explanation and specific `null` output discussion.
- The question asking whether the change was intended.
- The suggestion to restore previous behavior.
- Generated interface notes and API signature metadata.
- Mentions of tests and test names.

## Risk Notes

- The original text implies behavior for nested arrays, but the strongest explicit requirement is for empty TOML arrays decoded through `map[string]interface{}`.
- The prompt does not specify whether this behavior should apply to all interface-typed targets or only maps, so wording stays general while preserving the example.

## Original Prompt

x = [] decodes into nil
The following program works differently between 0.3.1 and 0.4.1:

```
package main

import (
    "log"
    "reflect"

    "github.com/BurntSushi/toml"
)

func main() {
    blob := `
    x = []
    `
    var d map[string]interface{}
    if _, err := toml.Decode(blob, &d); err != nil {
        log.Fatal(err)
    }

    x := d["x"]
    log.Println(x, reflect.ValueOf(x).IsNil())
}
```

On 0.3.1,
```
[] false
```

On 0.4.1,
```
[] true
```

For my use case, this is a problem because I'm subsequently encoding the value into JSON and it writes `x = null` instead of the expected `x = []`. It also goes deeper than that, for instance `x = [[]]` would get JSONed as `x = [null]`.

Is this an intended change? Do you think it would be possible/correct to restore the previous behaviour?

## Original Interface

Function: Decode(data string, v interface{}) (MetaData, error)
Location: decode.go
Inputs:
- **data** (string): TOML document to be parsed.
- **v** (interface{}): Pointer to a value where the decoded data will be stored; commonly a pointer to a struct, map, or other Go data structure.
Outputs:
- **MetaData**: Information about the TOML document (e.g., keys, decoded types). Can be ignored if not needed.
- **error**: Non‑nil if parsing fails or if the target type is incompatible with the TOML data.
Description: Parses the TOML text supplied in *data* and populates the value pointed to by *v* with the corresponding Go representation. When *v* is a `*map[string]interface{}`, empty TOML arrays (`[]`) are decoded into a non‑nil `[]interface{}` slice, ensuring that subsequent handling (e.g., JSON encoding) treats the field as an empty array rather than `null`. This behavior is directly exercised by the test `TestDecodeInterfaceSlice`.
