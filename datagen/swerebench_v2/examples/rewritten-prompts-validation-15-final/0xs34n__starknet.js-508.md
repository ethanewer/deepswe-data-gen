# 0xs34n__starknet.js-508

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

`decodeShortString` should accept both hex-encoded ShortStrings and decimal numeric strings, and return the original ASCII text in either case. Right now a decimal string is accepted but decoded into garbage instead of the correct value.

Keep `encodeShortString(str: string)` and `decodeShortString(str: string)` compatible with their current behavior: `encodeShortString` still returns a `0x`-prefixed hex string for an ASCII-only ShortString of up to 31 characters, and `decodeShortString` still turns either a hex string or a decimal numeric string back into the original ASCII string.

Validation must remain in place: non-ASCII input should throw `"<value> is not an ASCII string"`, overly long ShortStrings should throw `"<value> is too long"`, and inputs that are neither valid hex nor valid decimal numeric strings should throw `"<value> is not Hex or decimal"`.

## Preserved Requirements

- decodeShortString must handle both hex strings and decimal numeric strings.
- decodeShortString must return the same original ASCII string for either representation.
- encodeShortString must accept ASCII-only strings up to 31 characters.
- encodeShortString must return a 0x-prefixed hex string representing the ASCII bytes.
- Non-ASCII input must throw "<value> is not an ASCII string".
- Inputs longer than 31 characters must throw "<value> is too long".
- Inputs that are neither valid hex nor valid decimal numeric strings must throw "<value> is not Hex or decimal".
- Public symbol names `encodeShortString` and `decodeShortString` must remain available with the same callable signatures.

## Removed Noise

- Issue/bug report template sections and headings.
- External PR/status commentary.
- Environment details that do not affect the behavior requirement.
- Screenshots/N/A placeholders.
- The concrete repository/file path references.
- The internal implementation snippet shown in the report.
- The specific reproduction script and console output, while keeping the observed failure behavior.

## Risk Notes

- Decimal parsing can be ambiguous if the input also looks like a hex string without a prefix; the function should preserve the documented validation behavior.
- The existing contract limits ShortStrings to 31 ASCII characters, so decoding should not broaden that range.
- The behavior for invalid decimal strings should remain distinct from non-ASCII input errors.

## Original Prompt

decodeShortString : wrong answer of the function
**Describe the bug**
Hello,
I noted a problem with the function `utils/shortstrings.ts/decodeShortString`.
If I use it with an hex string ('0x321....456'), everything is fine.
But if the str:string is  an integer string ('1542233...56'), the function accept it, process without fail, and send back a wrong answer.
```typescript
export function decodeShortString(str: string) {
  return removeHexPrefix(str).replace(/.{2}/g, (hex) => String.fromCharCode(parseInt(hex, 16)));
}
```

**To Reproduce the problem :**
```typescript
const myText = "decimal number fail!";
console.log("myText =", myText);
const encodedText: string = shortString.encodeShortString(myText);
console.log("encodedText =", encodedText);
const myTextDecoded = shortString.decodeShortString(encodedText);
console.log("finalString =", myTextDecoded);
const decimalEncoded: string = BigInt(encodedText).toString(10);
console.log("decimalEncoded =", decimalEncoded);
const wrongTextDecoded : string= shortString.decodeShortString(decimalEncoded);
console.log("wrongTextDecoded =", wrongTextDecoded);
```
Result :
```
myText = decimal number fail!
encodedText = 0x646563696d616c206e756d626572206661696c21
finalString = decimal number fail!
decimalEncoded = 573160112338783013729123589963347449102752902177
wrongTextDecoded = W1`#8x0r#Xc4tI'R
```


**Expected behavior**
This function should handle properly both formats.

**Screenshots**
N/A

**Desktop (please complete the following information):**

- Browser & version : N/A
- Node version : "ts-node": "^10.9.1"
- StarkNet.js version : 4.21.0
- Network [ : N/A

**Additional context**
A PR is ready to send.

## Original Interface

Function: encodeShortString(str: string)
Location: src/utils/shortString.ts
Inputs: An ASCII‑only string of maximum 31 characters. Throws an error if the string contains non‑ASCII characters (`"<value> is not an ASCII string"`) or exceeds 31 characters (`"<value> is too long"`).
Outputs: A hex‑encoded string prefixed with “0x” that represents the ASCII characters (each character converted to its two‑digit hex code). Example: `"hello"` → `"0x68656c6c6f"`.
Description: Encodes a StarkNet ShortString (≤31 ASCII chars) into the hex format required by Cairo contracts.

Function: decodeShortString(str: string)
Location: src/utils/shortString.ts
Inputs: A string that must be either:
  • a hex string (e.g., `"0x68656c6c6f"`), or
  • a decimal numeric string (e.g., `"448378203247"`).
The input must consist solely of ASCII characters; otherwise an error `"<value> is not an ASCII string"` is thrown. If the string is neither a valid hex nor a decimal numeric string, an error `"<value> is not Hex or decimal"` is thrown.
Outputs: The original ASCII string represented by the hex or decimal input (maximum 31 characters). Example: `"0x68656c6c6f"` → `"hello"`; `"448378203247"` → `"hello"`.
Description: Decodes a StarkNet ShortString from either its hex representation or its decimal representation back to the original ASCII string, performing input validation and error handling.
