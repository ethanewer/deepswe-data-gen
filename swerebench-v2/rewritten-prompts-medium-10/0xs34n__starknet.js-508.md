# 0xs34n__starknet.js-508

- repo: 0xs34n/starknet.js
- language: ts
- difficulty: medium

## Rewritten Prompt

`decodeShortString` should correctly decode a StarkNet ShortString from either a hex string or a decimal numeric string. Right now, a decimal string can be accepted and decoded into the wrong text instead of returning the original ASCII string.

The function should validate its input and only accept ASCII strings that are valid short-string encodings. If the value is neither valid hex nor valid decimal, it should throw an error indicating that it is not Hex or decimal. If the input contains non-ASCII characters, it should throw an ASCII-related error.

`encodeShortString` should continue to produce a `0x`-prefixed hex string for ASCII-only strings up to 31 characters, and should reject non-ASCII or overly long input with the appropriate error.

## Preserved Requirements

- decodeShortString must decode both hex strings and decimal numeric strings into the original ASCII short string.
- decodeShortString must validate input and reject values that are neither valid hex nor valid decimal.
- decodeShortString must reject non-ASCII input with an ASCII-related error.
- encodeShortString must return a `0x`-prefixed hex encoding of the ASCII string.
- encodeShortString must reject non-ASCII strings.
- encodeShortString must reject strings longer than 31 characters.

## Removed Noise

- Issue template sections such as Describe the bug, To Reproduce, Expected behavior, Screenshots, Desktop, and Additional context.
- Example reproduction code and printed sample outputs.
- Mention of a ready PR.
- Environment/version metadata.
- Exact source file path references from the report.
- Solution hint showing the current implementation.

## Risk Notes

- The original report does not fully specify whether decimal decoding should mirror hex decoding for all valid short strings or only for values that correspond to encoded ASCII; the prompt preserves the intended behavior without over-specifying internals.
- Error wording for invalid input is preserved at a high level, but the repository may use slightly different exact messages for existing validation helpers.

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
