# rezact__rezact-43

- repo: Rezact/Rezact
- language: ts
- difficulty: hard

## Rewritten Prompt

Allow `Signal` values to swap fragment content as well as single HTMLElements. In the example below, assigning a fragment to a signal should update the rendered output without breaking existing interactivity: after clicking the button that sets `<span>...</span>`, the button that sets a fragment like `<>...</>` should also work, and the first button should still keep working afterward.

```tsx
import {Signal} from '@rezact/rezact/signals'

export default function Test() {
  let $elmRef = new Signal(<p>Not Changed</p>);

  const works = () => {
    $elmRef = <span>
      <p>This</p>
      <p>Works</p>
    </span>
  };

  const doesntWork = () => {
    $elmRef = <>
      <p>This</p>
      <p>Doesn't</p>
    </>
  }

  return (
    <>
      {$elmRef}
      <button onClick={works}>Change (works)</button>
      <button onClick={doesntWork}>Change (doesn't)</button>
    </>
  );
}
```

## Preserved Requirements

- `Signal` must support swapping in fragment content, not only single HTMLElements.
- Assigning a fragment to a signal should render correctly.
- Updating to a fragment must not break existing event handling or future updates; both buttons should remain functional after either update.
- The public import `@rezact/rezact/signals` and the `Signal` API must remain usable as shown.

## Removed Noise

- Issue/PR-style framing such as "IMPROVEMENT".
- Source-location reference to a specific line range in the repository.
- GitHub URL and other external links.
- Meta commentary about the current code's internal limitation wording beyond the observable behavior.

## Risk Notes

- The exact mutation model of `Signal` assignment is inferred from the example and should be preserved in a behaviorally compatible way.
- The fragment case should match existing rendering semantics closely enough that the DOM remains interactive after replacement.

## Original Prompt

IMPROVEMENT: Allow Signals to Swap Fragments
Consider the example below:

```tsx
import {Signal} from '@rezact/rezact/signals'

export default function Test() {

  let $elmRef = new Signal(<p>Not Changed</p>);

  const works = () => {
    $elmRef = <span>
      <p>This</p>
      <p>Works</p>
    </span>
  };

  const doesntWork = () => {
    $elmRef = <>
      <p>This</p>
      <p>Doesn't</p>
    </>
  }

  return (
    <>
      {$elmRef}
      <button onClick={works}>Change (works)</button>
      <button onClick={doesntWork}>Change (doesn't)</button>
    </>
  );
}

```

The first button works fine, but clicking the second button stops even the first button from working.  The current code allows swapping HTMLElements, but not Fragments.


https://github.com/Rezact/Rezact/blob/3983220cf8280366599ee67ffc44389adae4b2eb/src/lib/rezact/signals.ts#L135-L143

## Original Interface

No new interfaces are introduced.
