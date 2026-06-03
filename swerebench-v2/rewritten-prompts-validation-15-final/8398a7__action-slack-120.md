# 8398a7__action-slack-120

- repo: 8398a7/action-slack
- language: ts
- difficulty: easy

## Rewritten Prompt

Ensure `SLACK_WEBHOOK_URL` is validated before creating the Slack client. If the webhook URL is missing, `null`, or an empty string, the action should fail with `Error('Specify secrets.SLACK_WEBHOOK_URL')` instead of later throwing an obscure runtime error like `Cannot read property 'replace' of null`.

Preserve the `Client` constructor behavior so it accepts `Client.constructor(props: With, token: string, webhookUrl?: string | null)` and only creates the Slack webhook client when a valid string URL is provided. The validation should treat `undefined`, `null`, and `''` as invalid.

## Preserved Requirements

- `SLACK_WEBHOOK_URL` must be validated as a string before use.
- Missing, `null`, or empty webhook URLs must throw `Error('Specify secrets.SLACK_WEBHOOK_URL')`.
- The `Client` constructor accepts `props: With`, `token: string`, and optional `webhookUrl?: string | null`.
- The constructor must continue to return a new `Client` instance when given a valid webhook URL.
- The behavior must avoid the later null dereference / `replace` runtime error by failing early.

## Removed Noise

- Issue template headings such as feature request / describe the problem / solution / alternatives / additional context.
- The typo example using `${{ secrets.SLACK_WEBHOOK.URL }}` as a demonstration.
- The explicit runtime stack symptom beyond the observable failure mode.
- References to tests, test expectations, or implementation exercise notes.
- Source-location details from generated interface notes.
- Repository and language metadata.
- Any PR/metadata framing and solution hints about where to implement the change.

## Risk Notes

- The interface notes say the constructor may receive `null`, `undefined`, or an empty string and should reject all three; the prompt preserves that exact contract.
- The exact error message is preserved to avoid changing downstream behavior.
- The prompt avoids naming internal files or implementation steps while keeping the public constructor signature and validation semantics.

## Original Prompt

Validate SLACK_WEBHOOK_URL
**Is your feature request related to a problem? Please describe.**

Say, a user made a typo like:

```
SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK.URL }}
```

the value of `SLACK_WEBHOOK_URL` would be `null`. This results in an obscure error like:

```
Cannot read property 'replace' of null
```

**Describe the solution you'd like**

Validate that `SLACK_WEBHOOK_URL` must be a string.

**Describe alternatives you've considered**

N/A

**Additional context**

N/A

## Original Interface

Method: Client.constructor(props: With, token: string, webhookUrl?: string | null) Location: src/client.ts
Inputs:
- **props** – configuration object of type `With` (contains fields like `fields`, `status`, etc.).
- **token** – GitHub token as a non‑empty string.
- **webhookUrl** – optional Slack Incoming Webhook URL; may be a string, `null`, or omitted. The constructor validates this value and throws if it is `undefined`, `null`, or an empty string (`''`).
Outputs: Returns a new `Client` instance. Throws `Error('Specify secrets.SLACK_WEBHOOK_URL')` when the webhook URL is missing, null, or empty.
Description: Creates a client for sending Slack messages, initializing the Octokit instance and the `IncomingWebhook` only after confirming that a valid webhook URL was supplied. This change adds `null` handling and emptiness checking, which is exercised in the tests that expect an exception for invalid webhook values.
