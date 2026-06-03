# 8398a7__action-slack-184

- repo: 8398a7/action-slack
- language: ts
- difficulty: easy

## Rewritten Prompt

In reusable GitHub workflows, Slack notifications should detect the current job the same way they do in a normal workflow. Right now, the job lookup can fail in reusable-workflow runs, which means the Slack payload may omit the `job` and `took` fields.

When building the message payload, if the current run’s job name is reported in the reusable-workflow format `"<caller-job> / <callee-job>"`, it should still match the callee job name so the payload includes the job link and elapsed time as usual. Keep the existing behavior for normal workflows.

The `Client` API should remain compatible: `new Client(withParams, token, baseUrl, webhookUrl)` constructs a client, and `Client.prepare(initial?: string)` returns the final Slack payload object with the same shape as before, including the `job` and `took` fields when available.

## Preserved Requirements

- Reusable-workflow runs should still detect the current job and produce the usual Slack payload behavior.
- The payload should include the `job` field and the `took` field when job detection succeeds.
- The job lookup must continue to work for normal workflows.
- Reusable-workflow job names may appear as "<caller-job> / <callee-job>" and should still match the callee job name.
- Public API compatibility must be preserved for `new Client(withParams, token, baseUrl, webhookUrl)`.
- Public API compatibility must be preserved for `Client.prepare(initial?: string)`, which returns the Slack payload object.
- The returned payload shape should remain the same, including text, attachments, username, icon_emoji, icon_url, and channel.

## Removed Noise

- Issue-template sections and headings such as problem description, solution desired, alternatives considered, and additional context.
- External image/link references and GitHub URLs.
- The example YAML workflows and redundancy discussion about passing caller job names through inputs.
- Implementation hint showing a specific string-matching change.
- References to PR creation, tests, and exact source locations.

## Risk Notes

- Matching reusable-workflow job names by suffix could be ambiguous if multiple reusable workflows lead to duplicated job names.
- The observable behavior should still prefer exact job-name matches when present.
- The payload’s `job` field is expected to remain a link to the job run, and `took` should reflect elapsed time when the job is detected.

## Original Prompt

Auto detect the job in reusable workflow
**Is your feature request related to a problem? Please describe.**
In reusable workflow, action-slack cannot find job from [job list of current run](https://github.com/8398a7/action-slack/blob/v3.12.0/src/fields.ts#L141-L145). So field `job` and `took` are not available.

![image](https://user-images.githubusercontent.com/7571111/148557364-2e74f0ae-9622-467c-a66b-124b4e5ab892.png)

<details>
<summary>Example workflow</summary>

Reusable workflow
```yaml
# reusable workflow
name: shared-greet-and-notify
on:
  workflow_call:
    secrets:
      SLACK_WEBHOOK_URL
jobs:
  greet-and-notify:
    runs-on: ubuntu-latest
    steps:
      - name: greet
        run: echo hello
      - name: notify
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,took,workflow,job,ref,message
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()
```

Actual workflow
```yaml
# using the reusable workflow
name: call-reusable
on: [push]
jobs:
  greeting:
    uses: owner/repo/.github/workflows/shared-greet-and-notify.yml@main
    secrets:
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```
</details>

In reusable workflow's job, `github.job` is the job name defined in the reusable workflow. But the format of job name in API response is `"<caller-job> / <callee-job>"`
(In abobe example, `github.job` is `"something-and-notify"` and job name in the response is `"greeting / greet-and-notify"`)

**Describe the solution you'd like**
I want the job to be detected in the same way as a normal workflow.
IMO, most common cases can be resolved by checking if the job name in response ends with `" / <callee-job>"`.
```diff
-    const currentJob = resp?.data.jobs.find(job => job.name === this.jobName);
+    const currentJob = resp?.data.jobs.find(job => job.name === this.jobName || job.name.endsWith(` / ${this.jobName}`));
```
This will be wrong when workflow uses multiple reusable workflow AND job names are duplicated. But I think that it seems rare case.
If this seems good, I'll create PR.

**Describe alternatives you've considered**
There are available by passing caller's job name via `inputs` and setting `job_name`. but it is very redundancy.

<details>
<summary>Example</summary>

```yaml
name: shared-greet-and-notify
on:
  workflow_call:
    inputs:
      caller:
        type: string
        required: true
    secrets:
      SLACK_WEBHOOK_URL
jobs:
  greet-and-notify:
    runs-on: ubuntu-latest
    steps:
      - name: greet
        run: echo hello
      - name: notify
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,took,workflow,job,ref,message
          job_name: ${{ inputs.caller }} / greet-and-notify
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: always()
```

```yaml
name: call-reusable
on: [push]
jobs:
  greeting:
    uses: owner/repo/.github/workflows/shared-greet-and-notify.yml@main
    with:
      caller: greeting # unfortunately, cannot using ${{ github.job }} in here
    secrets:
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```
</details>

**Additional context**
N/A

## Original Interface

Method: Client.prepare(this, initial?: string)
Location: src/client.ts
Inputs: Optional string used as initial message or context (can be empty string). No other parameters; uses the Client instance’s configuration (With, token, base URL, webhook URL).
Outputs: Promise&lt;{ text: string; attachments: Array&lt;{ author_name: string; color: string; fields: Array&lt;{ short: boolean; title: string; value: string }&gt; }&gt;; username: string; icon_emoji: string; icon_url: string; channel: string }&gt; – the Slack payload that will be posted. In the reusable‑workflow scenario it includes a “job” field whose value is a link to the job run and a “took” field with the elapsed time.
Description: Generates the final Slack message payload based on the provided With configuration and current GitHub run information. It is invoked by tests to verify that job detection works for reusable workflows.

Function: new Client(withParams: With, token: string, baseUrl: string, webhookUrl: string)
Location: src/client.ts
Inputs:
• withParams – an object matching the With interface (fields, status, etc.).
• token – GitHub token string used for API calls.
• baseUrl – Base URL for the GitHub API (e.g., https://api.github.com).
• webhookUrl – Slack webhook URL where the message will be sent.
Outputs: Instance of Client ready to call .prepare().
Description: Constructs a Client that can compose and send Slack notifications. The tests create a client with custom With parameters and then call .prepare('') to obtain the message payload.
