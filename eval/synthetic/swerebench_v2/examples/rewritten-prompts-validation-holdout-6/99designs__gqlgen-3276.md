# 99designs__gqlgen-3276

- repo: 99designs/gqlgen
- language: go
- difficulty: easy

## Rewritten Prompt

After upgrading to `v0.17.50`, requests that use `handler.NewDefaultServer` started failing with `transport not supported` whenever the client sends an `Accept` header that includes `text/event-stream`, even for ordinary query and mutation requests.

The server should still handle normal HTTP POST GraphQL operations when SSE is not registered, and the presence of `text/event-stream` in `Accept` should not by itself prevent a request from being served by the regular transports. If SSE is registered, it should be used for subscription operations only, without breaking queries and mutations.

Keep the existing default server behavior and transport compatibility intact, but make sure non-subscription operations are not rejected just because the client advertises SSE support in `Accept`.

## Preserved Requirements

- Requests created with `handler.NewDefaultServer` must continue to work for queries and mutations after upgrading to `v0.17.50`.
- An `Accept` header containing `text/event-stream` must not cause ordinary HTTP POST GraphQL operations to fail with `transport not supported`.
- If SSE is registered, it should be used for subscription operations only.
- Normal query and mutation transport selection should continue to work when SSE is absent or not applicable.
- The fix must preserve existing default server behavior and transport compatibility.

## Removed Noise

- Version comparison details between `v0.17.49` and `v0.17.50` beyond the observable regression.
- References to external GitHub issue/PR numbers and URLs.
- Discussion of the `urql` client library, its popularity, and its internal header construction.
- Speculation about whether the client header is in error.
- Implementation diagnosis about which transport added SSE support or how it is wired internally.
- Suggested internal fix strategy and solution hints.
- Markdown quotes and code snippets from external repositories.

## Risk Notes

- The behavior depends on transport negotiation, so changes must avoid breaking existing subscription handling.
- The prompt does not specify whether SSE should be ignored, skipped, or deprioritized for non-subscription operations; only the observable outcome matters.
- Care is needed to preserve compatibility for clients that legitimately advertise SSE in `Accept` while still serving standard POST operations.

## Original Prompt

Getting "transport not supported" errors after updating to v0.17.50
Hi.  We recently upgraded from `v0.17.49` to `v0.17.50`, and all requests (queries/mutations) started failing with an error "transport not supported".

After some digging, I discovered the cause.  #3153 added the following to the HTTP POST transport:

https://github.com/99designs/gqlgen/blob/4157ef997ff03647fe4ff9ede27b505fc098932f/graphql/handler/transport/http_post.go#L30-L32

This is a problem because we just use `handler.NewDefaultServer`, which does not set up the SSE transport by default.  Even if I add it with `srv.AddTransport(transport.SSE{})`, it still doesn't solve the problem because that would only be for subscriptions, not regular queries and mutations.

I was also surprised as for why the `Accept` header had `text/event-stream` for a non-subscription request in the first place.  Turns out it was coming from our use of the popular `urql` client library, which does support GraphQL-SSE.  And also, it has this header hardcoded for every request:

https://github.com/urql-graphql/urql/blob/b9f34fd19ee5f9022db4d2f9eb610943c787dac1/packages/core/src/internal/fetchOptions.ts#L149-L154

```ts
const headers: HeadersInit = {
    accept:
      operation.kind === 'subscription'
        ? 'text/event-stream, multipart/mixed'
        : 'application/graphql-response+json, application/graphql+json, application/json, text/event-stream, multipart/mixed',
  };
```

That might be in error - I'm not sure (I reported here: https://github.com/urql-graphql/urql/issues/3673).  But nonetheless, urql has > 1M downloads weekly - so this is bound to affect a lot of people.

From a general HTTP point of view though, the `Accept` header is there to communicate which content types that the client is able to understand, and indeed urql does understand SSE.   IMHO, the presence of `text/event-stream` in the accept header shouldn't mean that the request _can't_ be served by HTTP POST.

As a suggested fix, the SSE transport, if registered, would be used for subscription operations.  Otherwise the normal list of transports should work as they would otherwise.  The HTTP POST transport shouldn't have to remove itself from SSE requests.

## Original Interface

No new interfaces are introduced.
