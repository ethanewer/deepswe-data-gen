# 99designs__gqlgen-3276

- repo: 99designs/gqlgen
- language: go
- difficulty: easy

## Rewritten Prompt

After upgrading to the newer gqlgen release, regular GraphQL queries and mutations started failing with "transport not supported" when the client sends an `Accept` header that includes `text/event-stream`. This should not happen for non-subscription operations.

Ensure that an SSE-capable `Accept` header does not prevent HTTP POST from handling ordinary queries and mutations. SSE should still be used for subscription operations when it is actually registered, but the default transport selection for non-subscriptions must continue to work normally.

In short: the presence of `text/event-stream` in `Accept` should not cause regular POST requests to be rejected just because SSE is available or mentioned in the header.

## Preserved Requirements

- Regular GraphQL queries and mutations must not fail with "transport not supported" solely because the request `Accept` header includes `text/event-stream`.
- HTTP POST handling should still work normally for non-subscription operations.
- SSE should be used for subscription operations when SSE is registered.
- The `Accept` header should be interpreted as indicating what the client can understand, not as a reason to reject ordinary POST requests.

## Removed Noise

- Version numbers and upgrade history from v0.17.49 to v0.17.50.
- Issue-style background explanation and personal commentary.
- References to specific upstream code lines and external URLs.
- Discussion of the urql client library, download counts, and a separate upstream issue report.
- Speculation about whether the client header is in error.
- Suggested implementation details about how transports should be registered or removed.

## Risk Notes

- The exact transport-selection behavior for mixed `Accept` headers should be preserved for subscriptions while relaxing rejection for regular operations.
- The fix should avoid changing unrelated HTTP content negotiation behavior beyond the reported failure mode.

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
