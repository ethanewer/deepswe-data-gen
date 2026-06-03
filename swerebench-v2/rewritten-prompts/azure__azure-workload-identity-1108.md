# azure__azure-workload-identity-1108

- repo: Azure/azure-workload-identity
- language: go
- difficulty: easy

## Rewritten Prompt

When the webhook injects the proxy sidecar into a Pod, the sidecar must be placed at the start of the container list rather than appended to the end. This should ensure the sidecar is available before application containers begin running, avoiding repeated application restarts caused by the sidecar coming up too late.

## Preserved Requirements

- The proxy sidecar is injected into the Pod's container list.
- The sidecar must be prepended to the container list, not appended.
- The ordering should prevent application containers from starting before the sidecar is available.
- The change should avoid application restarts caused by the sidecar being added too late.

## Removed Noise

- Repository and language metadata.
- External GitHub link and line reference.
- Issue-style phrasing and suggestion wording.
- Implementation hint about swapping arguments in an append call.
- PR/test references and other benchmark metadata.

## Risk Notes

- The exact interaction with post-start hooks is implied by the original issue; preserve the behavioral effect without over-specifying implementation details.
- If the webhook already mutates multiple container-related lists, ensure only the user-visible pod startup behavior changes as requested.

## Original Prompt

Prepend proxy sidecar to the container list to prevent application restarts
[Currently](https://github.com/Azure/azure-workload-identity/blob/37dc12fdf67ec0bd54d3e3f5c24273a0207707e8/pkg/webhook/webhook.go#L236C7-L236C7), proxy sidecar gets added to the end of the containers list. In this case, the application container may crash a couple of times before the sidecar becomes available.

I suggest prepending the sidecar to the containers list (by swapping containers and corev1.Container{...} in append call), which will block all other containers due to the post-start hook in the sidecar.

## Original Interface

No new interfaces are introduced.
