# gridtools__gt4py-1449

- repo: GridTools/gt4py
- language: python
- difficulty: hard

## Rewritten Prompt

A stencil that assigns a tuple of temporaries from fields on different locations should not produce invalid ITIR. For example, a field operator like `_mo_velocity_advection_stencil_17` can assign `z_v_grad_w_wp, ddt_w_adv_wp = z_v_grad_w, ddt_w_adv`, but that must not end up as a lifted tuple expression that combines fields from incompatible domains. The resulting intermediate representation should remain type-correct so temporary handling and type inference can succeed even when the tuple elements live on different domains.

## Preserved Requirements

- Tuple assignment from fields on different locations must not violate ITIR.
- Fields that live on different domains must not be combined into an invalid lifted tuple expression.
- The behavior should work for field operators/stencils that use temporaries like `z_v_grad_w_wp, ddt_w_adv_wp = z_v_grad_w, ddt_w_adv`.
- Temporary handling must remain compatible with type inference.
- The issue should be fixed even when the tuple-lifted expression would otherwise appear in the generated intermediate representation.

## Removed Noise

- Repository and language metadata.
- Issue/PR-style explanatory text and diagnostics.
- The concrete ITIR snippet and internal lambda/lift syntax.
- The note about the problem disappearing when all lifts are force inlined.
- References to the temporary pass implementation detail and type inference failure cause.
- Boilerplate about generated interface notes and compatibility notes with no content.
- File/path-oriented context.

## Risk Notes

- The original example uses specific field and dimension names; the behavioral requirement is preserved, but the exact minimal reproducer may involve other domain combinations too.
- The prompt implies invalid cross-domain tuple lifting is the core bug; implementation details about where the fix belongs are intentionally omitted.

## Original Prompt

tuples of fields of different locations violate ITIR
The following stencil from Icon4Py currently fails when using temporaries:
```python
@field_operator
def _mo_velocity_advection_stencil_17(
    e_bln_c_s: Field[[CEDim], wpfloat],
    z_v_grad_w: Field[[EdgeDim, KDim], vpfloat],
    ddt_w_adv: Field[[CellDim, KDim], vpfloat],
) -> Field[[CellDim, KDim], vpfloat]:
    z_v_grad_w_wp, ddt_w_adv_wp = z_v_grad_w, ddt_w_adv
    ...
```
The resulting ITIR contains the following:
```
(↑(λ(__arg0, __arg1) → {·__arg0, ·__arg1}))(z_v_grad_w, ddt_w_adv)
```
However since `z_v_grad_w` and `ddt_w_adv` live on different domains this is invalid. This currently breaks the temporary pass (as type inference fails). However when all lifts are force inlined the problem disappears so it never surfaced before.

## Original Interface

No new interfaces are introduced.
