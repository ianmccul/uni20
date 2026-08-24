# Reviewing Uni20 Changes

## Review Goal

A Uni20 review looks first for incorrect behavior, invalid assumptions,
regressions, missing evidence, and architectural inconsistencies. Formatting and
minor style suggestions come after correctness findings.

Independent AI review is useful for substantial changes, but a model approval is
not proof. Compilers, tests, sanitizers, reference calculations, and explicit
contracts provide the primary evidence.

## Review Order

Read a change in this order:

1. The maintainer-approved decision, issue, or canonical design document.
2. New and changed tests, including the source of their expected results.
3. The implementation diff.
4. Nearby implementations that should obey the same contract.
5. Documentation and examples affected by the behavior.

Do not assume that an existing implementation or test defines intended behavior
when the change is specifically correcting that behavior.

## Core Questions

1. Is the stated contract internally consistent and appropriate for Uni20?
2. Does the regression test fail for the previous behavior?
3. Could the test pass for a materially wrong implementation?
4. Is the oracle independent: an exact small case, reference implementation,
   provider comparison, finite difference, adjoint identity, or similar evidence?
5. Does the implementation address the root cause rather than one symptom?
6. Which neighboring paths share the same assumption or helper?
7. Are error, cancellation, and partial-result paths valid?
8. Does the change introduce hidden allocation, materialization,
   synchronization, host/device transfer, or dense fallback?
9. Are canonical docs still accurate, and did the change affect any durable
   repository-wide guidance?
10. What checks were unavailable, and what residual risk remains?

## Reachability and Representability

A correctness finding needs a concrete construction and execution path using
inputs permitted by the documented API and subsystem invariants. The range of
an underlying integer carrier does not by itself define the supported domain of
a tensor dimension, allocation or workspace size, stride, provider integer, or
model quantum number.

For an overflow finding, identify:

- the exact arithmetic operation;
- how its operands arise from valid inputs;
- why the operation occurs before an existing constructor, codec, storage,
  provider, or allocation check; and
- the observable incorrect behavior.

Do not promote a counterexample which requires physically unconstructible
storage, model-invalid metadata, or violation of an established precondition
into a correctness defect. Do not request a hot-path guard solely to handle
such a state. If the permitted input domain is unclear, report a contract
question instead.

This does not exempt parsers, serialized or otherwise untrusted metadata,
provider integer lowering, allocation planning, or arithmetic reached by
ordinary valid inputs. Those boundaries must reject unrepresentable values
before unsafe arithmetic or observable mutation.

## Numerical and Linear Algebra Checklist

Consider only dimensions relevant to the operation, but check them explicitly:

- real and complex scalars;
- supported precision types, including binary128 where configured;
- row-major, column-major, conjugating, and supported strided layouts;
- rank-0, rank-1, singleton, zero-extent, and empty cases;
- negative or unsupported strides;
- overflow in norms, dimensions, leading dimensions, and workspace sizes;
- provider differences across reference, BLAS/LAPACK, MKL, OpenBLAS, and
  MPLAPACK paths;
- aliasing and destructive-input contracts;
- convergence, tolerance, residual, and error-bound definitions.

Do not require every item for every change. State why an omitted dimension is
irrelevant when that is not obvious.

## Tensor, Mdspan, and Backend Checklist

- Does the mdspan accessor change value semantics?
- Is direct pointer access permitted by the accessor, rather than merely by a
  pointer-shaped data handle?
- Are const element semantics preserved for read-only views?
- Does backend selection preserve storage kind and location?
- Are unsupported layouts declined cleanly or materialized explicitly?
- Are output allocation, mutation, and ownership decisions made at the correct
  layer?
- Does a fallback silently move data, erase metadata, or change the mathematical
  operation?

## Async and AD Checklist

- Are coroutine lambdas captureless and `static`?
- Are values needed after suspension passed by value into the coroutine frame?
- Are `EpochQueue`, buffer, and alias lifetimes valid?
- Does a decline preserve inputs and fixed/update outputs, and limit any visible
  mutation to operation-authorized preparation of a replaceable output?
- Are exceptions delivered to every required output without stranding waiters?
- Can an unobserved failed or cancelled branch block dependent work?
- Are read/write epochs ordered by causality rather than scheduler timing?
- Does reverse-mode accumulation preserve the documented real or complex
  derivative convention?
- Are absent gradients and graph pruning handled without deadlock?

## Symmetry and Block-Sparse Checklist

- Are quantum numbers, block keys, local spaces, and leg orientations preserved?
- Does each generated block satisfy the applicable selection rule?
- Does any adapter flatten a symmetry-aware tensor into dense storage?
- If a dense projection exists for diagnostics, is it explicitly named and
  prevented from feeding back into the symmetry-aware calculation?
- Do contraction worklists preserve logical block identity and placement?

Loss of symmetry metadata is a correctness defect, not a performance tradeoff.

## Reporting Findings

Lead with findings ordered by severity and include file and line references.
Use these categories when helpful:

- **Blocker:** incorrect results, undefined behavior, deadlock, data loss,
  symmetry loss, or an invalid scientific contract.
- **Important:** credible regression, missing failure handling, weak oracle, or
  significant hidden cost that should be resolved before the checkpoint lands.
- **Suggestion:** maintainability or clarity improvement that does not invalidate
  the change.
- **Question:** an unresolved assumption whose answer may change the finding.

If no defect is found, say so directly and identify remaining test gaps or
configuration risk. Do not pad the review with style comments to manufacture
findings.
