# Axis Labels, Contraction, and Braiding

Status: design note. This is intended to fix the core policy before Python
bindings and higher-level tensor-network interfaces acquire accidental
semantics. It complements the block-sparse tensor and symmetry design in
[BlockTensor](block_tensor.md). The ordered `Domain`/`Codomain` boundary and
independent `Space`/`DualSpace` semantics are defined in
[Spaces, Duals, and Tensor Morphisms](spaces_duals_and_morphisms.md).

## Problem

Tensor-network code benefits from human-readable leg names such as `left`, `phys`, `right`, `bra`, `ket`, or `bond_12`. These names make diagrams, diagnostics, and debugging substantially easier.

However, labels are dangerous if they become a second way to address tensor axes. A tensor already has a canonical axis order. If labels are also used as an independent axis-ordering system, the library can end up with two competing truths:

- the integer axis order used by storage, strides, kernels, and permutations;
- the label order used by contraction expressions and user code.

This is fragile even for ordinary dense tensors. It becomes mathematically untenable for fermionic, braided, or non-Abelian tensor networks, where reordering legs can carry signs, R-matrices, fusion-tree changes, or other algebraic data. In those settings, changing labels cannot repair an incorrect leg order. The fix must be an explicit tensor operation.

## Core Rule

Integer axis positions are canonical.

Labels are optional metadata. They may describe axes, aid display, and assert user intent, but they do not decide which axis an operation acts on.

Consequences:

- `extent(2)` is canonical.
- `permute([1, 0, 2])` is canonical.
- `contract(A, 2, B, 0)` is canonical.
- `label="right"` is metadata attached to whichever axis currently carries that label.
- Labels move with axes under permutation.
- Relabeling does not change algebraic structure.

For `BlockTensor`, each canonical axis position is one ordered occurrence in
the tensor's domain or codomain. The occurrence refers to an identity-bearing
space, possibly through `DualSpace`. Axis position, morphism side, object
duality, and display label are distinct properties; none can be reconstructed
from another.

This is deliberately stricter than label-driven tensor contraction systems. It is less magical, but it is inspectable, testable, and compatible with braided tensor categories.

## Labels As Assertions

Labels are still useful as optional checks. A contraction can specify explicit axis numbers and also assert expected labels:

```python
contract(A, 2, B, 0, expect=("right", "left"))
```

This means:

- contract axis 2 of `A` with axis 0 of `B`;
- if `A` axis 2 has a label, it must be compatible with `"right"`;
- if `B` axis 0 has a label, it must be compatible with `"left"`;
- if either assertion fails, report an error before doing the contraction.

The label check catches wrong-axis bugs without making labels the addressing mechanism.

Labels do not need to be globally unique unless a future API explicitly asks for a unique label set. Ambiguous or repeated labels are acceptable as display metadata; they become errors only when a caller asks to use them as assertions in a context that requires a single expected axis.

A more object-oriented spelling could be:

```python
contract(A.axis(2).expect("right"), B.axis(0).expect("left"))
```

The key point is that the user still names the axis position.

## Permutations And Labels

If labels are attached to axes, then a permutation moves labels with the axes:

```python
A.labels == ["left", "phys", "right"]
B = A.permute([1, 0, 2])
B.labels == ["phys", "left", "right"]
```

This is safe because labels describe the axes after the operation. They are not used to decide what the permutation means.

Relabeling is only metadata mutation:

```python
A.set_axis_label(2, "bond")
```

This must not alter storage, strides, leg order, symmetry data, braiding, or contraction semantics.

## Braiding And Adjacency

For ordinary dense tensors with no symmetry or braiding, any pair of compatible axes can be contracted. For braided, fermionic, or non-Abelian tensor networks, the ordering of legs is part of the mathematical state.

In those contexts, contraction should require adjacent compatible legs. If the requested contraction would cross other legs, it should fail and require an explicit braid or permutation operation first.

Example policy:

```python
contract(A, 2, B, 0)  # valid only if these are adjacent/boundary legs
contract(A, 1, B, 0)  # error if axis 1 is not adjacent to the contraction boundary
A2 = A.braid_to_boundary(1)
contract(A2, boundary_axis, B, 0)
```

The exact API can change, but the semantic rule should not: if algebraic reordering is required, the user must request it explicitly.

This is essential for fermionic signs and non-Abelian R/F-move bookkeeping. A label change cannot stand in for a braid.

## Presentation Layer

The presentation layer can use labels heavily, because presentation is not algebra.

Useful diagnostics include:

- tensor-network sketches with labeled legs;
- contraction diagrams showing selected axes;
- errors that show why a non-adjacent contraction is illegal;
- diagrams before and after `permute` or `braid`;
- optional display of extents, block sectors, directions, conjugation, and symmetry information.

A simple terminal sketch is already valuable:

```text
       +---+       +---+
i ---> | A |-- a ->| B |----> j
       +---+       +---+
         |           |
         b           c
```

Unicode, color, Markdown, HTML, and Jupyter renderers can improve the presentation, but the data model should remain semantic: nodes, axes, labels, directions, and operations.

## Python Interface Guidance

Python should not make persistent string labels canonical. There is no robust way to make strings behave like typed axes without recreating the same ambiguity this design is trying to avoid.

Recommended Python style:

```python
L, P, R = 0, 1, 2
A.extent(P)
A = A.permute([P, L, R])
contract(A, 2, B, 0, expect=("right", "left"))
```

An `einsum`-style layer may use labels inside a single expression:

```python
einsum("lir,rjs->lijs", A, B)
```

Those labels are local to the expression. They should not become persistent tensor state that silently drives later contractions.

## C++ Interface Guidance

C++ has more room for stronger axis handles or tag objects:

```cpp
struct left_axis {};
struct phys_axis {};
struct right_axis {};

extent<phys_axis>(A);
permute<phys_axis, left_axis, right_axis>(A);
```

or runtime handles:

```cpp
auto right = A.axis(2).expect("right");
auto left = B.axis(0).expect("left");
contract(A, right, B, left);
```

These mechanisms are acceptable only if they remain unambiguous handles to canonical axes. They must not introduce a second independent leg ordering.

## Design Principle

Labels should help users see and verify tensor structure. They should not define tensor structure.

If a tensor has the wrong leg order, the repair is a permutation or braid. If a tensor has the wrong labels, the repair is relabeling. These are different operations and the API must keep them different.
