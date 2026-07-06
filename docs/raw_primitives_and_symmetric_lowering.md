# Raw Primitives And Symmetric Lowering

Status: design note.

## Guiding Principle

Uni20 should keep a clear separation between:

- raw dense tensor primitives, which operate on storage, views, strides, axes, and arithmetic; and
- symmetric tensor operations, which operate on quantum numbers, sectors, block structure, fusion rules, and representation data.

The dense primitive layer should not know about symmetry. The symmetric layer should lower its work into dense primitive programs.

This separation is important because some optimizations that are natural at the raw arithmetic level look unnatural, or even ill-typed, if described as operations on a whole symmetric tensor. That is not a problem. It means the optimization belongs in the lowering step.

## Dense Primitive Layer

The dense layer should provide a small set of layout and arithmetic primitives:

- view construction from shape, strides, offset, and storage;
- synthetic views, such as diagonals with computed strides;
- elementwise map and zip;
- reduction over one or more axes;
- contraction over paired axes;
- permutation and materialization;
- reshape, split, and coalesce axes;
- explicit copy between compatible views.

These primitives should be expressed in terms of resolved dense data views. They should not mention quantum numbers, sectors, labels, fusion trees, or block sparsity.

For example, a trace over two dense axes is not fundamentally a special tensor operation. It is a synthetic diagonal view followed by a reduction over the diagonal axis.

## Symmetric Tensor Layer

The symmetric tensor layer owns:

- sector lookup and block selection;
- charge conservation;
- basis orientation and dual legs;
- fusion and splitting rules;
- recoupling transformations;
- intertwiner and coupling-coefficient data;
- blockwise contraction scheduling;
- decisions about when to materialize, coalesce, or regroup blocks.

The output of this layer should be a lowering plan: a sequence or graph of dense primitive operations on concrete block data.

In particular, it is valid for a symmetric operation to lower into raw arithmetic that does not resemble a single clean operation on the original symmetric tensor. Examples include:

- coalescing several compatible dense blocks before a contraction;
- grouping contractions by dense block shape;
- summing contributions from different sector paths when weighted by the appropriate coupling coefficient;
- rewriting recoupling moves into dense block operations with scalar prefactors, such as 6j-symbol factors;
- exploiting intertwiners that are implicit in the symmetric representation but explicit in the raw arithmetic.

These are not violations of the abstraction. They are the point of the abstraction boundary.

## Design Rule

The symmetric layer may perform high-level algebraic rewrites, but the dense layer should remain simple and mechanical.

Do not add a dense primitive because it is meaningful for symmetric tensors. Add a dense primitive because it is a useful operation on raw tensor data.

Conversely, do not reject a raw primitive lowering because it looks strange at the symmetric tensor level. If it is algebraically valid after sector resolution and coupling coefficients are accounted for, then it is a legitimate lowering strategy.

The intended flow is:

```text
symmetric tensor expression
  -> sector/block/intertwiner lowering plan
  -> dense primitive graph
  -> backend execution
```

This keeps the backend and kernel layers reusable, testable, and independent of quantum-number machinery.
