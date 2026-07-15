# Trace As A Dense Reduction Primitive

Status: design note.

## Summary

For a generic dense tensor, trace should be implemented as:

```text
synthetic diagonal view
  -> reduction over the diagonal axis
```

It should not be treated as a special high-level tensor operation in the dense backend.

This note is about dense tensor data. Symmetric tensors may expose a trace-like operation, but that operation should first lower through sector/block logic and then call dense primitives on the relevant blocks.

## Dense Trace Model

Given an input tensor with shape and element strides, tracing axes `a` and `b` requires:

```text
shape[a] == shape[b]
```

The diagonal of those two axes is represented as a synthetic axis `t` with extent:

```cpp
diagonal_extent = shape[a];
```

and stride:

```cpp
diagonal_stride = stride[a] + stride[b];
```

The surviving axes retain their input strides. The trace is then:

```text
out[survivors...] = sum_t input_view[survivors..., t]
```

where:

```text
input_view offset =
    base_offset
  + sum_i survivor_index[i] * survivor_stride[i]
  + t * diagonal_stride
```

So trace is a partial reduction over a strided view.

## Required Dense Primitives

This motivates two important dense tensor primitives:

- arbitrary or synthetic strided views;
- reduction over one or more axes.

With those primitives, matrix diagonal, vector sum, tensor trace, partial trace, and many related operations are all special cases of the same mechanism.

For example:

- diagonal of a matrix: synthetic diagonal view, no reduction;
- trace of a matrix: synthetic diagonal view, reduce the diagonal axis;
- trace of an n-leg tensor: synthetic diagonal view, reduce the diagonal axis, keep the surviving axes;
- sum of vector elements: ordinary one-axis reduction.

## Layout Optimization

The public output axis order should normally be the input axis order with the traced axes removed. That logical order does not have to be the same as the physical traversal order used by the backend.

A dense trace implementation should build survivor records:

```cpp
struct SurvivorAxis {
  int logical_axis;
  index_type extent;
  index_type input_stride;
};
```

Then, for backend execution:

1. Drop extent-1 survivor axes from the kernel layout, while preserving them in output metadata.
2. Sort the remaining survivor axes by increasing `input_stride`.
3. Coalesce adjacent physical axes when:

   ```cpp
   next.input_stride == current.input_stride * current.extent
   ```

4. Use the coalesced layout for the reduction kernel.
5. Represent the returned tensor with the required logical output order, either directly or as a view/permutation where possible.

For non-coalescable survivor axes, increasing input stride is the natural default physical order. It gives the smallest-stride input dimension the fastest-varying output coordinate, improving locality without changing logical semantics.

## Backend Mapping

A custom kernel can implement this directly, but the primitive is also a natural match for backend libraries that support strided reductions.

For example, cuTENSOR has tensor descriptors with explicit strides and a generic partial reduction operation. A dense GPU trace can be expressed as a reduction over the synthetic diagonal axis whose stride is `stride[a] + stride[b]`. This is a better primitive than materializing an identity tensor or routing trace through a general contraction.

Backend selection can still choose a custom kernel for unsupported dtypes, small fixed cases, or builds without the relevant backend library. The design point is that the operation being selected is a strided reduction, not a special trace kernel.

## Symmetric Tensors

Trace at the level of a block-sparse quantum-number-aware tensor is a different operation. It may involve sector constraints, dual leg conventions, block matching, and representation data.

The symmetric trace should therefore lower into a set of dense trace/reduction operations over blocks after the relevant sector logic has been resolved. The dense primitive layer should not be burdened with quantum-number semantics.
