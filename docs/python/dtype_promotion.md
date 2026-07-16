# Python Dtype Promotion

Status: design note for future Python Tensor arithmetic. The current Python
extension does not yet expose Tensor values.

This document proposes a promotion contract. The contract becomes authoritative
only when Tensor arithmetic is implemented and covered by binding tests.
Operation wrappers should then share one policy rather than defining independent
promotion rules.

## Scope

The initial numerical contract covers enabled real floating and complex
floating Tensor dtypes. Integer and boolean tensors may be added for indices,
masks, charges, and selected arithmetic, but mixed numerical behavior is not
implicitly enabled until it has its own table and tests.

The set of exposed dtypes is generated from one binding scalar list. Optional
extended precision may participate in Uni20-to-Uni20 operations when configured,
but NumPy and DLPack interoperability is exposed only where those protocols have
a compatible dtype contract.

## Core Rules

1. Tensor-tensor arithmetic uses an explicit promotion table.
2. Python scalar precision is weak relative to a floating Tensor dtype.
3. A Python complex scalar promotes a real Tensor to the corresponding complex
   precision.
4. In-place operations do not replace storage with a wider dtype.
5. Integer true division produces a floating result according to an explicit
   operation rule.
6. Reductions and algorithms may choose a separate accumulator or working
   precision, but must document and report it.
7. Backend availability does not change the mathematical result dtype.

These rules follow NumPy 2.x and the Python Array API where that behavior is
appropriate, without inheriting unspecified or accidental corner cases.

## Tensor-Tensor Promotion

For the initial 32-bit and 64-bit floating families:

| lhs | rhs | result |
|---|---|---|
| `float32` | `float32` | `float32` |
| `float32` | `float64` | `float64` |
| `float32` | `complex64` | `complex64` |
| `float32` | `complex128` | `complex128` |
| `float64` | `float64` | `float64` |
| `float64` | `complex64` | `complex128` |
| `float64` | `complex128` | `complex128` |
| `complex64` | `complex64` | `complex64` |
| `complex64` | `complex128` | `complex128` |
| `complex128` | `complex128` | `complex128` |

The table is symmetric. An enabled real or complex extended-precision dtype
extends the same precision lattice, subject to operation and backend support.

Result dtype is computed before kernel selection. A provider decline may select
another backend, but it does not silently narrow or widen the result.

## Python Scalars

For Tensor and ordinary Python scalar arithmetic:

| Tensor dtype | Python scalar kind | result |
|---|---|---|
| `float32` | `int` or `float` | `float32` |
| `float32` | `complex` | `complex64` |
| `float64` | `int` or `float` | `float64` |
| `float64` | `complex` | `complex128` |
| `complex64` | `int`, `float`, or `complex` | `complex64` |
| `complex128` | `int`, `float`, or `complex` | `complex128` |

The scalar value must still be representable according to the operation's
conversion policy. Preserving `float32` does not require silently accepting a
Python integer too large to convert meaningfully.

NumPy scalar objects and zero-dimensional arrays are array-like typed values,
not untyped Python scalars. Their explicit dtype participates in the
tensor-tensor promotion rule.

## In-Place Arithmetic

In-place arithmetic preserves the existing Tensor dtype and allocation:

```python
x = uni20.ones((10,), dtype=uni20.float32)
x += 1.0      # float32
x *= 1.0j     # error: would require complex storage
```

The right operand may be converted to the existing dtype only when the
operation permits it. In-place arithmetic never replaces storage merely to
accommodate a promoted result.

## Division and Reductions

True division has an operation-specific result rule:

- floating and complex inputs use the normal promotion table;
- integer and boolean true division is unavailable until its result dtype is
  explicitly selected;
- floor division, remainder, and comparison require separate contracts.

Reduction result and accumulator dtype are separate questions. Each reduction
must specify:

- input dtype;
- accumulator dtype;
- returned scalar or Tensor dtype;
- device/host result placement;
- empty-input behavior.

No default accumulator widening is inferred from an incidental C++ host scalar.

## Algorithmic Working Precision

An algorithm may use a wider internal scalar for error estimation, projected
subspaces, or accumulation. That is distinct from promoting the full Tensor
operation.

For example, diagnostics for a Krylov operation may report:

```text
input dtype: float32
matvec dtype: float32
projected matrix dtype: float64
output dtype: float32
```

Such widening is an explicit algorithm policy. It does not arise because a
normalization scalar happened to be represented as C++ `double`.

## C++ Binding Boundary

Python scalar conversion is centralized. Individual bindings do not separately
interpret `int`, `float`, `complex`, NumPy scalars, or zero-dimensional arrays.

Typed native operations remain free to use distinct scalar types for output and
inputs:

```cpp
template <class Output, class Left, class Right>
void binary_operation(Output& output, Left const& lhs, Right const& rhs);
```

The binding does not cast both inputs to the output dtype merely to fit a
single-dtype implementation. An operation materializes converted inputs only
when its documented algorithm requires that conversion.

## Diagnostics and Tests

Tests cover every exposed pair of Tensor dtypes, operand order, Python scalar
kind, in-place rejection, backend fallback, and configured extended-precision
case.

Numerical diagnostics should make visible:

- input dtypes;
- output dtype;
- accumulator or working dtype where different;
- backend and device;
- any explicit materialization or conversion.

## Deferred Extensions

- Integer and boolean arithmetic tables.
- Reduction accumulator defaults.
- Python exposure and external interchange policy for binary128.
- Low-precision `float16` and `bfloat16` families.
- Quantized or packed scalar types.

These extensions should add rows to the shared contract rather than creating
operation-local promotion rules.

## References

- [NumPy NEP 50: Promotion rules for Python scalars](https://numpy.org/neps/nep-0050-scalar-promotion.html)
- [NumPy data type promotion](https://numpy.org/doc/stable/reference/arrays.promotion.html)
- [Python Array API type promotion](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
