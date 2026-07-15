# Python Dtype Promotion Policy for Uni20

Status: design note, not current API behavior. Uni20's current Python bindings expose only lightweight smoke-test and build-metadata helpers. This note records how NumPy, the Python Array API standard, and SciPy treat mixed precision array operations, and turns those rules into a proposed policy for future Uni20 tensor bindings.

## Motivation

Tensor libraries need predictable dtype behavior. A user who allocates a `float32` tensor usually cares about at least one of memory footprint, bandwidth, GPU throughput, interoperability with model parameters, or reproducibility against other `float32` code. A scalar literal such as `2.0` should not silently promote the whole tensor to `float64` unless the API has explicitly promised that behavior.

This matters especially for tensor-network algorithms. Normalization, Krylov recurrences, time evolution, and truncation routines frequently combine arrays with host scalars. The host scalar may be computed in higher precision for stability, but applying it to a `float32` tensor should normally preserve the tensor dtype unless the algorithm has explicitly chosen a wider working dtype.

## NumPy 2.x Behavior

NumPy documents dtype promotion as finding a common dtype for operations mixing different dtypes. For two NumPy array dtypes, the result usually has a kind and precision at least as high as the inputs. For example, `float32 + float16` gives `float32`, and `int8 + int64` gives `int64`.

The important rule for Uni20 is NumPy's handling of Python scalars after NumPy 2.0, formalized by [NEP 50: Promotion rules for Python scalars](https://numpy.org/neps/nep-0050-scalar-promotion.html) and documented in [Data type promotion in NumPy](https://numpy.org/doc/stable/reference/arrays.promotion.html). When a NumPy array is combined with a Python scalar, NumPy generally considers the scalar kind but ignores the scalar precision. The array dtype controls the result precision.

For example, NumPy documents this behavior:

```python
arr_float32 = np.array([1, 2.5, 2.1], dtype="float32")
arr_float32 + 10.0
# dtype remains float32
```

The design reason is clear: promoting a `float32` array to `float64` merely because the scalar literal is a Python `float` is usually undesirable. In Python, `10.0` is a double-precision host scalar, but for array arithmetic it is treated as a value to be coerced into the array dtype when that is compatible.

There are still traps in NumPy:

- Python scalar precision can be ignored, so low-precision arrays may overflow or round where a user expected a wider result.
- Integer operations may overflow without warnings for arrays.
- Mixed signed and unsigned integer promotion can produce surprising results, including `float64` for combinations such as `int64` and `uint64`.
- Integer arrays combined with Python `float` or `complex` have special behavior.
- Reductions such as `sum` and `prod` have their own accumulator rules.

Even with these traps, NumPy has an explicit policy and documentation. The important lesson is not to copy every NumPy corner case blindly, but to define and test Uni20's policy rather than letting it emerge accidentally from implementation details.

## Python Array API Standard

The [Python Array API standard type promotion rules](https://data-apis.org/array-api/latest/API_specification/type_promotion.html) are even more directly relevant for a modern array library. The standard specifies promotion tables for array-array operations and separately specifies how Python scalars interact with arrays.

For array and Python scalar operations, the standard says that if the scalar is compatible with the array dtype, the expected behavior is equivalent to converting the scalar to a zero-dimensional array with the same dtype as the array, then performing the array operation.

For real floating arrays:

```text
float32 array <op> Python int/float scalar -> scalar converted to float32
float64 array <op> Python int/float scalar -> scalar converted to float64
```

For complex scalars with real floating arrays, the standard promotes to the complex dtype with the same precision:

```text
float32 array <op> Python complex scalar -> complex64
float64 array <op> Python complex scalar -> complex128
```

This is a good policy for Uni20. It preserves the array precision by default, while still allowing real-to-complex promotion when the scalar value genuinely requires a complex dtype.

## SciPy Linear Algebra

SciPy linear algebra generally dispatches to BLAS/LAPACK routines based on array dtype. The [`scipy.linalg.get_lapack_funcs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.get_lapack_funcs.html) documentation states that LAPACK type prefixes are selected from the array dtypes: `s`, `d`, `c`, and `z` for `float32`, `float64`, `complex64`, and `complex128`.

This means that SciPy's low-level linear algebra boundary is dtype-aware. Passing `float32` arrays selects single-precision LAPACK routines where available. Passing `complex64` selects complex single-precision routines. The implementation may still use algorithm-specific work arrays or promote in selected high-level functions, but the general model is that array dtype is a meaningful numerical contract.

For Uni20, this suggests that typed kernel dispatch should be based on tensor dtype, not on incidental host scalar types. If an algorithm chooses a wider internal dtype, that should be an explicit algorithm policy with diagnostics, not a side effect of dividing by a host `double`.

## Proposed Uni20 Policy

Uni20 Python tensor arithmetic should follow these rules unless a specific function documents otherwise.

1. Array dtype controls scalar arithmetic precision.

```python
float32_tensor / 2.0      # returns float32
float64_tensor / 2.0      # returns float64
complex64_tensor * 0.5   # returns complex64
complex128_tensor * 0.5  # returns complex128
```

2. Python complex scalars promote real floating tensors to complex at the same precision.

```python
float32_tensor * (1.0j)  # returns complex64
float64_tensor * (1.0j)  # returns complex128
```

3. Tensor-tensor operations use an explicit promotion table.

```text
float32 + float64       -> float64
float32 + complex64     -> complex64
float64 + complex64     -> complex128
complex64 + complex128  -> complex128
```

4. Integer and boolean tensor arithmetic should be conservative.

Integer and boolean tensors are useful for indices, masks, charges, metadata, and small discrete data. They should not silently participate in tensor-network linear algebra as though they were physical scalar fields. Mixed integer/float arithmetic should either follow a documented array-standard rule or require an explicit cast at higher-level numerical boundaries.

5. In-place operations must not silently widen storage.

```python
x = tensor(dtype=float32)
x += 1.0       # preserves float32
x *= 1.0j      # should error or require explicit complex conversion
```

In-place operations may narrow the scalar to the tensor dtype when safe. They should not replace the storage with a wider dtype. If the operation cannot be represented in the existing dtype, it should fail with a clear message.

6. Algorithmic mixed precision must be explicit.

For example, a Krylov exponential may choose to build a small projected matrix in `float64` even when the input vector is `float32`. That is an algorithmic decision. It should be visible in the API or diagnostics:

```text
state dtype: float32
matvec dtype: float32
projected matrix dtype: float64
subspace exponential dtype: float64
output dtype: float32
```

The same algorithm should not accidentally promote the full Krylov basis and matvec path to `float64` merely because a normalization scalar is represented as a host `double`.

## C++/Python Boundary Rules

The Python policy should be supported by C++ API design.

Use typed tensor views at kernel boundaries. Do not funnel Python dtype-specific bindings into a C++ function that erases the dtype and later redispatches through runtime integer tags.

Separate host algorithm scalars from tensor scalar objects. A host scalar used for convergence logic should not automatically determine the dtype of a tensor expression.

Prefer explicit helper names for dtype-preserving operations:

```cpp
scale_preserve_dtype(x, alpha);
divide_preserve_dtype(x, norm);
axpy_preserve_dtype(y, alpha, x);
```

Avoid generic arithmetic when the numerical policy matters:

```cpp
// Bad if alpha is a host double and x is float32.
x = x / alpha;

// Better: the policy is visible.
divide_preserve_dtype(x, alpha);
```

If a function intentionally widens a tensor, name that behavior:

```cpp
auto x64 = astype<float64>(x);
auto y = krylov_exponential_high_precision_projection(op, x);
```

## Diagnostics

Silent dtype changes are numerical behavior, not implementation details. Uni20 solvers should make dtype choices observable.

For iterative linear algebra, diagnostics should include:

- input dtype;
- operator/matvec dtype;
- working vector dtype;
- small projected matrix dtype;
- scalar reduction dtype;
- output dtype;
- backend/device;
- iteration count;
- stopping reason;
- final residual or error estimate.

These diagnostics need not be printed by default, but they should be available programmatically. Debug or verbose modes may display a compact summary.

## Recommended Initial Promotion Table

For floating and complex tensor dtypes:

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

For tensor and Python scalar arithmetic:

| tensor dtype | Python scalar kind | result |
|---|---|---|
| `float32` | `int` or `float` | `float32` |
| `float32` | `complex` | `complex64` |
| `float64` | `int` or `float` | `float64` |
| `float64` | `complex` | `complex128` |
| `complex64` | `int`, `float`, or `complex` | `complex64` |
| `complex128` | `int`, `float`, or `complex` | `complex128` |

This is close to the Array API scalar rule and NumPy 2.x scalar-promotion behavior.

## Open Questions

- Should Uni20 support integer tensors as arithmetic tensors, or restrict them to indices, masks, charges, and metadata?
- Should Python `float` with integer tensors promote to floating point, error, or follow NumPy exactly?
- Should reductions on `float32` accumulate in `float32`, `float64`, or expose both options?
- Should Krylov algorithms default to `float64` projected matrices for `float32` vectors?
- Should GPU reductions return host scalars in double precision for `float32` input, or preserve precision unless explicitly requested?

The first implementation should choose a narrow, documented policy and test it thoroughly. It is easier to relax strict rules later than to recover from silent promotion behavior that users accidentally depend on.

## References

- [NumPy: Data type promotion in NumPy](https://numpy.org/doc/stable/reference/arrays.promotion.html)
- [NumPy NEP 50: Promotion rules for Python scalars](https://numpy.org/neps/nep-0050-scalar-promotion.html)
- [Python Array API standard: Type Promotion Rules](https://data-apis.org/array-api/latest/API_specification/type_promotion.html)
- [SciPy: `scipy.linalg.get_lapack_funcs`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.get_lapack_funcs.html)
