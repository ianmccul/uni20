# src/uni20/level1

This directory contains simple dense elementwise and reduction primitives. These
are intended as small building blocks for tensor and linalg code once storage,
layout, and dispatch decisions have already been resolved.

## Contents

- `assign.hpp`: dense assignment helpers.
- `transform.hpp`: eager unary/binary overwrite and in-place transform helpers.
- `zip_transform.hpp`: elementwise binary/zip transform helpers.
- `sum.hpp`: reduction helper.

## Notes

- Keep these primitives generic over scalar and view types where practical.
- Do not add symmetry-specific logic here; block and sector decisions belong in
  the symmetry/tensor lowering layer.

## Related Documentation

- [Source tree map](../)
- [Raw primitives and symmetric lowering](../../../docs/symmetry/raw_primitives_and_lowering.md)
- [Trace as a dense reduction](../../../docs/linalg/trace_reduction.md)
