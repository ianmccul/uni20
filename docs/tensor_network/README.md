# Tensor-Network Documentation

This directory contains the currently implemented tensor-network foundations
and background from the earlier TensorContraction integration work.

## Current Foundations

- [Sparse Matrices](sparse_matrix.md) documents the row-oriented sparse
  container used by operator prototypes.
- [Local Operators](operators.md), [Models](models.md), and
  [Finite MPS](mps.md) document the present small tensor-network layer.

## Integration Background

- [TensorContraction Integration Findings](contraction_integration_findings.md)
  records conclusions imported from the separate integration lineage.
- [R/A/B/C Contraction Scheduling](rabc_contraction_scheduling.md) develops the
  effective-Hamiltonian scheduling and cost model.
- [R/A/B/C Lanczos Fixtures](rabc_lanczos_fixtures.md) records the capture and
  replay workflow used by that lineage.

The R/A/B/C executables and fixture tooling described by the background notes
are not runnable targets on the current main branch. Their durable architectural
conclusions feed the [architecture](../architecture/README.md) and
[symmetry](../symmetry/README.md) designs.
