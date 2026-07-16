# src/uni20/linalg/async/detail

This directory contains implementation helpers shared by asynchronous linalg
operation wrappers.

## Contents

- `output.hpp`: output-capability and fixed-alias helpers for wrappers that
  write through owner-retaining async tensor descriptors.

These helpers are not standalone operation APIs. Public async linalg entry
points live in the [parent directory](../).

See [Async Tensor Kernel Authoring](../../../../../docs/async/kernel_authoring.md)
for the output, lifetime, and exception-propagation contract.
