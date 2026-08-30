# src/uni20/storage

This directory contains storage policy types used by tensor objects and views.
Storage policies describe allocation, handles, default layouts, and default
backend tags without owning tensor mathematics.

## Contents

- `host_storage.hpp`: aligned pageable-host storage with uninitialized scalar
  allocation and the default CPU backend list.
- `generated_storage.hpp`: compact backend-neutral policy for read-only values
  calculated by an accessor instead of stored element-by-element.
- `cuda_accessor.hpp`: CUDA-device-callable pointer accessors, complex
  execution-value proxies, and named CUDA transformations.
- `cuda_storage.hpp`: CUDA device-storage policy with deferred
  `CudaBufferView` descriptors and a storage-selected cuBLAS backend.

## Notes

- New storage policies should make immediate-handle or deferred-descriptor
  creation, layout defaults, and default backend selection explicit.
- Context-capable policies provide `context_type`, `make_storage(context,
  size)`, and `make_storage_like(storage, size)`. `CudaStorage` ordinarily
  resolves the installed runtime's default `DeviceResources`; an explicit
  resource set selects another enrolled device or an isolated test setup.
  Shape replacement preserves the original resources.
- `GeneratedStorage` marks compact read-only tensors whose accessors calculate
  values instead of addressing an element allocation. It is backend-neutral
  when combined with concrete storage operands.
- Do not hide host/device transfers or synchronization inside storage policy
  hooks; higher layers need those effects to remain visible.
- Blocking versus coroutine resource admission is an operation-dispatch choice,
  not a storage-policy distinction. Direct `CudaTensor` operations may block
  while acquiring provider resources; `Async<CudaTensor>` operations use an
  available coroutine backend hook through `co_dispatch_kernel`.

## Related Documentation

- [Source tree map](../)
- [Storage kind and location](../../../docs/architecture/storage_kind_and_location.md)
- [CUDA runtime foundation](../../../docs/backends/cuda/runtime.md)
- [Tensor creation and reshape](../../../docs/tensor/creation_and_reshape.md)
- [Async storage and identity](../../../docs/async/storage.md)
