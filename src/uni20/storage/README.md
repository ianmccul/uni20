# src/uni20/storage

This directory contains storage policy types used by tensor objects and views.
Storage policies describe allocation, handles, default layouts, and default
backend tags without owning tensor mathematics.

## Contents

- `vectorstorage.hpp`: host `std::vector` storage policy with layout-stride
  mapping and a CPU default backend tag.
- `generated_storage.hpp`: compact backend-neutral policy for read-only values
  calculated by an accessor instead of stored element-by-element.

## Notes

- New storage policies should make handle creation, layout defaults, and default
  backend selection explicit.
- `GeneratedStorage` marks compact read-only tensors whose accessors calculate
  values instead of addressing an element allocation. It is backend-neutral
  when combined with concrete storage operands.
- Do not hide host/device transfers or synchronization inside storage policy
  hooks; higher layers need those effects to remain visible.

## Related Documentation

- [Source tree map](../README.md)
- [Storage kind and location](../../../docs/architecture/storage_kind_and_location.md)
- [Tensor creation and reshape](../../../docs/tensor/creation_and_reshape.md)
- [Async storage and identity](../../../docs/async/storage.md)
