# `src/uni20/storage`

This directory contains storage policy types used by tensor objects and views.
Storage policies describe allocation, handles, default layouts, and default
backend tags without owning tensor mathematics.

## Contents

- `vectorstorage.hpp`: host `std::vector` storage policy with layout-stride
  mapping and a CPU default backend tag.

## Notes

- New storage policies should make handle creation, layout defaults, and default
  backend selection explicit.
- Do not hide host/device transfers or synchronization inside storage policy
  hooks; higher layers need those effects to remain visible.
