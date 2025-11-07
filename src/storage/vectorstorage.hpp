#pragma once

#include "common/mdspan.hpp"
#include "kernel/cpu/cpu.hpp"
#include "tensor/layout.hpp"

#include <vector>

namespace uni20
{

struct VectorStorage
{
    template <typename ElementType> using storage_t = std::vector<ElementType>;

    using default_layout_policy = stdex::layout_stride;
    using default_mapping_builder = layout::LayoutRight;

    template <typename ElementType> static auto make_handle(storage_t<ElementType>& storage) noexcept -> ElementType*
    {
      return storage.data();
    }

    template <typename ElementType>
    static auto make_handle(storage_t<ElementType> const& storage) noexcept -> ElementType const*
    {
      return storage.data();
    }

    using tag_t = cpu_tag;
};

} // namespace uni20
