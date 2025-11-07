#pragma once

#include "backend/backends.hpp"
#include <vector>

namespace uni20
{

struct VectorStorage
{
    template <typename ElementType> using storage_t = std::vector<ElementType>;

    template <typename ElementType>
    static auto make_handle(storage_t<ElementType>& storage) noexcept -> ElementType*
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
