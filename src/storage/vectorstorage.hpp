#pragma once

#include "backend/backends.hpp"
#include <vector>

namespace uni20
{

struct VectorStorage
{
    template <typename ElementType> struct storage_t = std::vector<ElementType>;

    using tag_t = cpu_tag;
};

} // namespace uni20
