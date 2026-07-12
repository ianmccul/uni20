#pragma once

#include <uni20/common/mdspan.hpp>
#include <uni20/config.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/tensor/layout.hpp>

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

#if UNI20_BACKEND_BLAS
    using backend_selector_type =
        linalg::backend_list<linalg::LapackBackend, linalg::BlasBackend, linalg::CpuReferenceBackend>;
#else
    using backend_selector_type = linalg::backend_list<linalg::LapackBackend, linalg::CpuReferenceBackend>;
#endif

    /// \brief Return the default ordered linalg backends for host vector storage.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
#if UNI20_BACKEND_BLAS
      return backend_selector_type{linalg::LapackBackend{}, linalg::BlasBackend{}, linalg::CpuReferenceBackend{}};
#else
      return backend_selector_type{linalg::LapackBackend{}, linalg::CpuReferenceBackend{}};
#endif
    }
};

} // namespace uni20
