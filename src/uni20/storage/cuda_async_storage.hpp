#pragma once

/**
 * \file cuda_async_storage.hpp
 * \ingroup tensor
 * \brief Non-blocking CUDA Tensor storage and opaque mdspan descriptors.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace uni20::cuda
{

/// \brief Non-owning opaque view into one CUDA buffer.
/// \details The view carries allocation identity and an element offset, but
///          deliberately exposes no raw device pointer or host-side value
///          conversion. CUDA lowering must first acquire synchronized access
///          to `buffer()` on an operation stream.
template <class ElementType> class CudaBufferView {
  public:
    using element_type = ElementType;
    using value_type = std::remove_const_t<element_type>;
    using buffer_type = CudaBuffer<value_type>;
    using buffer_reference = std::conditional_t<std::is_const_v<element_type>, buffer_type const&, buffer_type&>;

    constexpr CudaBufferView() noexcept = default;

    /// \brief Construct a view at the beginning of a CUDA buffer.
    explicit constexpr CudaBufferView(buffer_reference buffer) noexcept : buffer_(&buffer) {}

    /// \brief Convert a mutable view into a read-only view.
    template <class OtherElement>
      requires(std::is_const_v<element_type> && !std::is_const_v<OtherElement> &&
               std::same_as<std::remove_const_t<OtherElement>, value_type>)
    constexpr CudaBufferView(CudaBufferView<OtherElement> other) noexcept
        : buffer_(other.buffer_), offset_(other.offset_)
    {}

    /// \brief Return whether this view identifies a CUDA buffer.
    [[nodiscard]] explicit constexpr operator bool() const noexcept { return buffer_ != nullptr; }

    /// \brief Return the CUDA buffer whose completion ledger governs access.
    [[nodiscard]] auto buffer() const -> buffer_reference
    {
      CHECK(buffer_ != nullptr, "cannot resolve an empty CUDA buffer view");
      return *buffer_;
    }

    /// \brief Return this view's element offset from the allocation start.
    [[nodiscard]] constexpr std::size_t element_offset() const noexcept { return offset_; }

    /// \brief Return a view advanced by an element offset.
    [[nodiscard]] constexpr CudaBufferView offset_by(std::size_t offset) const noexcept
    {
      CudaBufferView result = *this;
      result.offset_ += offset;
      return result;
    }

    friend constexpr bool operator==(CudaBufferView const&, CudaBufferView const&) = default;

  private:
    using buffer_pointer = std::add_pointer_t<std::remove_reference_t<buffer_reference>>;

    buffer_pointer buffer_ = nullptr;
    std::size_t offset_ = 0;

    template <class> friend class CudaBufferView;
};

/// \brief Mdspan accessor for non-blocking CUDA Tensor storage.
/// \details Indexed access produces another opaque buffer view. It never
///          dereferences device memory or performs an implicit transfer.
template <class ElementType> struct AsyncAccessor
{
    using element_type = ElementType;
    using reference = CudaBufferView<element_type>;
    using data_handle_type = reference;
    using offset_policy = AsyncAccessor;
    using offset_type = std::size_t;

    [[nodiscard]] constexpr reference access(data_handle_type handle, offset_type offset) const noexcept
    {
      return handle.offset_by(offset);
    }

    [[nodiscard]] constexpr data_handle_type offset(data_handle_type handle, offset_type offset) const noexcept
    {
      return handle.offset_by(offset);
    }
};

/// \brief Factory that resolves CUDA async accessors for Tensor descriptors.
struct AsyncAccessorFactory
{
    template <class ElementType> using accessor_t = AsyncAccessor<ElementType>;

    template <class ElementType, class Storage>
    [[nodiscard]] constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }
};

} // namespace uni20::cuda

namespace uni20
{

/// \brief Tensor storage policy for non-blocking CUDA execution.
/// \details Allocation requires an explicit `cuda::DeviceContext`. The policy
///          selects CUDA backends and exposes opaque async CUDA handles; stream
///          and provider-resource acquisition remains operation-local.
struct CudaAsyncStorage
{
    using context_type = cuda::DeviceContext;
    using accessor_factory_type = cuda::AsyncAccessorFactory;
    using backend_selector_type = linalg::backend_list<linalg::CublasBackend>;

    template <class ElementType> using storage_t = cuda::CudaBuffer<ElementType>;

    template <class ElementType>
    [[nodiscard]] static auto make_storage(context_type& context, std::size_t size) -> storage_t<ElementType>
    {
      return storage_t<ElementType>{context, size};
    }

    template <class ElementType>
    [[nodiscard]] static auto make_storage_like(storage_t<ElementType> const& storage,
                                                std::size_t size) -> storage_t<ElementType>
    {
      return storage_t<ElementType>{storage.context(), size};
    }

    template <class ElementType>
    [[nodiscard]] static auto make_handle(storage_t<ElementType>& storage) noexcept -> cuda::CudaBufferView<ElementType>
    {
      return cuda::CudaBufferView<ElementType>{storage};
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_handle(storage_t<ElementType> const& storage) noexcept -> cuda::CudaBufferView<ElementType const>
    {
      return cuda::CudaBufferView<ElementType const>{storage};
    }

    /// \brief Return the ordered CUDA backend list for async device storage.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CublasBackend{}};
    }
};

template <class ElementType>
inline constexpr bool enable_backend_writable_accessor<cuda::AsyncAccessor<ElementType>> =
    !std::is_const_v<ElementType>;

} // namespace uni20
