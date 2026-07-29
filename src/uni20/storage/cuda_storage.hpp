#pragma once

/**
 * \file cuda_storage.hpp
 * \ingroup tensor
 * \brief CUDA Tensor storage with deferred device-memory data descriptors.
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

/// \brief Accessor for a CUDA mdspan whose device pointer has been leased.
/// \details Indexed access applies the mapped offset and returns an element
///          reference. The accessor must be evaluated only in an execution
///          domain where the leased CUDA pointer is directly accessible.
template <class ElementType> struct CudaPointerAccessor
{
    using element_type = ElementType;
    using data_handle_type = element_type*;
    using reference = element_type&;
    using offset_policy = CudaPointerAccessor;
    using offset_type = std::size_t;

    [[nodiscard]] constexpr reference access(data_handle_type handle, offset_type offset) const noexcept
    {
      return handle[offset];
    }

    [[nodiscard]] constexpr data_handle_type offset(data_handle_type handle, offset_type offset) const noexcept
    {
      if (offset == 0) return handle;
      return handle + offset;
    }
};

} // namespace uni20::cuda

namespace uni20
{

/// \brief CUDA pointer accessors are valid only in the CUDA execution domain.
template <class ElementType>
inline constexpr bool enable_accessor_in_domain<cuda::CudaPointerAccessor<ElementType>, cuda_access_domain> = true;

} // namespace uni20

namespace uni20::cuda
{

/// \brief Factory that resolves CUDA accessors for Tensor descriptors.
struct CudaAccessorFactory
{
    template <class ElementType> using accessor_t = CudaPointerAccessor<ElementType>;
    template <class ElementType> using device_accessor_t = CudaPointerAccessor<ElementType>;

    template <class ElementType, class Storage>
    [[nodiscard]] constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }

    template <class ElementType, class Storage>
    [[nodiscard]] constexpr auto make_device_accessor(Storage const&) const noexcept -> device_accessor_t<ElementType>
    {
      return device_accessor_t<ElementType>{};
    }
};

} // namespace uni20::cuda

namespace uni20
{

/// \brief Tensor storage policy for device-resident CUDA values.
/// \details Ordinary allocation uses the installed CUDA runtime's default
///          device. An explicit `cuda::DeviceResources` selects another enrolled
///          device or an isolated resource set used by tests. The policy selects
///          CUDA backends and exposes deferred `CudaBufferView` descriptors with
///          eventual pointer accessors. Stream and provider-resource admission
///          remains operation-local. Direct Tensor operations may block during
///          admission. `Async<Tensor>` operations use coroutine-aware dispatch
///          when a backend provides it, while retaining the same storage and
///          device-mdspan representation.
struct CudaStorage
{
    using context_type = cuda::DeviceResources;
    using accessor_factory_type = cuda::CudaAccessorFactory;
#if UNI20_BACKEND_CUBLAS
    using backend_selector_type = linalg::backend_list<linalg::CublasBackend, linalg::CudaReferenceBackend>;
#else
    using backend_selector_type = linalg::backend_list<linalg::CudaReferenceBackend>;
#endif

    template <class ElementType> using storage_t = cuda::CudaBuffer<ElementType>;

    template <class ElementType>
    [[nodiscard]] static auto make_storage(context_type& context, std::size_t size) -> storage_t<ElementType>
    {
      return storage_t<ElementType>{context, size};
    }

    template <class ElementType>
    [[nodiscard]] static auto make_storage(cuda::Device device, std::size_t size) -> storage_t<ElementType>
    {
      return storage_t<ElementType>{cuda::device_resources(device.ordinal()), size};
    }

    template <class ElementType>
    [[nodiscard]] static auto make_storage_like(storage_t<ElementType> const& storage,
                                                std::size_t size) -> storage_t<ElementType>
    {
      return storage_t<ElementType>{storage.resources(), size};
    }

    template <class ElementType>
    [[nodiscard]] static bool storage_is_compatible(storage_t<ElementType> const& storage, cuda::Device device)
    {
      return storage.device() == device;
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_data_descriptor(storage_t<ElementType>& storage) noexcept -> cuda::CudaBufferView<ElementType>
    {
      return cuda::CudaBufferView<ElementType>{storage};
    }

    template <class ElementType>
    [[nodiscard]] static auto
    make_data_descriptor(storage_t<ElementType> const& storage) noexcept -> cuda::CudaBufferView<ElementType const>
    {
      return cuda::CudaBufferView<ElementType const>{storage};
    }

    /// \brief Return the ordered backend list for CUDA device storage.
    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
#if UNI20_BACKEND_CUBLAS
      return backend_selector_type{linalg::CublasBackend{}, linalg::CudaReferenceBackend{}};
#else
      return backend_selector_type{linalg::CudaReferenceBackend{}};
#endif
    }
};

} // namespace uni20
