#pragma once

/**
 * \file cuda_accessor.hpp
 * \ingroup tensor
 * \brief CUDA-device-callable accessors for leased Tensor storage.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/concepts.hpp>

#include <cuda/std/complex>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace uni20::cuda
{

/// \brief Assignable CUDA execution proxy over one persistent complex value.
/// \details Persistent storage remains `uni20::complex<Real>`. Device reads and
///          writes use `cuda::std::complex<Real>` without treating the two
///          complex class types as alias-compatible objects.
template <class Real> class CudaComplexReference {
  public:
    using logical_type = uni20::complex<Real>;
    using execution_type = ::cuda::std::complex<Real>;

    UNI20_HOST_DEVICE constexpr CudaComplexReference(logical_type* storage, std::size_t offset) noexcept
        : components_(reinterpret_cast<Real*>(storage) + 2 * offset)
    {}

    UNI20_HOST_DEVICE constexpr CudaComplexReference(CudaComplexReference const&) noexcept = default;

    /// \brief Load the persistent components into a CUDA execution value.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr operator execution_type() const noexcept
    {
      return execution_type{components_[0], components_[1]};
    }

    /// \brief Store a CUDA execution value componentwise.
    UNI20_HOST_DEVICE constexpr CudaComplexReference& operator=(execution_type const& value) noexcept
    {
      components_[0] = value.real();
      components_[1] = value.imag();
      return *this;
    }

    /// \brief Copy the observed value rather than rebinding the proxy.
    UNI20_HOST_DEVICE constexpr CudaComplexReference& operator=(CudaComplexReference const& value) noexcept
    {
      return *this = static_cast<execution_type>(value);
    }

    /// \brief Store a host logical value componentwise.
    UNI20_HOST_DEVICE constexpr CudaComplexReference& operator=(logical_type const& value) noexcept
    {
      auto const* components = reinterpret_cast<Real const*>(&value);
      components_[0] = components[0];
      components_[1] = components[1];
      return *this;
    }

  private:
    Real* components_;
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

    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle,
                                                               offset_type offset) const noexcept
    {
      return handle[offset];
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      if (offset == 0) return handle;
      return handle + offset;
    }
};

/// \brief Mutable CUDA accessor for persistent `uni20::complex` storage.
template <class Real> struct CudaPointerAccessor<uni20::complex<Real>>
{
    using element_type = uni20::complex<Real>;
    using data_handle_type = element_type*;
    using reference = CudaComplexReference<Real>;
    using offset_policy = CudaPointerAccessor;
    using offset_type = std::size_t;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle,
                                                               offset_type offset) const noexcept
    {
      return reference{handle, offset};
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      if (offset == 0) return handle;
      return handle + offset;
    }
};

/// \brief Read-only CUDA accessor for persistent `uni20::complex` storage.
template <class Real> struct CudaPointerAccessor<uni20::complex<Real> const>
{
    using element_type = uni20::complex<Real> const;
    using data_handle_type = element_type*;
    using reference = ::cuda::std::complex<Real>;
    using offset_policy = CudaPointerAccessor;
    using offset_type = std::size_t;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle,
                                                               offset_type offset) const noexcept
    {
      auto const* components = reinterpret_cast<Real const*>(handle) + 2 * offset;
      return reference{components[0], components[1]};
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      if (offset == 0) return handle;
      return handle + offset;
    }
};

/// \brief Named CUDA-callable complex conjugation operation.
struct CudaConjugate
{
    template <class Real>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(::cuda::std::complex<Real> value) const noexcept
        -> ::cuda::std::complex<Real>
    {
      return ::cuda::std::conj(value);
    }

    template <class Real>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(CudaComplexReference<Real> value) const noexcept
        -> ::cuda::std::complex<Real>
    {
      using execution_type = ::cuda::std::complex<Real>;
      return ::cuda::std::conj(static_cast<execution_type>(value));
    }
};

/// \brief Read-only CUDA accessor that conjugates persistent complex storage.
template <class Real> struct CudaConjugatingPointerAccessor
{
    using element_type = uni20::complex<Real> const;
    using data_handle_type = element_type*;
    using reference = ::cuda::std::complex<Real>;
    using offset_policy = CudaConjugatingPointerAccessor;
    using offset_type = std::size_t;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle,
                                                               offset_type offset) const noexcept
    {
      return CudaConjugate{}(CudaPointerAccessor<element_type>{}.access(handle, offset));
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      return CudaPointerAccessor<element_type>{}.offset(handle, offset);
    }
};

/// \brief Produce a read-only CUDA pointer accessor without a generic lvalue adaptor.
template <class ElementType>
  requires(!std::is_const_v<ElementType>)
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto const_accessor(CudaPointerAccessor<ElementType> const&)
{
  return CudaPointerAccessor<ElementType const>{};
}

} // namespace uni20::cuda

namespace uni20
{

template <class Real> struct remove_proxy_reference<cuda::CudaComplexReference<Real>>
{
    using type = uni20::complex<Real>;
};

template <class Real> struct logical_value<::cuda::std::complex<Real>>
{
    using type = uni20::complex<Real>;
};

/// \brief CUDA pointer accessors are valid only in the CUDA execution domain.
template <class ElementType>
inline constexpr bool enable_accessor_in_domain<cuda::CudaPointerAccessor<ElementType>, cuda_access_domain> = true;

template <class Real>
inline constexpr bool enable_accessor_in_domain<cuda::CudaConjugatingPointerAccessor<Real>, cuda_access_domain> = true;

} // namespace uni20
