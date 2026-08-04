#pragma once

/**
 * \file mdspec_traits.hpp
 * \ingroup linalg
 * \brief Shared CUDA reference-backend traits for raw buffer mdspecs.
 */

#include <uni20/storage/cuda_storage.hpp>

#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{

template <class Accessor> struct RawCudaAccessorTraits
{
    static constexpr bool is_raw = false;
    static constexpr bool is_mutable = false;
    static constexpr bool is_const = false;
};

template <class ElementType> struct RawCudaAccessorTraits<uni20::cuda::CudaPointerAccessor<ElementType>>
{
    static constexpr bool is_raw = true;
    static constexpr bool is_mutable = !std::is_const_v<ElementType>;
    static constexpr bool is_const = std::is_const_v<ElementType>;
};

template <class Mdspec>
inline constexpr bool is_raw_cuda_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    RawCudaAccessorTraits<typename std::remove_cvref_t<Mdspec>::accessor_type>::is_raw;

template <class Mdspec>
inline constexpr bool is_raw_mutable_cuda_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    RawCudaAccessorTraits<typename std::remove_cvref_t<Mdspec>::accessor_type>::is_mutable;

template <class Mdspec>
inline constexpr bool is_raw_const_cuda_mdspec =
    uni20::cuda::BufferMdspec<Mdspec> &&
    RawCudaAccessorTraits<typename std::remove_cvref_t<Mdspec>::accessor_type>::is_const;

} // namespace uni20::linalg::detail::cuda_reference
