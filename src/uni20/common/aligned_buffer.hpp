#pragma once

/**
 * \file aligned_buffer.hpp
 * \brief Aligned, (un)initialized and temporary buffer allocation utilities.
 * \defgroup common_utilities Core support utilities
 * \details
 *   This header provides low-overhead, cache-line-aligned buffers, exposed as
 *   `std::unique_ptr<T[], Deleter>` factories:
 *   - `allocate_temporary_buffer<T>(N, align)` constructs elements when
 *     required and cleans them up automatically.
 *   - `allocate_temporary_buffer_uninitialized<T>(N, align)` returns raw
 *     storage but still ensures destructors run on release.
 *   - `allocate_uninitialized_buffer<T>(N, align)` exposes raw storage without
 *     ever invoking constructors or destructors.
 * \note All returned unique_ptr instances are aligned to `align` bytes (default 64).
 */

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

namespace uni20
{
namespace detail
{

/// \brief Stateless deleter that frees aligned memory without invoking destructors.
/// \tparam T Element type held in the buffer.
/// \ingroup internal
template <typename T> struct aligned_deleter
{
    void operator()(T* p) const noexcept
    {
      if (!p) return;
#if defined(_MSC_VER)
      _aligned_free(p);
#else
      std::free(p);
#endif
    }
};

/// \brief Deleter that destroys each element before freeing aligned memory.
/// \tparam T Element type held in the buffer.
/// \ingroup internal
template <typename T> struct aligned_destructor_deleter
{
    std::size_t count;
    explicit aligned_destructor_deleter(std::size_t n) noexcept : count(n) {}

    void operator()(T* p) const noexcept
    {
      if (!p) return;
      // invoke destructor on each element
      for (std::size_t i = 0; i < count; ++i)
      {
        p[i].~T();
      }
#if defined(_MSC_VER)
      _aligned_free(p);
#else
      std::free(p);
#endif
    }
};

/// \brief Allocate raw storage with the requested alignment.
/// \param bytes Total number of bytes requested.
/// \param align Desired alignment in bytes.
/// \return Pointer to aligned storage suitable for manual placement new.
/// \throws std::bad_alloc If the allocation fails.
/// \ingroup internal
inline void* allocate_raw(std::size_t bytes, std::size_t align)
{
  void* ptr = nullptr;
#if defined(_MSC_VER)
  ptr = _aligned_malloc(bytes, align);
  if (!ptr) throw std::bad_alloc();
#else
  // posix_memalign only cares that:
  //   - align is pow-2 and ≥ sizeof(void*)
  //   - size can be any value
  if (int err = posix_memalign(&ptr, align, bytes); err)
  {
    throw std::bad_alloc();
  }
#endif

  return ptr;
}

/// \brief Alias for a raw buffer of `T[]` with no-constructor and no-destructor semantics.
/// \tparam T Element type owned by the buffer.
/// \ingroup internal
template <typename T> using aligned_buf_t = std::unique_ptr<T[], aligned_deleter<T>>;

/// \brief Alias for a buffer of `T[]` whose deleter destroys elements before freeing storage.
/// \tparam T Element type owned by the buffer.
/// \ingroup internal
template <typename T> using aligned_buf_with_dtor_t = std::unique_ptr<T[], aligned_destructor_deleter<T>>;

} // namespace detail

/// \brief Opt a type into allocation without construction or destruction.
/// \details The default accepts types that the standard library reports as
///          trivially copyable and trivially destructible. The standard complex
///          specializations are explicitly accepted as a practical extension:
///          Uni20 relies on their scalar-array representation and trivial
///          lifetime behavior even on standard libraries whose type traits do
///          not yet report them as trivially copyable.
/// \tparam T Candidate element type.
/// \ingroup common_utilities
template <typename T>
inline constexpr bool enable_uninitialized_storage =
    std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

template <typename Real>
inline constexpr bool enable_uninitialized_storage<std::complex<Real>> =
    std::is_trivially_destructible_v<std::complex<Real>>;

/// \brief Whether raw allocation may establish storage for `T` without construction.
template <typename T>
concept uninitialized_ok = enable_uninitialized_storage<std::remove_cv_t<T>>;

/// \brief Allocate raw, aligned storage for `T[N]` without running constructors or destructors.
/// \details When `T` is not trivially copyable you must placement-new each element before use
///          and invoke `std::destroy_n` prior to releasing the buffer.
/// \tparam T Element type stored in the buffer.
/// \param N Number of elements to allocate.
/// \param align Byte alignment for the allocation (defaults to 64).
/// \return Unique pointer that releases the raw storage without invoking destructors.
/// \throws std::bad_alloc If allocation fails.
/// \ingroup common_utilities
template <typename T> detail::aligned_buf_t<T> allocate_uninitialized_buffer(std::size_t N, std::size_t align = 64)
{
  if (N == 0) return detail::aligned_buf_t<T>{nullptr, detail::aligned_deleter<T>{}};
  void* raw = detail::allocate_raw(sizeof(T) * N, align);
  return detail::aligned_buf_t<T>(static_cast<T*>(raw), detail::aligned_deleter<T>{});
}

/// \brief Allocate a temporary buffer of `T[N]` aligned to `align`.
/// \details
///   - If `T` is trivially initializable, the buffer is left uninitialized.
///   - Otherwise each `T` is default-constructed and the deleter runs destructors before freeing.
/// \tparam T Element type stored in the buffer.
/// \param N Number of elements to allocate.
/// \param align Alignment in bytes.
/// \return Unique pointer owning the memory and running destructors when needed.
/// \throws std::bad_alloc If allocation fails.
/// \ingroup common_utilities
template <typename T> auto allocate_temporary_buffer(std::size_t N, std::size_t align = 64)
{
  if constexpr (uninitialized_ok<T>)
  {
    // raw memory, no ctors/dtors
    return allocate_uninitialized_buffer<T>(N, align);
  }
  else
  {
    if (N == 0)
    {
      return detail::aligned_buf_with_dtor_t<T>{nullptr, detail::aligned_destructor_deleter<T>{0}};
    }
    // allocate raw
    void* raw = detail::allocate_raw(sizeof(T) * N, align);
    T* ptr = static_cast<T*>(raw);

    // default-construct each element
    for (std::size_t i = 0; i < N; ++i)
    {
      ::new (static_cast<void*>(ptr + i)) T();
    }

    // return with deleter that runs dtors then frees
    return detail::aligned_buf_with_dtor_t<T>(ptr, detail::aligned_destructor_deleter<T>{N});
  }
}

/// \brief Allocate a temporary buffer of `T[N]` that is always uninitialized.
/// \details The deleter destroys elements for non-trivial types, so callers must placement-new
///          each element before use when `T` is not trivially copyable.
/// \tparam T Element type stored in the buffer.
/// \param N Number of elements to allocate.
/// \param align Alignment in bytes.
/// \return Unique pointer owning the memory and running destructors when needed.
/// \throws std::bad_alloc If allocation fails.
/// \ingroup common_utilities
template <typename T> auto allocate_temporary_buffer_uninitialized(std::size_t N, std::size_t align = 64)
{
  if constexpr (uninitialized_ok<T>)
  {
    // trivially‐copyable → raw storage, no dtors
    return allocate_uninitialized_buffer<T>(N, align);
  }
  else
  {
    if (N == 0)
    {
      return detail::aligned_buf_with_dtor_t<T>{nullptr, detail::aligned_destructor_deleter<T>{0}};
    }
    void* raw = detail::allocate_raw(sizeof(T) * N, align);
    T* ptr = static_cast<T*>(raw);
    // *** no placement‐new here! user must do it. ***
    return detail::aligned_buf_with_dtor_t<T>(ptr, detail::aligned_destructor_deleter<T>{N});
  }
}

} // namespace uni20
