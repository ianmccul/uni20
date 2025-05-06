#pragma once

/**
 * @file aligned_buffer.hpp
 * @brief Aligned, (un)initialized and temporary buffer allocation utilities.
 *
 * This header provides low-overhead, cache-line-aligned buffers, as
 * std::unique_ptr<T[], deleter>
 *
 *   - allocate_temporary_buffer<T>(N, align)
 *     If T is trivially copyable, gives raw storage; otherwise placement-news
 *     each T and automatically runs dtors on free. This is always safe to
 *     use with no need to use placement-new or manual destruction.
 *
 *   - allocate_temporary_buffer_uninitialized<T>(N, align)
 *     Always gives raw storage, but for non-trivial T will run dtors on free
 *     (caller still must placement-new before use).
 *
 *   - allocate_uninitialized_buffer<T>(N, align)
 *     Returns a unique_ptr<T[]> of raw storage for N Ts—**no** constructors
 *     or destructors ever run.  If T requires non-trivial construction or
 *     destruction then the caller must use placement-new and std::destroy_n.
 *
 * @note All returned unique_ptrs are aligned to `align` bytes (default 64).
 */

#include <bit>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>

namespace uni20
{
namespace detail
{

/// \brief Stateless deleter: frees aligned memory, does NOT call any destructors.
/// \tparam T element type
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

/// \brief Deleter which calls each element's destructor, then frees aligned memory.
/// \tparam T element type
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

/// \brief Allocate `bytes` aligned to `align`, but if `bytes < align`,
///        reduce alignment to the largest power-of-two ≤ bytes,
///        with a floor of sizeof(void*).
/// \throws std::bad_alloc
inline void* allocate_raw(std::size_t bytes, std::size_t align)
{
  // if the buffer is smaller than the requested alignment,
  // drop the alignment to avoid wasted space:
  if (bytes < align)
  {
    // bit_floor(bytes) is the largest power-of-two ≤ bytes
    auto pf = std::bit_floor(bytes);
    // never go below pointer alignment:
    align = std::max<std::size_t>(pf, alignof(void*));
  }

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

/// \brief Alias for a raw buffer of `T[]` with no-ctor/no-dtor semantics.
/// Ownership = free via `std::free` / `_aligned_free`.
template <typename T> using aligned_buf_t = std::unique_ptr<T[], aligned_deleter<T>>;

/// \brief Alias for a buffer of `T*` whose deleter will run dtors then free.
template <typename T> using aligned_buf_with_dtor_t = std::unique_ptr<T, aligned_destructor_deleter<T>>;

} // namespace detail

/// \brief True for any T that can be safely copied into raw storage.
/// \note implies trivial copy-ctor, trivial move-ctor, trivial dtor.
template <typename T>
concept uninitialized_ok = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

/// \brief Allocate raw, aligned storage for `T[N]`, **no ctors or dtors run**.
/// \note If `T` is *not* trivially copyable, you must:
///   1. Placement-new each element before use:
///      \code
///      auto buf = allocate_uninitialized_buffer<MyType>(N);
///      MyType* p = buf.get();
///      for(size_t i = 0; i < N; ++i)
///        new (p + i) MyType(/*ctor args*/);
///      \endcode
///   2. When you’re done, call destructors *before* the buffer is freed:
///      \code
///      std::destroy_n(p, N);            // #include <memory>
///      \endcode
/// \tparam T   element type
/// \param N   how many Ts to allocate
/// \param align byte‐alignment (default 64)
/// \returns unique_ptr that will free the raw storage (no dtors)
template <typename T> detail::aligned_buf_t<T> allocate_uninitialized_buffer(std::size_t N, std::size_t align = 64)
{
  void* raw = detail::allocate_raw(sizeof(T) * N, align);
  return detail::aligned_buf_t<T>(static_cast<T*>(raw), detail::aligned_deleter<T>{});
}

/// \brief Allocate a temporary buffer of `T[N]`, aligned to `align`.
///        - If `T` is trivially_initializable, the buffer is left uninitialized.
///        - Otherwise each `T` is default-constructed in-place, and
///          on destruction each dtor is run before freeing.
/// \param N     number of elements
/// \param align alignment in bytes
/// \returns unique_ptr that owns the memory and runs dtors if needed.
/// \throws std::bad_alloc
template <typename T> auto allocate_temporary_buffer(std::size_t N, std::size_t align = 64)
{
  if constexpr (uninitialized_ok<T>)
  {
    // raw memory, no ctors/dtors
    return allocate_uninitialized_buffer<T>(N, align);
  }
  else
  {
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

/// Allocate a temporary buffer of T[N], **always uninitialized**, but
/// for non‐trivial T will call ~~T()~~ on each element when freed.
/// You must placement‐new() each element before use if T isn’t
/// trivially‐copyable.
template <typename T> auto allocate_temporary_buffer_uninitialized(std::size_t N, std::size_t align = 64)
{
  if constexpr (uninitialized_ok<T>)
  {
    // trivially‐copyable → raw storage, no dtors
    return allocate_uninitialized_buffer<T>(N, align);
  }
  else
  {
    void* raw = detail::allocate_raw(sizeof(T) * N, align);
    T* ptr = static_cast<T*>(raw);
    // *** no placement‐new here! user must do it. ***
    return detail::aligned_buf_with_dtor_t<T>(ptr, detail::aligned_destructor_deleter<T>{N});
  }
}

} // namespace uni20
