#include "common/aligned_buffer.hpp"
#include <complex>
#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>

static_assert(uni20::uninitialized_ok<std::complex<double>>, "complex<double> should be trivially copyable");

// A helper type that counts how many times its ctor and dtor run.
struct Tracker
{
    static inline int constructions = 0;
    static inline int destructions = 0;

    Tracker() noexcept { ++constructions; }
    ~Tracker() noexcept { ++destructions; }
};
static_assert(!uni20::uninitialized_ok<Tracker>, "Tracker has a non-trivial dtor, so uninitialized_ok must be false");

// Convenience alias for the detail namespace
namespace detail = uni20::detail;

//-----------------------------------------------------------------------------
// 1) Non-trivial: allocate_temporary_buffer => ctor N times, then dtor N times
//-----------------------------------------------------------------------------
TEST(AlignedBuffer, TemporaryNonTrivial_CtorThenDtor)
{
  constexpr int N = 7;
  Tracker::constructions = 0;
  Tracker::destructions = 0;

  {
    auto buf = uni20::allocate_temporary_buffer<Tracker>(N);
    // constructor should have run N times, but no destructor yet
    EXPECT_EQ(Tracker::constructions, N);
    EXPECT_EQ(Tracker::destructions, 0);
  }
  // leaving scope should run destructor N times
  EXPECT_EQ(Tracker::destructions, N);
}

//-----------------------------------------------------------------------------
// 2) Non-trivial, uninitialized: no ctor, but dtor N times
//-----------------------------------------------------------------------------
TEST(AlignedBuffer, TemporaryUninitializedNonTrivial_OnlyDtor)
{
  constexpr int N = 5;
  Tracker::constructions = 0;
  Tracker::destructions = 0;

  {
    auto buf = uni20::allocate_temporary_buffer_uninitialized<Tracker>(N);
    // uninitialized path: no ctor
    EXPECT_EQ(Tracker::constructions, 0);
    EXPECT_EQ(Tracker::destructions, 0);
  }
  // but destructor deleter still runs on each element
  EXPECT_EQ(Tracker::destructions, static_cast<int>(N));
}

//-----------------------------------------------------------------------------
// 3) Non-trivial, raw uninitialized: neither ctor nor dtor
//-----------------------------------------------------------------------------
TEST(AlignedBuffer, UninitializedBufferNonTrivial_NoCtorNoDtor)
{
  constexpr std::size_t N = 3;
  Tracker::constructions = 0;
  Tracker::destructions = 0;

  {
    auto buf = uni20::allocate_uninitialized_buffer<Tracker>(N);
    EXPECT_EQ(Tracker::constructions, 0);
    EXPECT_EQ(Tracker::destructions, 0);
  }
  EXPECT_EQ(Tracker::constructions, 0);
  EXPECT_EQ(Tracker::destructions, 0);
}

//-----------------------------------------------------------------------------
// 4) Trivial-copyable: std::complex<double> should use the no-dtor deleter
//    We check this at compile time via static_assert on the unique_ptr's
//    deleter_type.
//-----------------------------------------------------------------------------
TEST(AlignedBuffer, TemporaryTrivial_UsesNoDtorDeleter)
{
  using T = std::complex<double>;
  constexpr std::size_t N = 4;

  // The buffer type returned:
  using Buf = decltype(uni20::allocate_temporary_buffer<T>(N));
  static_assert(std::is_same_v<typename Buf::deleter_type, detail::aligned_deleter<T>>,
                "Trivial path must use aligned_deleter<T>");

  // And at runtime it must at least allocate/free without crashing:
  {
    auto buf = uni20::allocate_temporary_buffer<T>(N);
    (void)buf;
  }
}

TEST(AlignedBuffer, TemporaryUninitializedTrivial_UsesNoDtorDeleter)
{
  using T = std::complex<double>;
  constexpr std::size_t N = 6;

  using Buf = decltype(uni20::allocate_temporary_buffer_uninitialized<T>(N));
  static_assert(std::is_same_v<typename Buf::deleter_type, detail::aligned_deleter<T>>,
                "Uninitialized trivial path must use aligned_deleter<T>");

  {
    auto buf = uni20::allocate_temporary_buffer_uninitialized<T>(N);
    (void)buf;
  }
}

//-----------------------------------------------------------------------------
// 5) Alignment behavior: small vs large allocations
//-----------------------------------------------------------------------------
TEST(AlignedBuffer, AlignmentOfUninitializedSmall)
{
  // For bytes < align, we shrink alignment to bit_floor(bytes)
  auto buf = uni20::allocate_uninitialized_buffer<double>(1);
  auto ptr = reinterpret_cast<std::uintptr_t>(buf.get());

  // sizeof(double)==8, so alignment should be max(8, alignof(void*))==8
  EXPECT_EQ(ptr % 8u, 0u);
}

TEST(AlignedBuffer, AlignmentOfUninitializedLarge)
{
  // For 100 doubles, bytes=800>64, so alignment remains 64
  auto buf = uni20::allocate_uninitialized_buffer<double>(100);
  auto ptr = reinterpret_cast<std::uintptr_t>(buf.get());
  EXPECT_EQ(ptr % 64u, 0u);
}
