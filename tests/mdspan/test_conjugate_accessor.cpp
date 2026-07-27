#include <uni20/core/types.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

namespace
{
using extents_1d = stdex::dextents<uni20::index_type, 1>;
using extents_2d = stdex::dextents<uni20::index_type, 2>;
} // namespace

TEST(MdspanConjugateAccessor, ComplexConjViewConjugatesValues)
{
  using complex_type = uni20::complex<double>;

  std::vector<complex_type> storage{{1.0, 2.0}, {3.0, -4.0}, {-5.0, 6.0}, {-7.0, -8.0}};
  stdex::mdspan<complex_type, extents_2d, stdex::layout_left> span(storage.data(), 2, 2);

  auto conjugated = uni20::conj(span);
  static_assert(uni20::StridedMdspanLike<decltype(conjugated)>);
  static_assert(!uni20::MutableMdspanLike<decltype(conjugated)>);
  static_assert(std::is_const_v<typename decltype(conjugated)::element_type>);
  static_assert(uni20::accessor_applies_conjugation_v<typename decltype(conjugated)::accessor_type>);
  static_assert(uni20::mdspan_needs_conjugation_v<decltype(conjugated)>);

  EXPECT_EQ(conjugated.data_handle(), static_cast<complex_type const*>(storage.data()));
  EXPECT_EQ((conjugated[0, 0]), complex_type(1.0, -2.0));
  EXPECT_EQ((conjugated[1, 0]), complex_type(3.0, 4.0));
  EXPECT_EQ((conjugated[0, 1]), complex_type(-5.0, -6.0));
  EXPECT_EQ((conjugated[1, 1]), complex_type(-7.0, 8.0));
}

TEST(MdspanConjugateAccessor, DoubleConjCancelsConjugatingAccessor)
{
  using complex_type = uni20::complex<double>;

  std::vector<complex_type> storage{{1.0, 2.0}, {3.0, -4.0}, {-5.0, 6.0}, {-7.0, -8.0}};
  stdex::mdspan<complex_type, extents_2d, stdex::layout_left> span(storage.data(), 2, 2);

  auto conjugated = uni20::conj(span);
  auto roundtrip = uni20::conj(conjugated);

  static_assert(uni20::StridedMdspanLike<decltype(roundtrip)>);
  static_assert(!uni20::MutableMdspanLike<decltype(roundtrip)>);
  static_assert(std::is_same_v<decltype(roundtrip), stdex::mdspan<complex_type const, extents_2d, stdex::layout_left>>);
  static_assert(!uni20::accessor_applies_conjugation_v<typename decltype(roundtrip)::accessor_type>);
  static_assert(!uni20::mdspan_needs_conjugation_v<decltype(roundtrip)>);

  EXPECT_EQ(roundtrip.data_handle(), static_cast<complex_type const*>(storage.data()));
  EXPECT_EQ((roundtrip[0, 0]), storage[0]);
  EXPECT_EQ((roundtrip[1, 0]), storage[1]);
  EXPECT_EQ((roundtrip[0, 1]), storage[2]);
  EXPECT_EQ((roundtrip[1, 1]), storage[3]);
}

TEST(MdspanConjugateAccessor, NonComplexConjReturnsConstIdentityView)
{
  std::vector<double> storage{1.0, 2.0, 3.0};
  stdex::mdspan<double, extents_1d, stdex::layout_left> span(storage.data(), 3);

  auto identity = uni20::conj(span);

  static_assert(std::is_same_v<decltype(identity), stdex::mdspan<double const, extents_1d, stdex::layout_left>>);
  static_assert(!uni20::MutableMdspanLike<decltype(identity)>);
  static_assert(!uni20::mdspan_needs_conjugation_v<decltype(identity)>);

  EXPECT_EQ(identity.data_handle(), static_cast<double const*>(storage.data()));
  EXPECT_EQ(identity[0], 1.0);
  EXPECT_EQ(identity[1], 2.0);
  storage[1] = 42.0;
  EXPECT_EQ(identity[1], 42.0);
  EXPECT_EQ(storage[1], 42.0);
}
