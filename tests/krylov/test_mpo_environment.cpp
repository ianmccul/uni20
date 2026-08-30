#include <uni20/core/math.hpp>
#include <uni20/core/types.hpp>
#include <uni20/tensor_network/environment.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace
{

using RealSite = uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using RealMpo = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                               uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using SiteKey = typename RealSite::key_type;
using MpoKey = typename RealMpo::key_type;

TEST(MpoEnvironmentTest, BuildsIdentityBlocksForEveryBoundarySector)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const bond(symmetry, {{q0, 2}, {q1, 1}}, "boundary");
  uni20::LocalSpace const auxiliary(symmetry, {q0, q1}, "MPO-boundary");

  auto environment = uni20::tensor_network::make_identity_mpo_environment<double>(bond, auxiliary, 0);
  ASSERT_EQ(environment.stored_block_count(), 2);
  auto first = environment.block(typename decltype(environment)::key_type{{0, 0, 0}});
  EXPECT_DOUBLE_EQ((first[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((first[0, 1]), 0.0);
  EXPECT_DOUBLE_EQ((first[1, 0]), 0.0);
  EXPECT_DOUBLE_EQ((first[1, 1]), 1.0);
  auto second = environment.block(typename decltype(environment)::key_type{{1, 0, 1}});
  EXPECT_DOUBLE_EQ((second[0, 0]), 1.0);

  EXPECT_THROW((static_cast<void>(uni20::tensor_network::make_identity_mpo_environment<double>(bond, auxiliary, 1))),
               std::invalid_argument);
  EXPECT_THROW((static_cast<void>(uni20::tensor_network::make_identity_mpo_environment<double>(bond, auxiliary, 2))),
               std::invalid_argument);
}

TEST(MpoEnvironmentTest, UpdatesU1HeisenbergProductStateFromBothBoundaries)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  auto const qminus1 = uni20::make_qnum(symmetry, {{"N", -1}});
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left-boundary");
  uni20::BlockSpace const middle_bond(symmetry, {{q0, 1}}, "middle-bond");
  uni20::BlockSpace const right_bond(symmetry, {{q1, 1}}, "right-boundary");
  uni20::LocalSpace const physical(symmetry, {q0, q1}, "physical");
  uni20::LocalSpace const left_auxiliary(symmetry, {q0}, "left-MPO-bond");
  uni20::LocalSpace const middle_auxiliary(symmetry, {q0, qminus1, q1}, "middle-MPO-bond");
  uni20::LocalSpace const right_auxiliary(symmetry, {q0}, "right-MPO-bond");

  RealSite first_site(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{middle_bond}, {SiteKey{{0, 0, 0}}});
  RealSite second_site(symmetry, uni20::Domain{middle_bond, physical}, uni20::Codomain{right_bond},
                       {SiteKey{{0, 1, 0}}});
  first_site.block(SiteKey{{0, 0, 0}})[0, 0] = 1.0;
  second_site.block(SiteKey{{0, 1, 0}})[0, 0] = 1.0;

  RealMpo first_mpo(symmetry, uni20::Domain{left_auxiliary, physical}, uni20::Codomain{middle_auxiliary, physical},
                    {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 0, 1, 1}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{0, 1, 2, 0}}});
  first_mpo.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  first_mpo.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  first_mpo.block(MpoKey{{0, 0, 1, 1}})[] = 0.5;
  first_mpo.block(MpoKey{{0, 1, 2, 0}})[] = 0.5;

  RealMpo second_mpo(symmetry, uni20::Domain{middle_auxiliary, physical}, uni20::Codomain{right_auxiliary, physical},
                     {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{1, 1, 0, 0}}, MpoKey{{2, 0, 0, 1}}});
  second_mpo.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  second_mpo.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  second_mpo.block(MpoKey{{1, 1, 0, 0}})[] = 1.0;
  second_mpo.block(MpoKey{{2, 0, 0, 1}})[] = 1.0;

  auto left = uni20::tensor_network::make_identity_mpo_environment<double>(left_bond, left_auxiliary, 0);
  auto left_middle = uni20::tensor_network::extend_left_environment(left, first_site, first_mpo);
  ASSERT_EQ(left_middle.stored_block_count(), 1);
  auto left_final = uni20::tensor_network::extend_left_environment(left_middle, second_site, second_mpo);
  ASSERT_EQ(left_final.stored_block_count(), 1);
  EXPECT_NEAR((left_final.block_by_ordinal(0)[0, 0]), -0.25, 1.0e-14);

  auto right = uni20::tensor_network::make_identity_mpo_environment<double>(right_bond, right_auxiliary, 0);
  auto right_middle = uni20::tensor_network::extend_right_environment(right, second_site, second_mpo);
  ASSERT_EQ(right_middle.stored_block_count(), 1);
  auto right_final = uni20::tensor_network::extend_right_environment(right_middle, first_site, first_mpo);
  ASSERT_EQ(right_final.stored_block_count(), 1);
  EXPECT_NEAR((right_final.block_by_ordinal(0)[0, 0]), -0.25, 1.0e-14);
}

TEST(MpoEnvironmentTest, ConjugatesDistinctComplexBraSite)
{
  using value_type = uni20::complex<double>;
  using Site = uni20::tensor_network::MpsSite<value_type, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
  using Mpo = uni20::tensor_network::MpoSite<value_type, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                             uni20::LocalSpace>;
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left");
  uni20::BlockSpace const right_bond(symmetry, {{q0, 1}}, "right");
  uni20::LocalSpace const physical(symmetry, {q0}, "physical");
  uni20::LocalSpace const auxiliary(symmetry, {q0}, "auxiliary");
  typename Site::key_type const site_key{{0, 0, 0}};
  typename Mpo::key_type const mpo_key{{0, 0, 0, 0}};
  Site bra(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{right_bond}, {site_key});
  Site ket(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{right_bond}, {site_key});
  Mpo mpo(symmetry, uni20::Domain{auxiliary, physical}, uni20::Codomain{auxiliary, physical}, {mpo_key});
  value_type const bra_value{1.0, 2.0};
  value_type const ket_value{3.0, -4.0};
  value_type const coefficient{2.0, 1.0};
  bra.block(site_key)[0, 0] = bra_value;
  ket.block(site_key)[0, 0] = ket_value;
  mpo.block(mpo_key)[] = coefficient;

  auto left = uni20::tensor_network::make_identity_mpo_environment<value_type>(left_bond, auxiliary, 0);
  auto left_result = uni20::tensor_network::extend_left_environment(left, bra, mpo, ket);
  auto right = uni20::tensor_network::make_identity_mpo_environment<value_type>(right_bond, auxiliary, 0);
  auto right_result = uni20::tensor_network::extend_right_environment(right, bra, mpo, ket);
  value_type const expected = uni20::conj(bra_value) * coefficient * ket_value;
  EXPECT_NEAR((left_result.block_by_ordinal(0)[0, 0].real()), expected.real(), 1.0e-13);
  EXPECT_NEAR((left_result.block_by_ordinal(0)[0, 0].imag()), expected.imag(), 1.0e-13);
  EXPECT_NEAR((right_result.block_by_ordinal(0)[0, 0].real()), expected.real(), 1.0e-13);
  EXPECT_NEAR((right_result.block_by_ordinal(0)[0, 0].imag()), expected.imag(), 1.0e-13);
}

TEST(MpoEnvironmentTest, AccumulatesMultiplePhysicalPathsIntoOneOutputBlock)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left");
  uni20::BlockSpace const right_bond(symmetry, {{q0, 1}}, "right");
  uni20::LocalSpace const physical(symmetry, {q0, q0}, "physical");
  uni20::LocalSpace const auxiliary(symmetry, {q0}, "auxiliary");
  SiteKey const first_site_key{{0, 0, 0}};
  SiteKey const second_site_key{{0, 1, 0}};
  MpoKey const first_mpo_key{{0, 0, 0, 0}};
  MpoKey const second_mpo_key{{0, 1, 0, 1}};
  RealSite site(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{right_bond},
                {first_site_key, second_site_key});
  RealMpo mpo(symmetry, uni20::Domain{auxiliary, physical}, uni20::Codomain{auxiliary, physical},
              {first_mpo_key, second_mpo_key});
  site.block(first_site_key)[0, 0] = 2.0;
  site.block(second_site_key)[0, 0] = 3.0;
  mpo.block(first_mpo_key)[] = 5.0;
  mpo.block(second_mpo_key)[] = 7.0;

  auto environment = uni20::tensor_network::make_identity_mpo_environment<double>(left_bond, auxiliary, 0);
  auto result = uni20::tensor_network::extend_left_environment(environment, site, mpo);
  ASSERT_EQ(result.stored_block_count(), 1);
  EXPECT_DOUBLE_EQ((result.block_by_ordinal(0)[0, 0]), 5.0 * 2.0 * 2.0 + 7.0 * 3.0 * 3.0);
}

TEST(MpoEnvironmentTest, PreservesDenseBondGeometryInBothDirections)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left_bond(symmetry, {{q0, 2}}, "left");
  uni20::BlockSpace const right_bond(symmetry, {{q0, 3}}, "right");
  uni20::LocalSpace const physical(symmetry, {q0}, "physical");
  uni20::LocalSpace const auxiliary(symmetry, {q0}, "auxiliary");
  SiteKey const site_key{{0, 0, 0}};
  MpoKey const mpo_key{{0, 0, 0, 0}};
  RealSite bra(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{right_bond}, {site_key});
  RealSite ket(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{right_bond}, {site_key});
  RealMpo mpo(symmetry, uni20::Domain{auxiliary, physical}, uni20::Codomain{auxiliary, physical}, {mpo_key});
  auto bra_block = bra.block(site_key);
  auto ket_block = ket.block(site_key);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t column = 0; column < 3; ++column)
    {
      bra_block[row, column] = static_cast<double>(1 + row * 3 + column);
      ket_block[row, column] = static_cast<double>(7 + row * 3 + column);
    }
  }
  mpo.block(mpo_key)[] = 2.0;

  auto left = uni20::tensor_network::make_identity_mpo_environment<double>(left_bond, auxiliary, 0);
  auto left_result = uni20::tensor_network::extend_left_environment(left, bra, mpo, ket);
  auto left_block = left_result.block_by_ordinal(0);
  for (std::size_t bra_right = 0; bra_right < 3; ++bra_right)
  {
    for (std::size_t ket_right = 0; ket_right < 3; ++ket_right)
    {
      double expected = 0.0;
      for (std::size_t left = 0; left < 2; ++left)
        expected += 2.0 * bra_block[left, bra_right] * ket_block[left, ket_right];
      EXPECT_DOUBLE_EQ((left_block[bra_right, ket_right]), expected);
    }
  }

  auto right = uni20::tensor_network::make_identity_mpo_environment<double>(right_bond, auxiliary, 0);
  auto right_result = uni20::tensor_network::extend_right_environment(right, bra, mpo, ket);
  auto right_block = right_result.block_by_ordinal(0);
  for (std::size_t bra_left = 0; bra_left < 2; ++bra_left)
  {
    for (std::size_t ket_left = 0; ket_left < 2; ++ket_left)
    {
      double expected = 0.0;
      for (std::size_t right_index = 0; right_index < 3; ++right_index)
        expected += 2.0 * bra_block[bra_left, right_index] * ket_block[ket_left, right_index];
      EXPECT_DOUBLE_EQ((right_block[bra_left, ket_left]), expected);
    }
  }
}

} // namespace
