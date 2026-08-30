#include <uni20/tensor_network/two_site_dmrg.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using Site = uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using MpoSite = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                               uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using Mps = uni20::tensor_network::FiniteMps<double, uni20::BlockSpace, uni20::LocalSpace>;
using Mpo =
    uni20::tensor_network::FiniteMpo<double, uni20::LocalSpace, uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using Center = uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                    uni20::BlockSpace, uni20::PackedSparseBlockStorage<>>;
using SiteKey = typename Site::key_type;
using MpoKey = typename MpoSite::key_type;
using CenterKey = typename Center::key_type;

struct HeisenbergSpaces
{
    uni20::Symmetry symmetry{"N:U(1)"};
    uni20::QNum q0 = uni20::QNum::identity(symmetry);
    uni20::QNum q1 = uni20::make_qnum(symmetry, {{"N", 1}});
    uni20::QNum qminus1 = uni20::make_qnum(symmetry, {{"N", -1}});
    uni20::BlockSpace left{symmetry, {{q0, 1}}, "left-boundary"};
    uni20::BlockSpace middle{symmetry, {{q0, 1}, {q1, 1}}, "middle-bond"};
    uni20::BlockSpace right{symmetry, {{q1, 1}}, "right-boundary"};
    uni20::LocalSpace physical{symmetry, {q0, q1}, "physical"};
    uni20::LocalSpace left_auxiliary{symmetry, {q0}, "left-mpo"};
    uni20::LocalSpace middle_auxiliary{symmetry, {q0, qminus1, q1}, "middle-mpo"};
    uni20::LocalSpace right_auxiliary{symmetry, {q0}, "right-mpo"};
};

auto make_heisenberg_mps(HeisenbergSpaces const& spaces) -> Mps
{
  Site first(spaces.symmetry, uni20::Domain{spaces.left, spaces.physical}, uni20::Codomain{spaces.middle},
             {SiteKey{{0, 0, 0}}});
  Site second(spaces.symmetry, uni20::Domain{spaces.middle, spaces.physical}, uni20::Codomain{spaces.right},
              {SiteKey{{0, 1, 0}}, SiteKey{{1, 0, 0}}});
  first.block(SiteKey{{0, 0, 0}})[0, 0] = 1.0;
  second.block(SiteKey{{0, 1, 0}})[0, 0] = 1.0;
  second.block(SiteKey{{1, 0, 0}})[0, 0] = 1.0;
  return Mps(std::vector<Site>{std::move(first), std::move(second)});
}

auto make_heisenberg_mpo(HeisenbergSpaces const& spaces) -> Mpo
{
  MpoSite first(spaces.symmetry, uni20::Domain{spaces.left_auxiliary, spaces.physical},
                uni20::Codomain{spaces.middle_auxiliary, spaces.physical},
                {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 0, 1, 1}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{0, 1, 2, 0}}});
  first.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  first.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  first.block(MpoKey{{0, 0, 1, 1}})[] = 0.5;
  first.block(MpoKey{{0, 1, 2, 0}})[] = 0.5;

  MpoSite second(spaces.symmetry, uni20::Domain{spaces.middle_auxiliary, spaces.physical},
                 uni20::Codomain{spaces.right_auxiliary, spaces.physical},
                 {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 1, 0, 1}}, MpoKey{{1, 1, 0, 0}}, MpoKey{{2, 0, 0, 1}}});
  second.block(MpoKey{{0, 0, 0, 0}})[] = 0.5;
  second.block(MpoKey{{0, 1, 0, 1}})[] = -0.5;
  second.block(MpoKey{{1, 1, 0, 0}})[] = 1.0;
  second.block(MpoKey{{2, 0, 0, 1}})[] = 1.0;
  return Mpo(std::vector<MpoSite>{std::move(first), std::move(second)});
}

auto make_scalar_site(uni20::Symmetry symmetry, uni20::BlockSpace const& left, uni20::LocalSpace const& physical,
                      uni20::BlockSpace const& right, double value) -> Site
{
  Site result(symmetry, uni20::Domain{left, physical}, uni20::Codomain{right}, {SiteKey{{0, 0, 0}}});
  result.block_by_ordinal(0)[0, 0] = value;
  return result;
}

auto make_scalar_mpo_site(uni20::Symmetry symmetry, uni20::LocalSpace const& left, uni20::LocalSpace const& physical,
                          uni20::LocalSpace const& right) -> MpoSite
{
  MpoSite result(symmetry, uni20::Domain{left, physical}, uni20::Codomain{right, physical}, {MpoKey{{0, 0, 0, 0}}});
  result.block_by_ordinal(0)[] = 1.0;
  return result;
}

TEST(TwoSiteDmrgSweepTest, FindsTheHeisenbergGroundStateAndRefreshesTheCompletedSide)
{
  HeisenbergSpaces spaces;
  for (auto const direction : {uni20::tensor_network::MpsSweepDirection::left_to_right,
                               uni20::tensor_network::MpsSweepDirection::right_to_left})
  {
    Mps mps = make_heisenberg_mps(spaces);
    Mpo mpo = make_heisenberg_mpo(spaces);
    uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);
    auto sparse_center = uni20::contract_adjacent<1>(mps.site(0), mps.site(1));
    EXPECT_EQ(sparse_center.stored_block_count(), 1);
    EXPECT_EQ(sparse_center.legal_block_count(), 2);

    auto result = uni20::tensor_network::optimize_two_site_dmrg_bond(mps, mpo, cache, 0, direction);
    EXPECT_EQ(result.first_site, 0);
    EXPECT_EQ(result.direction, direction);
    EXPECT_NEAR(result.local_energy, -0.75, 1.0e-12);
    EXPECT_LT(result.residual_bound, 1.0e-12);
    EXPECT_EQ(result.bond.truncation.available_rank, 2);
    EXPECT_EQ(result.bond.truncation.retained_rank, 2);
    EXPECT_EQ(mps.site(0).codomain().template space<0>().label(), "middle-bond");

    auto center = uni20::contract_adjacent<1>(mps.site(0), mps.site(1));
    double const up_down = center.block(CenterKey{{0, 0, 1, 0}})[0, 0];
    double const down_up = center.block(CenterKey{{0, 1, 0, 0}})[0, 0];
    EXPECT_NEAR(up_down + down_up, 0.0, 1.0e-12);
    EXPECT_NEAR(std::abs(up_down), std::sqrt(0.5), 1.0e-12);

    EXPECT_TRUE(cache.left_cached(0));
    EXPECT_TRUE(cache.right_cached(2));
    if (direction == uni20::tensor_network::MpsSweepDirection::left_to_right)
    {
      EXPECT_TRUE(cache.left_cached(1));
      EXPECT_FALSE(cache.right_cached(1));
    }
    else
    {
      EXPECT_FALSE(cache.left_cached(1));
      EXPECT_TRUE(cache.right_cached(1));
    }
  }
}

TEST(TwoSiteDmrgSweepTest, TraversesEveryBondInBothDirectionsAndPreservesLabels)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::LocalSpace const physical(symmetry, {q0}, "physical");
  std::array bonds{uni20::BlockSpace(symmetry, {{q0, 1}}, "b0"), uni20::BlockSpace(symmetry, {{q0, 1}}, "b1"),
                   uni20::BlockSpace(symmetry, {{q0, 1}}, "b2"), uni20::BlockSpace(symmetry, {{q0, 1}}, "b3")};
  std::array auxiliaries{uni20::LocalSpace(symmetry, {q0}, "a0"), uni20::LocalSpace(symmetry, {q0}, "a1"),
                         uni20::LocalSpace(symmetry, {q0}, "a2"), uni20::LocalSpace(symmetry, {q0}, "a3")};
  Mps mps(std::vector<Site>{make_scalar_site(symmetry, bonds[0], physical, bonds[1], 2.0),
                            make_scalar_site(symmetry, bonds[1], physical, bonds[2], 3.0),
                            make_scalar_site(symmetry, bonds[2], physical, bonds[3], 4.0)});
  Mpo mpo(std::vector<MpoSite>{make_scalar_mpo_site(symmetry, auxiliaries[0], physical, auxiliaries[1]),
                               make_scalar_mpo_site(symmetry, auxiliaries[1], physical, auxiliaries[2]),
                               make_scalar_mpo_site(symmetry, auxiliaries[2], physical, auxiliaries[3])});
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);

  auto rightward = uni20::tensor_network::sweep_two_site_dmrg(mps, mpo, cache,
                                                              uni20::tensor_network::MpsSweepDirection::left_to_right);
  ASSERT_EQ(rightward.size(), 2);
  EXPECT_EQ(rightward[0].first_site, 0);
  EXPECT_EQ(rightward[1].first_site, 1);
  EXPECT_EQ(mps.revision(0), 1);
  EXPECT_EQ(mps.revision(1), 2);
  EXPECT_EQ(mps.revision(2), 1);
  EXPECT_EQ(mps.site(0).codomain().template space<0>().label(), "b1");
  EXPECT_EQ(mps.site(1).codomain().template space<0>().label(), "b2");
  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_TRUE(cache.left_cached(1));
  EXPECT_TRUE(cache.left_cached(2));
  EXPECT_TRUE(cache.right_cached(3));

  auto leftward = uni20::tensor_network::sweep_two_site_dmrg(mps, mpo, cache,
                                                             uni20::tensor_network::MpsSweepDirection::right_to_left);
  ASSERT_EQ(leftward.size(), 2);
  EXPECT_EQ(leftward[0].first_site, 1);
  EXPECT_EQ(leftward[1].first_site, 0);
  EXPECT_EQ(mps.revision(0), 2);
  EXPECT_EQ(mps.revision(1), 4);
  EXPECT_EQ(mps.revision(2), 2);
  EXPECT_EQ(mps.site(0).codomain().template space<0>().label(), "b1");
  EXPECT_EQ(mps.site(1).codomain().template space<0>().label(), "b2");
  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_TRUE(cache.right_cached(1));
  EXPECT_TRUE(cache.right_cached(2));
  EXPECT_TRUE(cache.right_cached(3));
}

TEST(TwoSiteDmrgSweepTest, RejectsAChainMismatchedCacheBeforeMutation)
{
  HeisenbergSpaces spaces;
  Mps attached_mps = make_heisenberg_mps(spaces);
  Mps other_mps = make_heisenberg_mps(spaces);
  Mpo mpo = make_heisenberg_mpo(spaces);
  uni20::tensor_network::MpoEnvironmentCache cache(attached_mps, mpo, 0, 0);

  EXPECT_THROW((static_cast<void>(uni20::tensor_network::optimize_two_site_dmrg_bond(
                   other_mps, mpo, cache, 0, uni20::tensor_network::MpsSweepDirection::left_to_right))),
               std::invalid_argument);
  EXPECT_EQ(other_mps.revision(0), 0);
  EXPECT_EQ(other_mps.revision(1), 0);
}

} // namespace
