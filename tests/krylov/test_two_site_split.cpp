#include <uni20/core/math.hpp>
#include <uni20/core/types.hpp>
#include <uni20/tensor_network/environment_cache.hpp>
#include <uni20/tensor_network/two_site_split.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace
{

using Site = uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
using Center = uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                    uni20::BlockSpace>;
using MpoSite = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                               uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using Mps = uni20::tensor_network::FiniteMps<double, uni20::BlockSpace, uni20::LocalSpace>;
using Mpo =
    uni20::tensor_network::FiniteMpo<double, uni20::LocalSpace, uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
using CenterKey = typename Center::key_type;
using SiteKey = typename Site::key_type;
using MpoKey = typename MpoSite::key_type;
using Decomposition = decltype(uni20::tensor_network::decompose_two_site_center(std::declval<Center const&>()));

template <class Storage>
concept CanMaterializeTwoSiteMpsSplit =
    requires(Decomposition const& decomposition, uni20::BlockSvdSelection<double> const& selection) {
      uni20::tensor_network::materialize_two_site_mps_split<Storage>(
          decomposition, selection, uni20::tensor_network::MpsSweepDirection::left_to_right);
    };

static_assert(CanMaterializeTwoSiteMpsSplit<uni20::SeparateSparseBlockStorage<>>);
static_assert(CanMaterializeTwoSiteMpsSplit<uni20::ParallelPackedSparseBlockStorage<>>);
static_assert(!CanMaterializeTwoSiteMpsSplit<uni20::PackedCompleteBlockStorage<>>);
static_assert(!CanMaterializeTwoSiteMpsSplit<uni20::AsyncSeparateSparseBlockStorage<>>);
static_assert(!CanMaterializeTwoSiteMpsSplit<uni20::PackedDiagonalBlockStorage<>>);

struct SplitSpaces
{
    uni20::Symmetry symmetry{"N:U(1)"};
    uni20::QNum q0 = uni20::QNum::identity(symmetry);
    uni20::QNum q1 = uni20::make_qnum(symmetry, {{"N", 1}});
    uni20::BlockSpace left{symmetry, {{q0, 1}}, "left-boundary"};
    uni20::BlockSpace initial_middle{symmetry, {{q0, 1}}, "initial-middle"};
    uni20::BlockSpace right{symmetry, {{q1, 1}}, "right-boundary"};
    uni20::LocalSpace left_physical{symmetry, {q0, q1}, "left-physical"};
    uni20::LocalSpace right_physical{symmetry, {q0, q1}, "right-physical"};
    uni20::LocalSpace left_auxiliary{symmetry, {q0}, "left-auxiliary"};
    uni20::LocalSpace middle_auxiliary{symmetry, {q0}, "middle-auxiliary"};
    uni20::LocalSpace right_auxiliary{symmetry, {q0}, "right-auxiliary"};
};

auto make_center(SplitSpaces const& spaces) -> Center
{
  Center result(spaces.symmetry, uni20::Domain{spaces.left, spaces.left_physical, spaces.right_physical},
                uni20::Codomain{spaces.right}, {CenterKey{{0, 0, 1, 0}}, CenterKey{{0, 1, 0, 0}}});
  result.block(CenterKey{{0, 0, 1, 0}})[0, 0] = std::sqrt(0.8);
  result.block(CenterKey{{0, 1, 0, 0}})[0, 0] = -std::sqrt(0.2);
  return result;
}

auto make_initial_mps(SplitSpaces const& spaces) -> Mps
{
  Site first(spaces.symmetry, uni20::Domain{spaces.left, spaces.left_physical}, uni20::Codomain{spaces.initial_middle},
             {SiteKey{{0, 0, 0}}});
  Site second(spaces.symmetry, uni20::Domain{spaces.initial_middle, spaces.right_physical},
              uni20::Codomain{spaces.right}, {SiteKey{{0, 1, 0}}});
  first.block_by_ordinal(0)[0, 0] = 1.0;
  second.block_by_ordinal(0)[0, 0] = 1.0;
  return Mps(std::vector<Site>{std::move(first), std::move(second)});
}

auto make_identity_mpo(SplitSpaces const& spaces) -> Mpo
{
  MpoSite first(spaces.symmetry, uni20::Domain{spaces.left_auxiliary, spaces.left_physical},
                uni20::Codomain{spaces.middle_auxiliary, spaces.left_physical},
                {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 1, 0, 1}}});
  MpoSite second(spaces.symmetry, uni20::Domain{spaces.middle_auxiliary, spaces.right_physical},
                 uni20::Codomain{spaces.right_auxiliary, spaces.right_physical},
                 {MpoKey{{0, 0, 0, 0}}, MpoKey{{0, 1, 0, 1}}});
  first.block(MpoKey{{0, 0, 0, 0}})[] = 1.0;
  first.block(MpoKey{{0, 1, 0, 1}})[] = 1.0;
  second.block(MpoKey{{0, 0, 0, 0}})[] = 1.0;
  second.block(MpoKey{{0, 1, 0, 1}})[] = 1.0;
  return Mpo(std::vector<MpoSite>{std::move(first), std::move(second)});
}

template <uni20::BlockTensorView Actual>
void expect_reconstructs(Actual const& actual, Center const& expected, double tolerance)
{
  EXPECT_EQ(actual.domain(), expected.domain());
  EXPECT_EQ(actual.codomain(), expected.codomain());
  EXPECT_EQ(actual.stored_block_count(), expected.stored_block_count());
  for (CenterKey const& key : expected.stored_keys())
  {
    auto actual_block = actual.find_block(key);
    ASSERT_TRUE(actual_block.has_value());
    EXPECT_NEAR(((*actual_block)[0, 0]), (expected.block(key)[0, 0]), tolerance);
  }
}

TEST(TwoSiteMpsSplitTest, AbsorbsSingularValuesAccordingToSweepDirection)
{
  SplitSpaces spaces;
  Center center = make_center(spaces);
  auto decomposition = uni20::tensor_network::decompose_two_site_center(center);
  auto selection = uni20::select_svd_states(decomposition.spectrum());

  auto left_to_right = uni20::tensor_network::materialize_two_site_mps_split(
      decomposition, selection, uni20::tensor_network::MpsSweepDirection::left_to_right,
      {.bond_label = "left-to-right-bond"});
  static_assert(std::same_as<decltype(left_to_right.first_site), Site>);
  static_assert(std::same_as<decltype(left_to_right.second_site), Site>);
  auto left_to_right_center = uni20::contract_adjacent<1>(left_to_right.first_site, left_to_right.second_site);
  expect_reconstructs(left_to_right_center, center, 1.0e-13);

  EXPECT_NEAR(std::abs(left_to_right.first_site.block(SiteKey{{0, 0, 0}})[0, 0]), 1.0, 1.0e-13);
  EXPECT_NEAR(std::abs(left_to_right.first_site.block(SiteKey{{0, 1, 1}})[0, 0]), 1.0, 1.0e-13);
  EXPECT_NEAR(std::abs(left_to_right.second_site.block(SiteKey{{0, 1, 0}})[0, 0]), std::sqrt(0.8), 1.0e-13);
  EXPECT_NEAR(std::abs(left_to_right.second_site.block(SiteKey{{1, 0, 0}})[0, 0]), std::sqrt(0.2), 1.0e-13);

  auto right_to_left = uni20::tensor_network::materialize_two_site_mps_split(
      decomposition, selection, uni20::tensor_network::MpsSweepDirection::right_to_left,
      {.bond_label = "right-to-left-bond"});
  auto right_to_left_center = uni20::contract_adjacent<1>(right_to_left.first_site, right_to_left.second_site);
  expect_reconstructs(right_to_left_center, center, 1.0e-13);

  EXPECT_NEAR(std::abs(right_to_left.first_site.block(SiteKey{{0, 0, 0}})[0, 0]), std::sqrt(0.8), 1.0e-13);
  EXPECT_NEAR(std::abs(right_to_left.first_site.block(SiteKey{{0, 1, 1}})[0, 0]), std::sqrt(0.2), 1.0e-13);
  EXPECT_NEAR(std::abs(right_to_left.second_site.block(SiteKey{{0, 1, 0}})[0, 0]), 1.0, 1.0e-13);
  EXPECT_NEAR(std::abs(right_to_left.second_site.block(SiteKey{{1, 0, 0}})[0, 0]), 1.0, 1.0e-13);

  EXPECT_EQ(left_to_right.truncation.retained_rank, 2);
  EXPECT_EQ(right_to_left.truncation.retained_rank, 2);
  EXPECT_EQ(left_to_right.direction, uni20::tensor_network::MpsSweepDirection::left_to_right);
  EXPECT_EQ(right_to_left.direction, uni20::tensor_network::MpsSweepDirection::right_to_left);
}

TEST(TwoSiteMpsSplitTest, InstallsTruncatedPairAndInvalidatesOnlyDependentCacheEntries)
{
  SplitSpaces spaces;
  Center center = make_center(spaces);
  auto decomposition = uni20::tensor_network::decompose_two_site_center(center);
  uni20::linalg::SvdTruncationPolicy<double> policy;
  policy.minimum_retained_extent = 1;
  policy.maximum_retained_extent = 1;
  auto selection = uni20::select_svd_states(decomposition.spectrum(), policy);

  Mps mps = make_initial_mps(spaces);
  Mpo mpo = make_identity_mpo(spaces);
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);
  cache.build_all();
  auto installed = uni20::tensor_network::replace_two_site_from_svd(
      mps, 0, decomposition, selection, uni20::tensor_network::MpsSweepDirection::left_to_right,
      {.bond_label = "truncated-bond"});

  EXPECT_EQ(mps.revision(0), 1);
  EXPECT_EQ(mps.revision(1), 1);
  EXPECT_EQ(mps.site(0).codomain().template space<0>(), mps.site(1).domain().template space<0>());
  EXPECT_EQ(mps.site(0).codomain().template space<0>().label(), "truncated-bond");
  EXPECT_EQ(mps.site(0).codomain().template space<0>().total_dim(), 1);
  EXPECT_EQ(installed.singular_values.domain().template space<0>(), mps.site(0).codomain().template space<0>());
  EXPECT_EQ(installed.singular_values.codomain().template space<0>(), mps.site(1).domain().template space<0>());
  EXPECT_EQ(installed.truncation.available_rank, 2);
  EXPECT_EQ(installed.truncation.retained_rank, 1);
  EXPECT_NEAR(installed.truncation.discarded_weight, 0.2, 1.0e-13);
  EXPECT_EQ(installed.direction, uni20::tensor_network::MpsSweepDirection::left_to_right);

  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_FALSE(cache.left_cached(1));
  EXPECT_FALSE(cache.left_cached(2));
  EXPECT_FALSE(cache.right_cached(0));
  EXPECT_FALSE(cache.right_cached(1));
  EXPECT_TRUE(cache.right_cached(2));
  EXPECT_NO_THROW(static_cast<void>(cache.left_environment(2)));
  EXPECT_NO_THROW(static_cast<void>(cache.right_environment(0)));

  auto truncated_center = uni20::contract_adjacent<1>(mps.site(0), mps.site(1));
  ASSERT_EQ(truncated_center.stored_block_count(), 1);
  EXPECT_NEAR((truncated_center.block(CenterKey{{0, 0, 1, 0}})[0, 0]), std::sqrt(0.8), 1.0e-13);
  EXPECT_FALSE(truncated_center.find_block(CenterKey{{0, 1, 0, 0}}).has_value());
}

TEST(TwoSiteMpsSplitTest, RejectsAnEmptyMpsSelection)
{
  SplitSpaces spaces;
  Center center = make_center(spaces);
  auto decomposition = uni20::tensor_network::decompose_two_site_center(center);
  auto selection = uni20::make_svd_selection(decomposition.spectrum(), std::span<uni20::BlockSvdStateId const>{});
  EXPECT_THROW((static_cast<void>(uni20::tensor_network::materialize_two_site_mps_split(
                   decomposition, selection, uni20::tensor_network::MpsSweepDirection::left_to_right))),
               std::invalid_argument);
}

TEST(TwoSiteMpsSplitTest, RejectsAnInvalidSweepDirection)
{
  SplitSpaces spaces;
  Center center = make_center(spaces);
  auto decomposition = uni20::tensor_network::decompose_two_site_center(center);
  auto selection = uni20::select_svd_states(decomposition.spectrum());
  EXPECT_THROW((static_cast<void>(uni20::tensor_network::materialize_two_site_mps_split(
                   decomposition, selection, static_cast<uni20::tensor_network::MpsSweepDirection>(-1)))),
               std::invalid_argument);
}

TEST(TwoSiteMpsSplitTest, AbsorbsRealSingularValuesIntoComplexSites)
{
  using value_type = uni20::complex<double>;
  using ComplexCenter = uni20::tensor_network::TwoSiteCenter<value_type, uni20::BlockSpace, uni20::LocalSpace,
                                                             uni20::LocalSpace, uni20::BlockSpace>;
  using ComplexSite =
      uni20::tensor_network::MpsSite<value_type, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left(symmetry, {{q0, 1}}, "left");
  uni20::BlockSpace const right(symmetry, {{q0, 1}}, "right");
  uni20::LocalSpace const first_physical(symmetry, {q0}, "first-physical");
  uni20::LocalSpace const second_physical(symmetry, {q0}, "second-physical");
  typename ComplexCenter::key_type const key{{0, 0, 0, 0}};
  ComplexCenter center(symmetry, uni20::Domain{left, first_physical, second_physical}, uni20::Codomain{right}, {key});
  value_type const expected{3.0, -4.0};
  center.block(key)[0, 0] = expected;

  auto decomposition = uni20::tensor_network::decompose_two_site_center(center);
  auto selection = uni20::select_svd_states(decomposition.spectrum());
  auto split = uni20::tensor_network::materialize_two_site_mps_split(
      decomposition, selection, uni20::tensor_network::MpsSweepDirection::right_to_left,
      {.bond_label = "complex-bond"});
  static_assert(std::same_as<decltype(split.first_site), ComplexSite>);
  static_assert(std::same_as<decltype(split.second_site), ComplexSite>);
  auto reconstructed = uni20::contract_adjacent<1>(split.first_site, split.second_site);

  EXPECT_NEAR((reconstructed.block(key)[0, 0].real()), expected.real(), 1.0e-13);
  EXPECT_NEAR((reconstructed.block(key)[0, 0].imag()), expected.imag(), 1.0e-13);
  EXPECT_NEAR((split.singular_values.diagonal_values_by_ordinal(0)[0]), 5.0, 1.0e-13);
}

} // namespace
