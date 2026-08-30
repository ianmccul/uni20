#include <uni20/tensor_network/environment_cache.hpp>

#include <gtest/gtest.h>

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

auto make_site(uni20::Symmetry symmetry, uni20::BlockSpace const& left, uni20::LocalSpace const& physical,
               uni20::BlockSpace const& right, double value) -> Site
{
  typename Site::key_type const key{{0, 0, 0}};
  Site result(symmetry, uni20::Domain{left, physical}, uni20::Codomain{right}, {key});
  auto block = result.block(key);
  using index_type = typename decltype(block)::index_type;
  for (index_type row = 0; row < block.extent(0); ++row)
  {
    for (index_type column = 0; column < block.extent(1); ++column)
      block[row, column] = value;
  }
  return result;
}

auto make_mpo_site(uni20::Symmetry symmetry, uni20::LocalSpace const& left, uni20::LocalSpace const& physical,
                   uni20::LocalSpace const& right, double value) -> MpoSite
{
  typename MpoSite::key_type const key{{0, 0, 0, 0}};
  MpoSite result(symmetry, uni20::Domain{left, physical}, uni20::Codomain{right, physical}, {key});
  result.block(key)[] = value;
  return result;
}

struct ScalarChainSpaces
{
    uni20::Symmetry symmetry{"N:U(1)"};
    uni20::QNum identity = uni20::QNum::identity(symmetry);
    uni20::BlockSpace left{symmetry, {{identity, 1}}, "b0"};
    uni20::BlockSpace first{symmetry, {{identity, 1}}, "b1"};
    uni20::BlockSpace second{symmetry, {{identity, 1}}, "b2"};
    uni20::BlockSpace right{symmetry, {{identity, 1}}, "b3"};
    uni20::LocalSpace physical{symmetry, {identity}, "physical"};
    uni20::LocalSpace left_auxiliary{symmetry, {identity}, "a0"};
    uni20::LocalSpace first_auxiliary{symmetry, {identity}, "a1"};
    uni20::LocalSpace second_auxiliary{symmetry, {identity}, "a2"};
    uni20::LocalSpace right_auxiliary{symmetry, {identity}, "a3"};
};

TEST(FiniteChainTest, ValidatesConnectivityAndReplacesAnInternalBondTogether)
{
  ScalarChainSpaces spaces;
  EXPECT_THROW((Mps(std::vector<Site>{})), std::invalid_argument);
  EXPECT_THROW((Mpo(std::vector<MpoSite>{})), std::invalid_argument);

  uni20::BlockSpace const disconnected(spaces.symmetry, {{spaces.identity, 1}}, "not-b1");
  EXPECT_THROW((Mps(std::vector<Site>{make_site(spaces.symmetry, spaces.left, spaces.physical, spaces.first, 1.0),
                                      make_site(spaces.symmetry, disconnected, spaces.physical, spaces.right, 1.0)})),
               std::invalid_argument);
  uni20::LocalSpace const disconnected_auxiliary(spaces.symmetry, {spaces.identity}, "not-a1");
  EXPECT_THROW(
      (Mpo(std::vector<MpoSite>{
          make_mpo_site(spaces.symmetry, spaces.left_auxiliary, spaces.physical, spaces.first_auxiliary, 1.0),
          make_mpo_site(spaces.symmetry, disconnected_auxiliary, spaces.physical, spaces.right_auxiliary, 1.0)})),
      std::invalid_argument);

  Mps mps(std::vector<Site>{make_site(spaces.symmetry, spaces.left, spaces.physical, spaces.first, 1.0),
                            make_site(spaces.symmetry, spaces.first, spaces.physical, spaces.right, 1.0)});
  uni20::BlockSpace const enlarged(spaces.symmetry, {{spaces.identity, 2}}, "enlarged-bond");
  auto first = make_site(spaces.symmetry, spaces.left, spaces.physical, enlarged, 2.0);
  auto second = make_site(spaces.symmetry, enlarged, spaces.physical, spaces.right, 3.0);
  mps.replace_pair(0, std::move(first), std::move(second));

  EXPECT_EQ(mps.revision(0), 1);
  EXPECT_EQ(mps.revision(1), 1);
  EXPECT_EQ(mps.site(0).codomain().template space<0>(), enlarged);
  EXPECT_EQ(mps.site(1).domain().template space<0>(), enlarged);
  EXPECT_EQ(mps.site(0).block_by_ordinal(0).extent(1), 2);
  EXPECT_EQ(mps.site(1).block_by_ordinal(0).extent(0), 2);

  uni20::BlockSpace const incompatible(spaces.symmetry, {{spaces.identity, 2}}, "different-label");
  EXPECT_THROW((mps.replace_pair(0, make_site(spaces.symmetry, spaces.left, spaces.physical, enlarged, 1.0),
                                 make_site(spaces.symmetry, incompatible, spaces.physical, spaces.right, 1.0))),
               std::invalid_argument);
}

TEST(MpoEnvironmentCacheTest, BuildsAllEntriesAndRebuildsExactDirectionalRanges)
{
  ScalarChainSpaces spaces;
  Mps mps(std::vector<Site>{make_site(spaces.symmetry, spaces.left, spaces.physical, spaces.first, 2.0),
                            make_site(spaces.symmetry, spaces.first, spaces.physical, spaces.second, 3.0),
                            make_site(spaces.symmetry, spaces.second, spaces.physical, spaces.right, 5.0)});
  Mpo mpo(std::vector<MpoSite>{
      make_mpo_site(spaces.symmetry, spaces.left_auxiliary, spaces.physical, spaces.first_auxiliary, 7.0),
      make_mpo_site(spaces.symmetry, spaces.first_auxiliary, spaces.physical, spaces.second_auxiliary, 11.0),
      make_mpo_site(spaces.symmetry, spaces.second_auxiliary, spaces.physical, spaces.right_auxiliary, 13.0)});
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);

  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_FALSE(cache.left_cached(1));
  EXPECT_FALSE(cache.left_cached(2));
  EXPECT_FALSE(cache.left_cached(3));
  EXPECT_FALSE(cache.right_cached(0));
  EXPECT_FALSE(cache.right_cached(1));
  EXPECT_FALSE(cache.right_cached(2));
  EXPECT_TRUE(cache.right_cached(3));

  cache.build_all();
  for (std::size_t bond = 0; bond <= cache.size(); ++bond)
  {
    EXPECT_TRUE(cache.left_cached(bond));
    EXPECT_TRUE(cache.right_cached(bond));
  }
  double const initial = (2.0 * 2.0 * 7.0) * (3.0 * 3.0 * 11.0) * (5.0 * 5.0 * 13.0);
  EXPECT_DOUBLE_EQ((cache.left_environment(3).block_by_ordinal(0)[0, 0]), initial);
  EXPECT_DOUBLE_EQ((cache.right_environment(0).block_by_ordinal(0)[0, 0]), initial);

  mps.replace_site(1, make_site(spaces.symmetry, spaces.first, spaces.physical, spaces.second, 4.0));
  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_TRUE(cache.left_cached(1));
  EXPECT_FALSE(cache.left_cached(2));
  EXPECT_FALSE(cache.left_cached(3));
  EXPECT_FALSE(cache.right_cached(0));
  EXPECT_FALSE(cache.right_cached(1));
  EXPECT_TRUE(cache.right_cached(2));
  EXPECT_TRUE(cache.right_cached(3));

  double const replacement = (2.0 * 2.0 * 7.0) * (4.0 * 4.0 * 11.0) * (5.0 * 5.0 * 13.0);
  EXPECT_DOUBLE_EQ((cache.left_environment(3).block_by_ordinal(0)[0, 0]), replacement);
  EXPECT_DOUBLE_EQ((cache.right_environment(0).block_by_ordinal(0)[0, 0]), replacement);

  mpo.replace_site(
      0, make_mpo_site(spaces.symmetry, spaces.left_auxiliary, spaces.physical, spaces.first_auxiliary, 17.0));
  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_FALSE(cache.left_cached(1));
  EXPECT_FALSE(cache.left_cached(2));
  EXPECT_FALSE(cache.left_cached(3));
  EXPECT_FALSE(cache.right_cached(0));
  EXPECT_TRUE(cache.right_cached(1));
  EXPECT_TRUE(cache.right_cached(2));
  EXPECT_TRUE(cache.right_cached(3));

  double const mpo_replacement = (2.0 * 2.0 * 17.0) * (4.0 * 4.0 * 11.0) * (5.0 * 5.0 * 13.0);
  EXPECT_DOUBLE_EQ((cache.left_environment(3).block_by_ordinal(0)[0, 0]), mpo_replacement);
  EXPECT_DOUBLE_EQ((cache.right_environment(0).block_by_ordinal(0)[0, 0]), mpo_replacement);
}

TEST(MpoEnvironmentCacheTest, InvalidatesTheUnionForAdjacentPairReplacement)
{
  ScalarChainSpaces spaces;
  Mps mps(std::vector<Site>{make_site(spaces.symmetry, spaces.left, spaces.physical, spaces.first, 1.0),
                            make_site(spaces.symmetry, spaces.first, spaces.physical, spaces.second, 1.0),
                            make_site(spaces.symmetry, spaces.second, spaces.physical, spaces.right, 1.0)});
  Mpo mpo(std::vector<MpoSite>{
      make_mpo_site(spaces.symmetry, spaces.left_auxiliary, spaces.physical, spaces.first_auxiliary, 1.0),
      make_mpo_site(spaces.symmetry, spaces.first_auxiliary, spaces.physical, spaces.second_auxiliary, 1.0),
      make_mpo_site(spaces.symmetry, spaces.second_auxiliary, spaces.physical, spaces.right_auxiliary, 1.0)});
  uni20::tensor_network::MpoEnvironmentCache cache(mps, mpo, 0, 0);
  cache.build_all();

  uni20::BlockSpace const enlarged(spaces.symmetry, {{spaces.identity, 2}}, "new-b1");
  mps.replace_pair(0, make_site(spaces.symmetry, spaces.left, spaces.physical, enlarged, 1.0),
                   make_site(spaces.symmetry, enlarged, spaces.physical, spaces.second, 1.0));

  EXPECT_TRUE(cache.left_cached(0));
  EXPECT_FALSE(cache.left_cached(1));
  EXPECT_FALSE(cache.left_cached(2));
  EXPECT_FALSE(cache.left_cached(3));
  EXPECT_FALSE(cache.right_cached(0));
  EXPECT_FALSE(cache.right_cached(1));
  EXPECT_TRUE(cache.right_cached(2));
  EXPECT_TRUE(cache.right_cached(3));

  EXPECT_NO_THROW(static_cast<void>(cache.left_environment(3)));
  EXPECT_NO_THROW(static_cast<void>(cache.right_environment(0)));
}

TEST(MpoEnvironmentCacheTest, RejectsMismatchedLengthAndPhysicalSpaces)
{
  ScalarChainSpaces spaces;
  Mps mps(std::vector<Site>{make_site(spaces.symmetry, spaces.left, spaces.physical, spaces.first, 1.0),
                            make_site(spaces.symmetry, spaces.first, spaces.physical, spaces.right, 1.0)});
  Mpo short_mpo(std::vector<MpoSite>{
      make_mpo_site(spaces.symmetry, spaces.left_auxiliary, spaces.physical, spaces.right_auxiliary, 1.0)});
  EXPECT_THROW((uni20::tensor_network::MpoEnvironmentCache(mps, short_mpo, 0, 0)), std::invalid_argument);

  uni20::LocalSpace const other_physical(spaces.symmetry, {spaces.identity}, "other-physical");
  Mpo other_mpo(std::vector<MpoSite>{
      make_mpo_site(spaces.symmetry, spaces.left_auxiliary, other_physical, spaces.first_auxiliary, 1.0),
      make_mpo_site(spaces.symmetry, spaces.first_auxiliary, other_physical, spaces.right_auxiliary, 1.0)});
  EXPECT_THROW((uni20::tensor_network::MpoEnvironmentCache(mps, other_mpo, 0, 0)), std::invalid_argument);
}

} // namespace
