#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/dense_space.hpp>
#include <uni20/symmetry/local_space.hpp>
#include <uni20/symmetry/morphism_boundary.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <concepts>
#include <tuple>
#include <type_traits>

using namespace uni20;

static_assert(Domain<>::size() == 0);
static_assert(Domain<>::empty());
static_assert(Codomain<>::size() == 0);
static_assert(Codomain<>::empty());

TEST(MorphismBoundaryTest, DomainPreservesSpaceTypesAndOrder)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const bond(sym, {{make_qnum(sym, {{"N", 0}}), 3}}, "left");
  LocalSpace const physical(sym,
                            {
                                make_qnum(sym, {{"N", 0}}),
                                make_qnum(sym, {{"N", 1}}),
                            },
                            "physical");

  auto domain = Domain{bond, physical};

  static_assert(std::same_as<decltype(domain), Domain<BlockSpace, LocalSpace>>);
  static_assert(std::same_as<decltype(domain)::space_type<0>, BlockSpace>);
  static_assert(std::same_as<decltype(domain)::space_type<1>, LocalSpace>);
  static_assert(std::same_as<decltype(domain.space<0>()), BlockSpace const&>);
  static_assert(std::same_as<decltype(domain.space<1>()), LocalSpace const&>);

  EXPECT_EQ(domain.size(), 2);
  EXPECT_FALSE(domain.empty());
  EXPECT_EQ(domain.space<0>(), bond);
  EXPECT_EQ(domain.space<1>(), physical);
  EXPECT_EQ(std::get<0>(domain.spaces()), bond);
  EXPECT_EQ(std::get<1>(domain.spaces()), physical);
}

TEST(MorphismBoundaryTest, CodomainHasDistinctTypeAndExactValueEquality)
{
  Symmetry const sym{"N:U(1)"};
  BlockSpace const bond(sym, {{make_qnum(sym, {{"N", 0}}), 3}}, "right");

  auto codomain = Codomain{bond};
  auto same = codomain;

  static_assert(std::same_as<decltype(codomain), Codomain<BlockSpace>>);
  static_assert(!std::same_as<decltype(codomain), Domain<BlockSpace>>);

  EXPECT_EQ(same, codomain);
  same.set_label<0>("new-right");
  EXPECT_NE(same, codomain);
  EXPECT_EQ(same.space<0>().label(), "new-right");
  EXPECT_EQ(codomain.space<0>().label(), "right");
  EXPECT_TRUE(std::ranges::equal(same.space<0>().sectors(), codomain.space<0>().sectors()));
}

TEST(MorphismBoundaryTest, RepeatedSpaceTypesAndUnitBoundariesAreValid)
{
  DenseSpace const row(2, "row");
  DenseSpace const column(3, "column");

  auto domain = Domain{row, column};
  Domain<> const unit_domain;
  Codomain<> const unit_codomain;

  static_assert(std::same_as<decltype(domain), Domain<DenseSpace, DenseSpace>>);

  EXPECT_EQ(domain.space<0>().extent(), 2);
  EXPECT_EQ(domain.space<1>().extent(), 3);
  EXPECT_TRUE(unit_domain.empty());
  EXPECT_TRUE(unit_codomain.empty());
  EXPECT_EQ(unit_domain, Domain<>{});
  EXPECT_EQ(unit_codomain, Codomain<>{});
}
