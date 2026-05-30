#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <span>
#include <stdexcept>

namespace utc = uni20::tensorcontraction;

namespace
{

utc::MatrixFamily make_family(std::initializer_list<utc::MatrixFamily::Block> blocks)
{
  return utc::MatrixFamily(std::span{blocks.begin(), blocks.size()});
}

} // namespace

TEST(TensorContractionEffectiveHamiltonianPlanTest, CompilesSyntheticSingleTermPlan)
{
  auto r = make_family({{5, 4}});
  auto a = make_family({{5, 2}});
  auto b = make_family({{2, 3}});
  auto c = make_family({{3, 4}});
  std::array terms{utc::EffectiveHamiltonianPlan::Term{0, 0, 0, 0, 1.5}};

  utc::EffectiveHamiltonianPlan plan(std::move(r), std::move(a), std::move(b), std::move(c), terms);
  EXPECT_EQ(plan.term_count(), 1);
  EXPECT_FALSE(plan.compiled());

  plan.compile();
  EXPECT_TRUE(plan.compiled());

  plan.compile();
  EXPECT_TRUE(plan.compiled());
}

TEST(TensorContractionEffectiveHamiltonianPlanTest, RejectsInvalidTermShape)
{
  auto r = make_family({{5, 4}});
  auto a = make_family({{5, 2}});
  auto b = make_family({{3, 3}});
  auto c = make_family({{3, 4}});
  std::array terms{utc::EffectiveHamiltonianPlan::Term{0, 0, 0, 0, 1.0}};

  EXPECT_THROW(utc::EffectiveHamiltonianPlan(std::move(r), std::move(a), std::move(b), std::move(c), terms),
               std::invalid_argument);
}

TEST(TensorContractionEffectiveHamiltonianPlanTest, RejectsMissingTermBlock)
{
  auto r = make_family({{5, 4}});
  auto a = make_family({{5, 2}});
  auto b = make_family({{2, 3}});
  auto c = make_family({{3, 4}});
  std::array terms{utc::EffectiveHamiltonianPlan::Term{1, 0, 0, 0, 1.0}};

  EXPECT_THROW(utc::EffectiveHamiltonianPlan(std::move(r), std::move(a), std::move(b), std::move(c), terms),
               std::out_of_range);
}
