#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

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

TEST(TensorContractionEffectiveHamiltonianPlanTest, AppliesSingleTermPlan)
{
  auto r = make_family({{2, 4}});
  auto a = make_family({{2, 3}});
  auto b = make_family({{3, 5}});
  auto c = make_family({{5, 4}});

  a.assign(0, std::array{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  b.assign(0, std::array{1.0, 0.5, -1.0, 2.0, 1.5, 0.0, -0.5, 3.0, 1.0, 2.5, 2.0, 1.0, 0.0, -1.5, 0.25});
  c.assign(0, std::array{1.0,  2.0,  0.5, -1.0, 0.0,  1.5, 2.5, 3.0, -2.0, 1.0,
                         0.75, 0.25, 4.0, -0.5, 1.25, 2.0, 3.5, 0.0, -1.0, 1.0});

  auto const a_values = a.values(0);
  auto const b_values = b.values(0);
  auto const c_values = c.values(0);
  std::array terms{utc::EffectiveHamiltonianPlan::Term{0, 0, 0, 0, 1.25}};

  utc::EffectiveHamiltonianPlan plan(std::move(r), std::move(a), std::move(b), std::move(c), terms);
  plan.apply();

  std::vector<double> bc(3 * 4, 0.0);
  for (std::size_t row = 0; row < 3; ++row)
  {
    for (std::size_t col = 0; col < 4; ++col)
    {
      for (std::size_t inner = 0; inner < 5; ++inner)
      {
        bc[row * 4 + col] += b_values[row * 5 + inner] * c_values[inner * 4 + col];
      }
    }
  }

  std::vector<double> expected(2 * 4, 0.0);
  for (std::size_t row = 0; row < 2; ++row)
  {
    for (std::size_t col = 0; col < 4; ++col)
    {
      for (std::size_t inner = 0; inner < 3; ++inner)
      {
        expected[row * 4 + col] += 1.25 * a_values[row * 3 + inner] * bc[inner * 4 + col];
      }
    }
  }

  auto result = plan.r_values(0);
  ASSERT_EQ(result.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
  {
    EXPECT_NEAR(result[i], expected[i], 1.0e-10);
  }
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
