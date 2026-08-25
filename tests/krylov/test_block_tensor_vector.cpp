#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/block_tensor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace
{

using Storage = uni20::SeparateSparseBlockStorage<>;
using Tensor =
    uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>, Storage>;
using Key = typename Tensor::key_type;

class U1BlockDiagonalOps : public uni20::krylov::BlockTensorVectorOps<Tensor> {
  public:
    using BlockTensorVectorOps::BlockTensorVectorOps;

    void matvec(Tensor& output, Tensor const& input)
    {
      static_cast<void>(this->vector_dimension(output));
      static_cast<void>(this->vector_dimension(input));

      constexpr std::array<double, 4> sector_zero_diagonal{1.0, 3.0, 4.0, 5.0};
      for (std::size_t ordinal = 0; ordinal < input.stored_block_count(); ++ordinal)
      {
        auto output_block = output.block_by_ordinal(ordinal);
        auto input_block = input.block_by_ordinal(ordinal);
        for (uni20::index_type column = 0; column < input_block.extent(1); ++column)
        {
          for (uni20::index_type row = 0; row < input_block.extent(0); ++row)
          {
            double const diagonal =
                ordinal == 0 ? sector_zero_diagonal[static_cast<std::size_t>(row + input_block.extent(0) * column)]
                             : 2.0;
            output_block[row, column] = diagonal * input_block[row, column];
          }
        }
      }
    }
};

static_assert(uni20::krylov::KrylovVectorOps<uni20::krylov::BlockTensorVectorOps<Tensor>, Tensor, double>);
static_assert(!uni20::krylov::KrylovOperator<uni20::krylov::BlockTensorVectorOps<Tensor>, Tensor>);
static_assert(uni20::krylov::KrylovMatrixFreeOperator<U1BlockDiagonalOps, Tensor, double>);

auto make_prototype(uni20::Symmetry const& symmetry, uni20::BlockSpace const& space) -> Tensor
{
  return Tensor(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}, Key{{1, 1}}});
}

void fill(Tensor& tensor, double value)
{
  for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
  {
    auto block = tensor.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value;
  }
}

TEST(BlockTensorVectorOpsTest, FreezesStructureAndAllocatesWithoutDenseProjection)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  Tensor prototype = make_prototype(symmetry, space);
  fill(prototype, 2.0);

  uni20::krylov::BlockTensorVectorOps<Tensor> ops(prototype);
  EXPECT_EQ(ops.problem_dimension(), 5);
  EXPECT_EQ(ops.vector_dimension(prototype), 5);

  Tensor allocated = ops.allocate_like(prototype);
  EXPECT_EQ(allocated.symmetry(), prototype.symmetry());
  EXPECT_EQ(allocated.domain(), prototype.domain());
  EXPECT_EQ(allocated.codomain(), prototype.codomain());
  EXPECT_TRUE(std::ranges::equal(allocated.stored_keys(), prototype.stored_keys()));
  EXPECT_DOUBLE_EQ(ops.norm(allocated), 0.0);

  auto relabelled_space = space;
  relabelled_space.set_label("other-state");
  Tensor relabelled = make_prototype(symmetry, relabelled_space);
  EXPECT_THROW(static_cast<void>(ops.vector_dimension(relabelled)), std::invalid_argument);

  Tensor different_pattern(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}});
  EXPECT_THROW(static_cast<void>(ops.vector_dimension(different_pattern)), std::invalid_argument);
}

TEST(BlockTensorVectorOpsTest, RunsU1BlockTensorThroughSymmetricLanczos)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 1}}, "state");
  Tensor initial = make_prototype(symmetry, space);
  fill(initial, 1.0);
  U1BlockDiagonalOps ops(initial);

  uni20::krylov::SymmetricEigenParams<double> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 5;
  params.spectrum = uni20::krylov::SpectrumPart::SmallestAlgebraic;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::symmetric_lanczos_standard<double>(ops, initial, params);

  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.size(), 2);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1.0e-12);
  EXPECT_NEAR(result.eigenvalues[1], 2.0, 1.0e-12);
  EXPECT_EQ(result.converged_count, 2);
  for (std::size_t index = 0; index < result.eigenvectors.size(); ++index)
  {
    Tensor residual = ops.allocate_like(result.eigenvectors[index]);
    ops.matvec(residual, result.eigenvectors[index]);
    ops.axpy(residual, -result.eigenvalues[index], result.eigenvectors[index]);
    EXPECT_LT(ops.norm(residual), 1.0e-11);
    EXPECT_NEAR(ops.norm(result.eigenvectors[index]), 1.0, 1.0e-12);
    EXPECT_TRUE(std::ranges::equal(result.eigenvectors[index].stored_keys(), initial.stored_keys()));
  }
}

} // namespace
