#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/symmetry/u1.hpp>

#include <uni20/async/debug_scheduler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace
{

using namespace uni20;

static_assert(!std::is_constructible_v<BlockSvdSelection<double>, std::vector<BlockSvdStateId>,
                                       linalg::SvdTruncationInfo<double>>);

class RecordingBatchScheduler : public async::DebugScheduler {
  public:
    std::size_t batch_calls = 0;
    std::vector<std::size_t> batch_sizes;

  private:
    void execute_batch_impl(async::LightweightTaskBatch const& batch) override
    {
      ++batch_calls;
      batch_sizes.push_back(batch.size());
      for (std::size_t index = 0; index < batch.size(); ++index)
        batch(index);
    }
};

auto sector_coordinate(BlockSpace const& space, QNum const& charge) -> std::size_t
{
  for (std::size_t sector = 0; sector < space.size(); ++sector)
  {
    if (space[sector].q == charge) return sector;
  }
  throw std::invalid_argument("charge is not present in the block space");
}

template <class Tensor> auto singular_value_bond(Tensor const& tensor) -> BlockSpace const&
{
  return tensor.domain().template space<0>();
}

template <class Tensor> auto singular_values_in_sector(Tensor& tensor, std::size_t sector)
{
  using key_type = typename std::remove_cvref_t<Tensor>::key_type;
  return tensor.diagonal_values(key_type{{sector, sector}});
}

template <class Storage> class BlockTensorSvdStorageTest : public ::testing::Test {};

using ImmediateSvdStorageTypes = ::testing::Types<SeparateSparseBlockStorage<>, ParallelSeparateSparseBlockStorage<>,
                                                  PackedSparseBlockStorage<>, ParallelPackedSparseBlockStorage<>>;
TYPED_TEST_SUITE(BlockTensorSvdStorageTest, ImmediateSvdStorageTypes);

TYPED_TEST(BlockTensorSvdStorageTest, FactorizesEveryImmediateLocalStoragePolicy)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 1}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, TypeParam>;
  using Key = typename Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 2.0;

  auto decomposition = block_svd(matrix);
  ASSERT_EQ(decomposition.spectrum().size(), 1);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[0].singular_value, 2.0);
}

TEST(BlockTensorSvd, ParallelStorageBatchesChargeSectorsAndRetainsExecutionPolicy)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  auto const q2 = make_qnum(symmetry, {{"N", 2}});
  BlockSpace const input(symmetry, {{q0, 1}, {q1, 2}, {q2, 3}}, "input");
  BlockSpace const output(symmetry, {{q0, 1}, {q1, 2}, {q2, 3}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, ParallelPackedSparseBlockStorage<>>;
  using Key = typename Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}, Key{{1, 1}}, Key{{2, 2}}});
  for (std::size_t sector = 0; sector < input.size(); ++sector)
  {
    auto block = matrix.block(Key{{sector, sector}});
    for (uni20::index_type index = 0; index < block.extent(0); ++index)
      block[index, index] = static_cast<double>(10 * sector + static_cast<std::size_t>(index) + 1);
  }

  RecordingBatchScheduler scheduler;
  async::ScopedScheduler use_scheduler(&scheduler);
  auto decomposition = block_svd(matrix);
  static_assert(std::same_as<typename decltype(decomposition)::block_execution_policy,
                             SchedulerBatchBlockExecution>);

  ASSERT_EQ(scheduler.batch_calls, 1);
  EXPECT_EQ(scheduler.batch_sizes, (std::vector<std::size_t>{3}));
  ASSERT_EQ(decomposition.sectors().size(), 3);
  EXPECT_EQ(decomposition.sectors()[0].charge, q0);
  EXPECT_EQ(decomposition.sectors()[1].charge, q1);
  EXPECT_EQ(decomposition.sectors()[2].charge, q2);
  ASSERT_EQ(decomposition.spectrum().size(), 6);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[0].singular_value, 23.0);
}

TEST(BlockTensorSvd, FactorizesARepartitionedThreeOneViewAsTwoTwo)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const left_bond(symmetry, {{q0, 1}}, "left-bond");
  LocalSpace const left_site(symmetry, {q0, q0}, "left-site");
  LocalSpace const right_site(symmetry, {q0, q0}, "right-site");
  BlockSpace const right_bond(symmetry, {{q0, 1}}, "right-bond");

  using Center =
      BlockTensor<double, Domain<BlockSpace, LocalSpace, LocalSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Center::key_type;
  Center center(symmetry, Domain{left_bond, left_site, right_site}, Codomain{right_bond},
                {Key{{0, 0, 0, 0}}, Key{{0, 1, 1, 0}}});
  center.block(Key{{0, 0, 0, 0}})[0, 0] = 4.0;
  center.block(Key{{0, 1, 1, 0}})[0, 0] = 1.0;

  auto matrix_view = repartition<MorphismSide::Domain, BoundaryEnd::Right>(center);
  static_assert(ImmediateBlockTensorView<decltype(matrix_view)>);
  static_assert(decltype(matrix_view)::domain_type::size() == 2);
  static_assert(decltype(matrix_view)::codomain_type::size() == 2);

  auto decomposition = block_svd(matrix_view);
  ASSERT_EQ(decomposition.spectrum().size(), 2);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[0].singular_value, 4.0);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[1].singular_value, 1.0);
}

TEST(BlockTensorSvd, SelectsAcrossSectorsAndMaterializesComplementIndependently)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const input(symmetry, {{q0, 2}, {q1, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}, {q1, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}, Key{{1, 1}}});
  matrix.block(Key{{0, 0}})[0, 0] = 4.0;
  matrix.block(Key{{0, 0}})[1, 1] = 1.0;
  matrix.block(Key{{1, 1}})[0, 0] = 3.0;

  auto decomposition = block_svd(matrix);
  ASSERT_EQ(decomposition.spectrum().size(), 3);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[0].singular_value, 4.0);
  EXPECT_EQ(decomposition.spectrum()[0].id.sector, q0);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[1].singular_value, 3.0);
  EXPECT_EQ(decomposition.spectrum()[1].id.sector, q1);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[2].singular_value, 1.0);
  EXPECT_EQ(decomposition.spectrum()[2].id.sector, q0);

  auto kept =
      select_svd_states(decomposition.spectrum(), linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 2});
  auto discarded = complement_svd_selection(decomposition.spectrum(), kept);
  auto kept_factors = materialize_svd(decomposition, kept, {.bond_label = "kept"});
  auto discarded_factors = materialize_svd(decomposition, discarded, {.bond_label = "discarded"});

  auto const& kept_bond = singular_value_bond(kept_factors.singular_values);
  EXPECT_EQ(kept_bond.label(), "kept");
  ASSERT_EQ(kept_bond.size(), 2);
  EXPECT_EQ(
      kept_bond[sector_coordinate(kept_bond, q0)].dim, 1);
  EXPECT_EQ(
      kept_bond[sector_coordinate(kept_bond, q1)].dim, 1);
  EXPECT_EQ(kept.truncation().retained_rank, 2);
  EXPECT_NEAR(kept.truncation().discarded_weight, 1.0 / 26.0, 1.0e-14);

  auto const& discarded_bond = singular_value_bond(discarded_factors.singular_values);
  ASSERT_EQ(discarded_bond.size(), 1);
  EXPECT_EQ(discarded_bond[0].q, q0);
  auto discarded_values = singular_values_in_sector(discarded_factors.singular_values, 0);
  ASSERT_EQ(discarded_values.size(), 1);
  EXPECT_DOUBLE_EQ(discarded_values[0], 1.0);
  EXPECT_EQ(discarded.truncation().retained_rank, 1);
  EXPECT_NEAR(discarded.truncation().discarded_weight, 25.0 / 26.0, 1.0e-14);
}

TEST(BlockTensorSvd, FullSelectionReconstructsEveryChargeSector)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const input(symmetry, {{q0, 2}, {q1, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}, {q1, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}, Key{{1, 1}}});
  matrix.block(Key{{0, 0}})[0, 0] = 4.0;
  matrix.block(Key{{0, 0}})[0, 1] = -2.0;
  matrix.block(Key{{0, 0}})[1, 0] = 1.0;
  matrix.block(Key{{0, 0}})[1, 1] = 3.0;
  matrix.block(Key{{1, 1}})[0, 0] = 2.0;
  matrix.block(Key{{1, 1}})[0, 1] = -1.0;

  auto decomposition = block_svd(matrix);
  auto selection = select_svd_states(decomposition.spectrum());
  auto factors = materialize_svd(decomposition, selection, {.bond_label = "exact"});

  for (std::size_t sector = 0; sector < input.size(); ++sector)
  {
    std::size_t const bond = sector_coordinate(singular_value_bond(factors.singular_values), input[sector].q);
    auto original = matrix.block(Key{{sector, sector}});
    using LeftKey = typename decltype(factors.left_singular_vectors)::key_type;
    using RightKey = typename decltype(factors.right_singular_vectors_adjoint)::key_type;
    auto left = factors.left_singular_vectors.block(LeftKey{{bond, sector}});
    auto right = factors.right_singular_vectors_adjoint.block(RightKey{{sector, bond}});
    auto values = singular_values_in_sector(factors.singular_values, bond);
    for (uni20::index_type column = 0; column < original.extent(0); ++column)
    {
      for (uni20::index_type row = 0; row < original.extent(1); ++row)
      {
        double reconstructed = 0.0;
        for (std::size_t state = 0; state < values.size(); ++state)
          reconstructed += left[static_cast<uni20::index_type>(state), row] * values[state] *
                           right[column, static_cast<uni20::index_type>(state)];
        EXPECT_NEAR(reconstructed, (original[column, row]), 1.0e-12);
      }
    }
  }
}

TEST(BlockTensorSvd, AssemblesRepeatedBoundaryFragmentsAndImplicitZeros)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  LocalSpace const local(symmetry, {q0, q0}, "local");
  BlockSpace const input_bond(symmetry, {{q0, 1}}, "input-bond");
  BlockSpace const output_bond(symmetry, {{q0, 1}}, "output-bond");

  using Tensor = BlockTensor<double, Domain<LocalSpace, BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Tensor::key_type;
  Tensor tensor(symmetry, Domain{local, input_bond}, Codomain{output_bond}, {Key{{0, 0, 0}}});
  tensor.block(Key{{0, 0, 0}})[0, 0] = 5.0;

  auto decomposition = block_svd(tensor);
  ASSERT_EQ(decomposition.spectrum().size(), 1);
  EXPECT_DOUBLE_EQ(decomposition.spectrum()[0].singular_value, 5.0);
  ASSERT_EQ(decomposition.sectors().size(), 1);
  ASSERT_EQ(decomposition.sectors()[0].stored_source_keys.size(), 1);
  EXPECT_EQ(decomposition.sectors()[0].stored_source_keys[0], (Key{{0, 0, 0}}));
  auto factors = materialize_svd(decomposition, select_svd_states(decomposition.spectrum()));

  using RightKey = typename decltype(factors.right_singular_vectors_adjoint)::key_type;
  auto stored = factors.right_singular_vectors_adjoint.block(RightKey{{0, 0, 0}});
  auto implicit_zero = factors.right_singular_vectors_adjoint.block(RightKey{{1, 0, 0}});
  EXPECT_NEAR(std::abs(stored[0, 0]), 1.0, 1.0e-12);
  EXPECT_NEAR((implicit_zero[0, 0]), 0.0, 1.0e-12);
}

TEST(BlockTensorSvd, MaterializesKeptAndPairedNullPartitionsFromOneDecomposition)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 2}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 2.0;

  auto decomposition = block_svd(matrix);
  auto kept =
      select_svd_states(decomposition.spectrum(), linalg::SvdTruncationPolicy<double>{.singular_value_cutoff = 1.0});
  auto paired_null = complement_svd_selection(decomposition.spectrum(), kept);
  auto kept_factors = materialize_svd(decomposition, kept, {.bond_label = "kept"});
  auto null_factors = materialize_svd(decomposition, paired_null, {.bond_label = "paired-null"});

  EXPECT_EQ(singular_value_bond(kept_factors.singular_values).label(), "kept");
  EXPECT_EQ(singular_value_bond(null_factors.singular_values).label(), "paired-null");
  ASSERT_EQ(kept_factors.singular_values.stored_block_count(), 1);
  ASSERT_EQ(null_factors.singular_values.stored_block_count(), 1);
  EXPECT_DOUBLE_EQ(singular_values_in_sector(kept_factors.singular_values, 0)[0], 2.0);
  EXPECT_DOUBLE_EQ(singular_values_in_sector(null_factors.singular_values, 0)[0], 0.0);

  using LeftKey = typename decltype(null_factors.left_singular_vectors)::key_type;
  using RightKey = typename decltype(null_factors.right_singular_vectors_adjoint)::key_type;
  auto null_left = null_factors.left_singular_vectors.block(LeftKey{{0, 0}});
  auto null_right = null_factors.right_singular_vectors_adjoint.block(RightKey{{0, 0}});
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    double image = 0.0;
    for (uni20::index_type column = 0; column < 2; ++column)
      image += matrix.block(Key{{0, 0}})[column, row] * null_right[column, 0];
    EXPECT_NEAR(image, 0.0, 1.0e-12);
  }
  EXPECT_NEAR((null_left[0, 0] * null_left[0, 0] + null_left[0, 1] * null_left[0, 1]), 1.0, 1.0e-12);
}

TEST(BlockTensorSvd, ComplexFactorsReconstructTheOriginalBlock)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 2}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}}, "output");

  using Scalar = uni20::complex<double>;
  using Matrix = BlockTensor<Scalar, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = Scalar{2.0, 1.0};
  matrix.block(Key{{0, 0}})[0, 1] = Scalar{-1.0, 0.5};
  matrix.block(Key{{0, 0}})[1, 0] = Scalar{0.25, -2.0};
  matrix.block(Key{{0, 0}})[1, 1] = Scalar{3.0, -0.75};

  auto decomposition = block_svd(matrix);
  auto factors = materialize_svd(decomposition, select_svd_states(decomposition.spectrum()));
  auto right_times_singular =
      contract<1, 0>(factors.right_singular_vectors_adjoint, factors.singular_values);
  auto reconstructed_tensor = contract<1, 0>(right_times_singular, factors.left_singular_vectors);
  auto reconstructed_block = reconstructed_tensor.block(Key{{0, 0}});
  using LeftKey = typename decltype(factors.left_singular_vectors)::key_type;
  using RightKey = typename decltype(factors.right_singular_vectors_adjoint)::key_type;
  auto left = factors.left_singular_vectors.block(LeftKey{{0, 0}});
  auto right = factors.right_singular_vectors_adjoint.block(RightKey{{0, 0}});
  auto values = singular_values_in_sector(factors.singular_values, 0);

  for (uni20::index_type column = 0; column < 2; ++column)
  {
    for (uni20::index_type row = 0; row < 2; ++row)
    {
      Scalar reconstructed{};
      for (std::size_t state = 0; state < values.size(); ++state)
        reconstructed += left[static_cast<uni20::index_type>(state), row] * values[state] *
                         right[column, static_cast<uni20::index_type>(state)];
      EXPECT_NEAR(std::abs(reconstructed - matrix.block(Key{{0, 0}})[column, row]), 0.0, 1.0e-12);
      EXPECT_NEAR(std::abs(reconstructed_block[column, row] - matrix.block(Key{{0, 0}})[column, row]), 0.0,
                  1.0e-12);
    }
  }
}

TEST(BlockTensorSvd, EmptySelectionMaterializesAnEmptyBond)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 1}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 2.0;

  auto decomposition = block_svd(matrix);
  auto selection =
      select_svd_states(decomposition.spectrum(), linalg::SvdTruncationPolicy<double>{.singular_value_cutoff = 3.0});
  auto factors = materialize_svd(decomposition, selection, {.bond_label = "empty"});

  EXPECT_EQ(selection.truncation().retained_rank, 0);
  EXPECT_EQ(singular_value_bond(factors.singular_values).label(), "empty");
  EXPECT_EQ(factors.singular_values.stored_block_count(), 0);
  EXPECT_EQ(factors.left_singular_vectors.stored_block_count(), 0);
  EXPECT_EQ(factors.right_singular_vectors_adjoint.stored_block_count(), 0);
}

TEST(BlockTensorSvd, MaterializesFullRightNullSpaceSeparately)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 3}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 1.0;
  matrix.block(Key{{0, 0}})[1, 1] = 2.0;

  auto decomposition = block_svd(matrix, linalg::SvdOptions{.right = linalg::SvdVectorExtent::Full});
  auto null_selection = decomposition.right_null_space();
  ASSERT_EQ(null_selection.state_ids().size(), 1);
  auto null_vectors =
      materialize_right_singular_vectors_adjoint(decomposition, null_selection, {.bond_label = "right-null"});

  using NullKey = typename decltype(null_vectors)::key_type;
  auto vector = null_vectors.block(NullKey{{0, 0}});
  ASSERT_EQ(vector.extent(0), 3);
  ASSERT_EQ(vector.extent(1), 1);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    double image = 0.0;
    for (uni20::index_type column = 0; column < 3; ++column)
      image += matrix.block(Key{{0, 0}})[column, row] * vector[column, 0];
    EXPECT_NEAR(image, 0.0, 1.0e-12);
  }
  double squared_norm = 0.0;
  for (uni20::index_type column = 0; column < 3; ++column)
    squared_norm += vector[column, 0] * vector[column, 0];
  EXPECT_NEAR(squared_norm, 1.0, 1.0e-12);
}

TEST(BlockTensorSvd, FullRightNullSpaceRetainsUnmatchedDomainCharge)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const input(symmetry, {{q0, 1}, {q1, 2}}, "input");
  BlockSpace const output(symmetry, {{q0, 1}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 2.0;

  auto decomposition = block_svd(matrix, linalg::SvdOptions{.right = linalg::SvdVectorExtent::Full});
  auto null_selection = decomposition.right_null_space();
  ASSERT_EQ(null_selection.state_ids().size(), 2);
  EXPECT_EQ(null_selection.state_ids()[0].sector, q1);
  EXPECT_EQ(null_selection.state_ids()[1].sector, q1);

  auto null_vectors = materialize_right_singular_vectors_adjoint(decomposition, null_selection);
  ASSERT_EQ(null_vectors.codomain().template space<0>().size(), 1);
  EXPECT_EQ(null_vectors.codomain().template space<0>()[0].q, q1);
  EXPECT_EQ(null_vectors.codomain().template space<0>()[0].dim, 2);
  using NullKey = typename decltype(null_vectors)::key_type;
  auto vectors = null_vectors.block(NullKey{{1, 0}});
  ASSERT_EQ(vectors.extent(0), 2);
  ASSERT_EQ(vectors.extent(1), 2);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_DOUBLE_EQ((vectors[row, column]), row == column ? 1.0 : 0.0);
    }
  }
}

TEST(BlockTensorSvd, FullLeftNullSpaceRetainsUnmatchedCodomainCharge)
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const input(symmetry, {{q0, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 1}, {q1, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {Key{{0, 0}}});
  matrix.block(Key{{0, 0}})[0, 0] = 2.0;

  auto decomposition = block_svd(matrix, linalg::SvdOptions{.left = linalg::SvdVectorExtent::Full});
  auto null_selection = decomposition.left_null_space();
  ASSERT_EQ(null_selection.state_ids().size(), 2);
  EXPECT_EQ(null_selection.state_ids()[0].sector, q1);
  EXPECT_EQ(null_selection.state_ids()[1].sector, q1);

  auto null_vectors = materialize_left_singular_vectors(decomposition, null_selection);
  ASSERT_EQ(null_vectors.domain().template space<0>().size(), 1);
  EXPECT_EQ(null_vectors.domain().template space<0>()[0].q, q1);
  EXPECT_EQ(null_vectors.domain().template space<0>()[0].dim, 2);
  using NullKey = typename decltype(null_vectors)::key_type;
  auto vectors = null_vectors.block(NullKey{{0, 1}});
  ASSERT_EQ(vectors.extent(0), 2);
  ASSERT_EQ(vectors.extent(1), 2);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_DOUBLE_EQ((vectors[column, row]), row == column ? 1.0 : 0.0);
    }
  }
}

} // namespace
