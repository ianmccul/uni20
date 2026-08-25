#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/symmetry/u1.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

namespace
{

using namespace uni20;

auto sector_coordinate(BlockSpace const& space, QNum const& charge) -> std::size_t
{
  for (std::size_t sector = 0; sector < space.size(); ++sector)
  {
    if (space[sector].q == charge) return sector;
  }
  throw std::invalid_argument("charge is not present in the block space");
}

template <class Storage> class BlockTensorSvdStorageTest : public ::testing::Test {};

using ImmediateSvdStorageTypes =
    ::testing::Types<SeparateSparseBlockStorage<>, ParallelSeparateSparseBlockStorage<>, PackedSparseBlockStorage<>>;
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

  EXPECT_EQ(kept_factors.singular_values.bond_space().label(), "kept");
  ASSERT_EQ(kept_factors.singular_values.bond_space().size(), 2);
  EXPECT_EQ(
      kept_factors.singular_values.bond_space()[sector_coordinate(kept_factors.singular_values.bond_space(), q0)].dim,
      1);
  EXPECT_EQ(
      kept_factors.singular_values.bond_space()[sector_coordinate(kept_factors.singular_values.bond_space(), q1)].dim,
      1);
  EXPECT_EQ(kept.truncation().retained_rank, 2);
  EXPECT_NEAR(kept.truncation().discarded_weight, 1.0 / 26.0, 1.0e-14);

  ASSERT_EQ(discarded_factors.singular_values.bond_space().size(), 1);
  EXPECT_EQ(discarded_factors.singular_values.bond_space()[0].q, q0);
  ASSERT_EQ(discarded_factors.singular_values.sector_values(0).extent(0), 1);
  EXPECT_DOUBLE_EQ(discarded_factors.singular_values.sector_values(0)[0], 1.0);
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
    std::size_t const bond = sector_coordinate(factors.singular_values.bond_space(), input[sector].q);
    auto original = matrix.block(Key{{sector, sector}});
    using LeftKey = typename decltype(factors.left_singular_vectors)::key_type;
    using RightKey = typename decltype(factors.right_singular_vectors_adjoint)::key_type;
    auto left = factors.left_singular_vectors.block(LeftKey{{bond, sector}});
    auto right = factors.right_singular_vectors_adjoint.block(RightKey{{sector, bond}});
    auto const& values = factors.singular_values.sector_values(bond);
    for (uni20::index_type column = 0; column < original.extent(0); ++column)
    {
      for (uni20::index_type row = 0; row < original.extent(1); ++row)
      {
        double reconstructed = 0.0;
        for (uni20::index_type state = 0; state < values.extent(0); ++state)
          reconstructed += left[state, row] * values[state] * right[column, state];
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

  EXPECT_EQ(kept_factors.singular_values.bond_space().label(), "kept");
  EXPECT_EQ(null_factors.singular_values.bond_space().label(), "paired-null");
  ASSERT_EQ(kept_factors.singular_values.sector_count(), 1);
  ASSERT_EQ(null_factors.singular_values.sector_count(), 1);
  EXPECT_DOUBLE_EQ(kept_factors.singular_values.sector_values(0)[0], 2.0);
  EXPECT_DOUBLE_EQ(null_factors.singular_values.sector_values(0)[0], 0.0);

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
  using LeftKey = typename decltype(factors.left_singular_vectors)::key_type;
  using RightKey = typename decltype(factors.right_singular_vectors_adjoint)::key_type;
  auto left = factors.left_singular_vectors.block(LeftKey{{0, 0}});
  auto right = factors.right_singular_vectors_adjoint.block(RightKey{{0, 0}});
  auto const& values = factors.singular_values.sector_values(0);

  for (uni20::index_type column = 0; column < 2; ++column)
  {
    for (uni20::index_type row = 0; row < 2; ++row)
    {
      Scalar reconstructed{};
      for (uni20::index_type state = 0; state < values.extent(0); ++state)
        reconstructed += left[state, row] * values[state] * right[column, state];
      EXPECT_NEAR(std::abs(reconstructed - matrix.block(Key{{0, 0}})[column, row]), 0.0, 1.0e-12);
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
  EXPECT_EQ(factors.singular_values.bond_space().label(), "empty");
  EXPECT_EQ(factors.singular_values.sector_count(), 0);
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
