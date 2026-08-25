#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/symmetry/u1.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace uni20;

int main()
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  BlockSpace const input(symmetry, {{q0, 3}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Key const key{{0, 0}};
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {key});
  matrix.block(key)[0, 0] = 1.0;
  matrix.block(key)[1, 1] = 2.0;

  auto decomposition = block_svd(matrix, linalg::SvdOptions{.right = linalg::SvdVectorExtent::Full});
  auto kept =
      select_svd_states(decomposition.spectrum(), linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 1});
  auto kept_factors = materialize_svd(decomposition, kept, {.bond_label = "kept"});
  auto right_null = decomposition.right_null_space();
  auto null_vectors =
      materialize_right_singular_vectors_adjoint(decomposition, right_null, {.bond_label = "right-null"});

  if (right_null.state_ids().size() != 1)
  {
    throw std::runtime_error("full right block SVD produced the wrong nullity");
  }
  using NullKey = typename decltype(null_vectors)::key_type;
  auto null_vector = null_vectors.block(NullKey{{0, 0}});
  double maximum_residual = 0.0;
  for (uni20::index_type row = 0; row < matrix.block(key).extent(1); ++row)
  {
    double image = 0.0;
    for (uni20::index_type column = 0; column < matrix.block(key).extent(0); ++column)
      image += matrix.block(key)[column, row] * null_vector[column, 0];
    maximum_residual = std::max(maximum_residual, std::abs(image));
  }
  double squared_norm = 0.0;
  for (uni20::index_type column = 0; column < null_vector.extent(0); ++column)
    squared_norm += null_vector[column, 0] * null_vector[column, 0];
  if (maximum_residual > 1.0e-12 || std::abs(squared_norm - 1.0) > 1.0e-12)
  {
    throw std::runtime_error("materialized right null vector is invalid");
  }

  std::cout << "BlockTensor SVD retained and null subspaces\n";
  std::cout << "  paired singular states: " << decomposition.spectrum().size() << '\n';
  std::cout << "  retained states: " << kept.truncation().retained_rank << '\n';
  std::cout << "  retained singular value: " << kept_factors.singular_values.sector_values(0)[0] << '\n';
  std::cout << "  unpaired right-null vectors: " << right_null.state_ids().size() << '\n';
  std::cout << "  null residual: " << maximum_residual << '\n';
  std::cout << "  null-vector squared norm: " << squared_norm << '\n';
}
