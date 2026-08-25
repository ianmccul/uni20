#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/symmetry/u1.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace uni20;

namespace
{

auto sector_coordinate(BlockSpace const& space, QNum const& charge) -> std::size_t
{
  for (std::size_t sector = 0; sector < space.size(); ++sector)
  {
    if (space[sector].q == charge) return sector;
  }
  throw std::invalid_argument("charge is not present in the block space");
}

void require_close(double actual, double expected)
{
  if (std::abs(actual - expected) > 1.0e-12)
  {
    throw std::runtime_error("block-SVD truncation example produced the wrong value");
  }
}

} // namespace

int main()
{
  Symmetry const symmetry{"N:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q1 = make_qnum(symmetry, {{"N", 1}});
  BlockSpace const input(symmetry, {{q0, 2}, {q1, 1}}, "input");
  BlockSpace const output(symmetry, {{q0, 2}, {q1, 2}}, "output");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  using Key = Matrix::key_type;
  Key const q0_key{{0, 0}};
  Key const q1_key{{1, 1}};
  Matrix matrix(symmetry, Domain{input}, Codomain{output}, {q0_key, q1_key});
  matrix.block(q0_key)[0, 0] = 4.0;
  matrix.block(q0_key)[1, 1] = 1.0;
  matrix.block(q1_key)[0, 0] = 3.0;

  auto decomposition = block_svd(matrix);
  auto kept =
      select_svd_states(decomposition.spectrum(), linalg::SvdTruncationPolicy<double>{.maximum_retained_extent = 2});
  auto discarded = complement_svd_selection(decomposition.spectrum(), kept);
  auto kept_factors = materialize_svd(decomposition, kept, {.bond_label = "kept"});
  auto discarded_factors = materialize_svd(decomposition, discarded, {.bond_label = "discarded"});

  if (kept.truncation().retained_rank != 2 || discarded.truncation().retained_rank != 1)
  {
    throw std::runtime_error("block-SVD selection produced the wrong ranks");
  }
  require_close(kept.truncation().discarded_weight, 1.0 / 26.0);
  auto const discarded_q0 = sector_coordinate(discarded_factors.singular_values.bond_space(), q0);
  require_close(discarded_factors.singular_values.sector_values(discarded_q0)[0], 1.0);

  std::cout << "sector-global BlockTensor SVD\n";
  std::cout << "  spectrum:\n";
  for (auto const& state : decomposition.spectrum())
  {
    std::cout << "    q=" << uni20::to_string(state.id.sector) << ", local-index=" << state.id.index
              << ", singular-value=" << state.singular_value << '\n';
  }
  std::cout << "  kept states: " << kept.truncation().retained_rank << '\n';
  std::cout << "  kept bond sectors: " << kept_factors.singular_values.bond_space().size() << '\n';
  std::cout << "  discarded states: " << discarded.truncation().retained_rank << '\n';
  std::cout << "  discarded weight: " << kept.truncation().discarded_weight << '\n';
  std::cout << "  independent bond labels: " << kept_factors.singular_values.bond_space().label() << ", "
            << discarded_factors.singular_values.bond_space().label() << '\n';
}
