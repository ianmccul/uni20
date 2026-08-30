#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/u1.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace uni20;

namespace
{

auto block_coordinate(BlockSpace const& space, QNum const& qnum) -> std::size_t
{
  for (std::size_t coordinate = 0; coordinate < space.size(); ++coordinate)
  {
    if (space[coordinate].q == qnum) return coordinate;
  }
  throw std::invalid_argument("quantum number is not present in the block space");
}

auto local_coordinate(LocalSpace const& space, QNum const& qnum) -> std::size_t
{
  for (std::size_t coordinate = 0; coordinate < space.size(); ++coordinate)
  {
    if (space[coordinate] == qnum) return coordinate;
  }
  throw std::invalid_argument("quantum number is not present in the local space");
}

void require_close(double actual, double expected)
{
  if (std::abs(actual - expected) > 1.0e-12)
  {
    throw std::runtime_error("AKLT BlockTensor example produced the wrong value");
  }
}

} // namespace

int main()
{
  Symmetry const symmetry{"Sz:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q_minus_half = make_qnum(symmetry, {{"Sz", U1{half_int{-0.5}}}});
  auto const q_plus_half = make_qnum(symmetry, {{"Sz", U1{half_int{0.5}}}});
  auto const q_minus_one = make_qnum(symmetry, {{"Sz", -1}});
  auto const q_plus_one = make_qnum(symmetry, {{"Sz", 1}});

  BlockSpace const virtual_spin_half(symmetry, {{q_minus_half, 1}, {q_plus_half, 1}}, "AKLT-virtual");
  LocalSpace const physical_spin_one(symmetry, {q_plus_one, q0, q_minus_one}, "spin-1");
  auto const minus_half = block_coordinate(virtual_spin_half, q_minus_half);
  auto const plus_half = block_coordinate(virtual_spin_half, q_plus_half);
  auto const minus_one = local_coordinate(physical_spin_one, q_minus_one);
  auto const zero = local_coordinate(physical_spin_one, q0);
  auto const plus_one = local_coordinate(physical_spin_one, q_plus_one);

  using Site = BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  // U(1) conservation leaves the two diagonal and two spin-flip blocks.
  typename Site::key_type const plus_key{{minus_half, plus_one, plus_half}};
  typename Site::key_type const zero_from_minus_key{{minus_half, zero, minus_half}};
  typename Site::key_type const zero_from_plus_key{{plus_half, zero, plus_half}};
  typename Site::key_type const minus_key{{plus_half, minus_one, minus_half}};
  Site site(symmetry, Domain{virtual_spin_half, physical_spin_one}, Codomain{virtual_spin_half},
            {plus_key, zero_from_minus_key, zero_from_plus_key, minus_key});

  double const off_diagonal = std::sqrt(2.0 / 3.0);
  double const diagonal = 1.0 / std::sqrt(3.0);
  // This gauge is the charge-oriented transpose of the common AKLT matrices.
  site.block(plus_key)[0, 0] = off_diagonal;
  site.block(zero_from_minus_key)[0, 0] = -diagonal;
  site.block(zero_from_plus_key)[0, 0] = diagonal;
  site.block(minus_key)[0, 0] = -off_diagonal;

  require_close(norm_host(site), std::sqrt(2.0));
  auto two_site_state = contract<2, 0>(site, site);
  typename decltype(two_site_state)::key_type const plus_minus_key{{minus_half, plus_one, minus_one, minus_half}};
  typename decltype(two_site_state)::key_type const zero_zero_key{{minus_half, zero, zero, minus_half}};
  double const plus_minus_amplitude = two_site_state.block(plus_minus_key)[0, 0];
  double const zero_zero_amplitude = two_site_state.block(zero_zero_key)[0, 0];
  require_close(plus_minus_amplitude, -2.0 / 3.0);
  require_close(zero_zero_amplitude, 1.0 / 3.0);

  std::cout << "U(1)-resolved spin-1 AKLT tensor\n";
  std::cout << "  one-site blocks: " << site.stored_block_count() << '\n';
  std::cout << "  one-site Frobenius norm: " << norm_host(site) << '\n';
  std::cout << "  two-site blocks: " << two_site_state.stored_block_count() << '\n';
  std::cout << "  amplitude(+1,-1): " << plus_minus_amplitude << '\n';
  std::cout << "  amplitude(0,0): " << zero_zero_amplitude << '\n';
}
