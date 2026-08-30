#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/u1.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>

using namespace uni20;

namespace
{

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
    throw std::runtime_error("product-state contraction produced the wrong amplitude");
  }
}

} // namespace

int main()
{
  Symmetry const symmetry{"Sz:U(1)"};
  auto const q0 = QNum::identity(symmetry);
  auto const q_down = make_qnum(symmetry, {{"Sz", U1{half_int{-0.5}}}});
  auto const q_up = make_qnum(symmetry, {{"Sz", U1{half_int{0.5}}}});
  LocalSpace const spin_half(symmetry, {q_down, q_up}, "spin-1/2");

  // The cumulative bond charge follows 0 -> +1/2 -> 0 for |up down>.
  BlockSpace const bond_0(symmetry, {{q0, 1}}, "bond-0");
  BlockSpace const bond_1(symmetry, {{q_up, 1}}, "bond-1");
  BlockSpace const bond_2(symmetry, {{q0, 1}}, "bond-2");

  using Site = BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, PackedSparseBlockStorage<>>;
  auto const down = local_coordinate(spin_half, q_down);
  auto const up = local_coordinate(spin_half, q_up);
  // Each key obeys q_right = q_left + q_physical.
  typename Site::key_type const up_key{{0, up, 0}};
  typename Site::key_type const down_key{{0, down, 0}};

  Site first_site(symmetry, Domain{bond_0, spin_half}, Codomain{bond_1}, {up_key});
  Site second_site(symmetry, Domain{bond_1, spin_half}, Codomain{bond_2}, {down_key});
  first_site.block(up_key)[0, 0] = 1.0;
  second_site.block(down_key)[0, 0] = 1.0;

  auto two_site_state = contract<2, 0>(first_site, second_site);
  typename decltype(two_site_state)::key_type const up_down_key{{0, up, down, 0}};
  double const amplitude = two_site_state.block(up_down_key)[0, 0];
  require_close(amplitude, 1.0);

  std::cout << "spin-half product state |up down>\n";
  std::cout << "  site blocks: " << first_site.stored_block_count() << " + " << second_site.stored_block_count()
            << '\n';
  std::cout << "  two-site blocks: " << two_site_state.stored_block_count() << '\n';
  std::cout << "  amplitude(up,down): " << amplitude << '\n';
}
