#include <uni20/models/heisenberg.hpp>
#include <uni20/mps/block_sparse_mps.hpp>
#include <uni20/mps/sparse_mpo_site.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

using namespace uni20;

namespace
{

auto one_sector(QNum q, std::size_t dim = 1) -> BlockSpace { return BlockSpace(q.symmetry(), {BlockSector{q, dim}}); }

auto find_sector(std::vector<TwoSiteSvdSector> const& sectors, U1 charge) -> TwoSiteSvdSector const&
{
  auto const it = std::find_if(sectors.begin(), sectors.end(), [&](TwoSiteSvdSector const& sector) {
    return u1_component(sector.shared_q, "Sz") == charge;
  });
  if (it == sectors.end())
  {
    throw std::logic_error("missing requested SVD sector");
  }
  return *it;
}

} // namespace

TEST(ThreeLegBlockMatrixTest, BuildsAllowedU1Blocks)
{
  auto const spin = make_spin_half_u1_site();
  BlockSpace const left = one_sector(QNum::identity(spin.symmetry), 1);
  BlockSpace const right(spin.symmetry, {BlockSector{spin.up, 2}, BlockSector{spin.down, 3}});

  auto tensor = ThreeLegBlockMatrix::with_allowed_blocks(left, spin.space, right);

  EXPECT_EQ(tensor.block_count(), 2);
  ThreeLegBlockKey const up_key{.row_sector = 0, .local = 0, .col_sector = 0};
  ThreeLegBlockKey const down_key{.row_sector = 0, .local = 1, .col_sector = 1};
  ThreeLegBlockKey const forbidden_key{.row_sector = 0, .local = 0, .col_sector = 1};
  EXPECT_TRUE(tensor.contains(up_key));
  EXPECT_TRUE(tensor.contains(down_key));
  EXPECT_FALSE(tensor.contains(forbidden_key));
  EXPECT_EQ(tensor.block(up_key).rows, 1);
  EXPECT_EQ(tensor.block(up_key).cols, 2);
  EXPECT_EQ(tensor.block(down_key).cols, 3);
  EXPECT_EQ(tensor.blocks_for_local(0).size(), 1);
  EXPECT_EQ(tensor.blocks_for_local(1).size(), 1);

  tensor.assign_block(up_key, std::array{2.0, 3.0});
  EXPECT_DOUBLE_EQ(tensor.values(up_key)[0], 2.0);
  EXPECT_DOUBLE_EQ(tensor.values(up_key)[1], 3.0);
  auto all_values = tensor.coalesced_values();
  ASSERT_EQ(all_values.size(), 5);
  EXPECT_EQ(tensor.values(up_key).data(), all_values.data());
  EXPECT_EQ(tensor.values(down_key).data(), all_values.data() + tensor.block(up_key).size());
  EXPECT_THROW(tensor.insert_zero_block(forbidden_key), std::invalid_argument);
}

TEST(SparseMpoSiteTest, CompilesHeisenbergComponentIntoChargeCheckedScalarEntries)
{
  auto const spin = make_spin_half_u1_site();
  auto const component = make_spin_half_heisenberg_bulk_component(spin, 2.0, 0.25);

  SparseMpoSite const mpo(component);

  EXPECT_EQ(mpo.nnz(), 14);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 0, .bra = 0, .ket = 0, .right_virtual = 0}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 0, .bra = 1, .ket = 1, .right_virtual = 0}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 0, .bra = 0, .ket = 1, .right_virtual = 1}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 0, .bra = 1, .ket = 0, .right_virtual = 2}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 0, .bra = 0, .ket = 0, .right_virtual = 4}), 0.125);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 1, .bra = 1, .ket = 0, .right_virtual = 4}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 2, .bra = 0, .ket = 1, .right_virtual = 4}), 1.0);
  EXPECT_DOUBLE_EQ(mpo.at(SparseMpoEntryKey{.left_virtual = 3, .bra = 0, .ket = 0, .right_virtual = 4}), 1.0);
  EXPECT_EQ(mpo.entries_from_left_virtual(0).size(), 8);
  EXPECT_EQ(mpo.entries_to_right_virtual(4).size(), 8);

  for (auto const& entry : mpo.entries())
  {
    EXPECT_TRUE(sparse_mpo_entry_allowed(mpo.left_virtual_space(), mpo.bra_space(), mpo.ket_space(),
                                         mpo.right_virtual_space(), entry.key));
  }
}

TEST(SparseMpoSiteTest, RejectsLocalOperatorCoefficientsWithWrongTransformCharge)
{
  auto const spin = make_spin_half_u1_site();
  auto const virtual_space = LocalSpace(QNum::identity(spin.symmetry));
  LocalOperator bad(spin.space, spin.space, spin.sp.transforms_as());
  bad.insert_or_assign(0, 0, 1.0);

  OperatorComponent component(spin.space, spin.space, virtual_space, virtual_space);
  component.insert_or_assign(0, 0, std::move(bad));

  EXPECT_THROW(static_cast<void>(SparseMpoSite(component)), std::invalid_argument);
}

TEST(BlockSparseFiniteMPSTest, BuildsAlternatingU1ProductStateWithCumulativeBondCharges)
{
  auto const spin = make_spin_half_u1_site();
  auto const indices = make_alternating_spin_half_indices(3);

  auto psi = make_block_sparse_product_state(spin.space, indices);

  ASSERT_EQ(psi.size(), 3);
  EXPECT_TRUE(is_scalar(psi[0].row_space()[0].q));
  EXPECT_EQ(u1_component(psi[0].col_space()[0].q, "Sz"), U1{0.5});
  EXPECT_EQ(u1_component(psi[1].row_space()[0].q, "Sz"), U1{0.5});
  EXPECT_TRUE(is_scalar(psi[1].col_space()[0].q));
  EXPECT_EQ(u1_component(psi[2].col_space()[0].q, "Sz"), U1{0.5});

  for (std::size_t site = 0; site < psi.size(); ++site)
  {
    ThreeLegBlockKey const key{.row_sector = 0, .local = indices[site], .col_sector = 0};
    ASSERT_EQ(psi[site].block_count(), 1);
    EXPECT_TRUE(psi[site].contains(key));
    EXPECT_DOUBLE_EQ(psi[site].values(key)[0], 1.0);
  }
}

TEST(TwoSiteSvdSectorTest, FusesBoundaryTwoSiteSpinHalfSectors)
{
  auto const spin = make_spin_half_u1_site();
  BlockSpace const boundary = one_sector(QNum::identity(spin.symmetry), 1);

  auto const sectors = make_two_site_svd_sectors(spin.space, spin.space, boundary, boundary);

  ASSERT_EQ(sectors.size(), 2);
  auto const& plus = find_sector(sectors, U1{0.5});
  ASSERT_EQ(plus.rows.size(), 1);
  ASSERT_EQ(plus.cols.size(), 1);
  EXPECT_EQ(plus.row_dim, 1);
  EXPECT_EQ(plus.col_dim, 1);
  EXPECT_EQ(plus.rows[0].left_physical, 0);
  EXPECT_EQ(plus.cols[0].right_physical, 1);

  auto const& minus = find_sector(sectors, U1{-0.5});
  ASSERT_EQ(minus.rows.size(), 1);
  ASSERT_EQ(minus.cols.size(), 1);
  EXPECT_EQ(minus.rows[0].left_physical, 1);
  EXPECT_EQ(minus.cols[0].right_physical, 0);

  std::vector<std::size_t> const ranks{2, 3};
  auto const shared = make_shared_bond_space_from_sector_ranks(sectors, ranks);
  ASSERT_EQ(shared.size(), 2);
  EXPECT_EQ(shared.total_dim(), 5);
  EXPECT_TRUE(shared.contains(plus.shared_q));
  EXPECT_TRUE(shared.contains(minus.shared_q));
}
