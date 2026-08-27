#include <gtest/gtest.h>

#include <uni20/linalg/dispatch.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/symmetry/morphism_boundary.hpp>
#include <uni20/symmetry/u1.hpp>
#include <uni20/tensor_network/rabc_contraction.hpp>

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

using MatrixBlocks = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>,
                                        uni20::ParallelSeparateSparseBlockStorage<>>;
using Key = MatrixBlocks::key_type;
using Term = uni20::tensor_network::RabcTerm<double>;
using RabcPlan = uni20::tensor_network::RabcContractionPlan<double, Key, Key, Key, Key>;

struct NoContractBackend
{
    static constexpr std::string_view name = "no_contract";
};

using UnsupportedRabcBackend = uni20::tensor_network::HostRightFirstRabcBackend<NoContractBackend>;
static_assert(
    uni20::linalg::probe_dispatch_kernel_types<UnsupportedRabcBackend, uni20::tensor_network::rabc_contract_op,
                                               MatrixBlocks&, RabcPlan const&, MatrixBlocks const&, MatrixBlocks const&,
                                               MatrixBlocks const&>() == uni20::linalg::KernelTypeAcceptance::no);

auto make_matrix_blocks(uni20::Symmetry const& symmetry, uni20::BlockSpace const& space) -> MatrixBlocks
{
  return MatrixBlocks(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}, Key{{1, 1}}});
}

auto matrix_keys() -> std::vector<Key> { return {Key{{0, 0}}, Key{{1, 1}}}; }

auto make_plan(std::vector<Term> terms) -> RabcPlan
{
  auto const keys = matrix_keys();
  return RabcPlan(keys, keys, keys, keys, std::move(terms));
}

void set_identity(auto block, double factor)
{
  for (uni20::index_type row = 0; row < block.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      block[row, column] = row == column ? factor : 0.0;
  }
}

TEST(RabcContraction, CanonicalizesDuplicateSparseCoefficients)
{
  RabcPlan const plan = make_plan(
      {{.r_key_index = 1, .a_key_index = 0, .b_key_index = 1, .c_key_index = 0, .coefficient = 2.0},
       {.r_key_index = 0, .a_key_index = 1, .b_key_index = 0, .c_key_index = 1, .coefficient = 5.0},
       {.r_key_index = 1, .a_key_index = 0, .b_key_index = 1, .c_key_index = 0, .coefficient = -0.5},
       {.r_key_index = 0, .a_key_index = 1, .b_key_index = 0, .c_key_index = 1, .coefficient = -5.0}});

  ASSERT_EQ(plan.term_count(), 1);
  EXPECT_EQ(plan.terms()[0].r_key_index, 1);
  EXPECT_DOUBLE_EQ(plan.terms()[0].coefficient, 1.5);
}

TEST(RabcContraction, DispatchesRightFirstAndReusesSharedBcGroup)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", uni20::U1{1}}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 2}}, "matrix-space");

  auto a = make_matrix_blocks(symmetry, space);
  auto b = make_matrix_blocks(symmetry, space);
  auto c = make_matrix_blocks(symmetry, space);
  auto output = make_matrix_blocks(symmetry, space);
  set_identity(a.block_by_ordinal(0), 1.0);
  set_identity(a.block_by_ordinal(1), 2.0);
  set_identity(c.block_by_ordinal(0), 1.0);
  auto b0 = b.block_by_ordinal(0);
  b0[0, 0] = 1.0;
  b0[0, 1] = 2.0;
  b0[1, 0] = 3.0;
  b0[1, 1] = 4.0;

  RabcPlan const plan = make_plan(
      {{.r_key_index = 0, .a_key_index = 0, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0},
       {.r_key_index = 1, .a_key_index = 1, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0}});

  auto selector = uni20::linalg::select_backend(uni20::tensor_network::rabc_contract_op{}, output);
  EXPECT_EQ(
      uni20::linalg::probe_dispatch_kernel(selector, uni20::tensor_network::rabc_contract_op{}, output, plan, a, b, c),
      uni20::linalg::KernelTypeAcceptance::yes);
  auto prepared = uni20::tensor_network::prepare_rabc_contract(output, plan, a, b, c);
  EXPECT_EQ(prepared.intermediate_count(), 1);
  prepared(output, a, b, c);

  auto r0 = output.block_by_ordinal(0);
  auto r1 = output.block_by_ordinal(1);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_DOUBLE_EQ((r0[row, column]), (b0[row, column]));
      EXPECT_DOUBLE_EQ((r1[row, column]), (2.0 * b0[row, column]));
    }
  }

  for (uni20::index_type row = 0; row < 2; ++row)
    for (uni20::index_type column = 0; column < 2; ++column)
      b0[row, column] *= 3.0;
  prepared(output, a, b, c);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_DOUBLE_EQ((r0[row, column]), (b0[row, column]));
      EXPECT_DOUBLE_EQ((r1[row, column]), (2.0 * b0[row, column]));
    }
  }
}

TEST(RabcContraction, ZerosOutputBlocksWithoutTerms)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", uni20::U1{1}}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 2}}, "matrix-space");
  auto a = make_matrix_blocks(symmetry, space);
  auto b = make_matrix_blocks(symmetry, space);
  auto c = make_matrix_blocks(symmetry, space);
  auto output = make_matrix_blocks(symmetry, space);
  set_identity(a.block_by_ordinal(0), 1.0);
  set_identity(b.block_by_ordinal(0), 1.0);
  set_identity(c.block_by_ordinal(0), 1.0);
  set_identity(output.block_by_ordinal(1), 9.0);

  RabcPlan const plan = make_plan(
      {{.r_key_index = 0, .a_key_index = 0, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0}});
  uni20::tensor_network::rabc_contract(output, plan, a, b, c);

  auto unused = output.block_by_ordinal(1);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
      EXPECT_DOUBLE_EQ((unused[row, column]), 0.0);
  }
}

TEST(RabcContraction, RejectsMissingLogicalKeysAndObviousCenterAliasing)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", uni20::U1{1}}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 2}}, "matrix-space");
  auto a = make_matrix_blocks(symmetry, space);
  auto b = make_matrix_blocks(symmetry, space);
  auto c = make_matrix_blocks(symmetry, space);
  auto output = make_matrix_blocks(symmetry, space);

  auto invalid_r_keys = matrix_keys();
  invalid_r_keys.push_back(Key{{2, 2}});
  auto const keys = matrix_keys();
  RabcPlan const invalid_plan(
      std::move(invalid_r_keys), keys, keys, keys,
      {{.r_key_index = 2, .a_key_index = 0, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0}});
  EXPECT_THROW(uni20::tensor_network::rabc_contract(output, invalid_plan, a, b, c), std::invalid_argument);

  RabcPlan const aliasing_plan = make_plan(
      {{.r_key_index = 0, .a_key_index = 0, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0}});
  EXPECT_THROW(uni20::tensor_network::rabc_contract(b, aliasing_plan, a, b, c), std::invalid_argument);
}

} // namespace
