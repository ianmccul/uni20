#include <gtest/gtest.h>

#include <uni20/linalg/dispatch.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/symmetry/morphism_boundary.hpp>
#include <uni20/symmetry/u1.hpp>
#include <uni20/tensor_network/rabc_contraction.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

namespace
{

using MatrixBlocks = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>,
                                        uni20::ParallelSeparateSparseBlockStorage<>>;
using Key = MatrixBlocks::key_type;

struct NoContractBackend
{
    static constexpr std::string_view name = "no_contract";
};

using UnsupportedRabcBackend = uni20::tensor_network::HostRightFirstRabcBackend<NoContractBackend>;
using RabcPlan = uni20::tensor_network::RabcContractionPlan<double>;

static_assert(
    uni20::linalg::probe_dispatch_kernel_types<UnsupportedRabcBackend, uni20::tensor_network::rabc_contract_op,
                                               MatrixBlocks&, RabcPlan const&, MatrixBlocks const&, MatrixBlocks const&,
                                               MatrixBlocks const&>() == uni20::linalg::KernelTypeAcceptance::no);

auto make_matrix_blocks(uni20::Symmetry const& symmetry, uni20::BlockSpace const& space) -> MatrixBlocks
{
  return MatrixBlocks(symmetry, uni20::Domain{space}, uni20::Codomain{space}, {Key{{0, 0}}, Key{{1, 1}}});
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
  using Term = uni20::tensor_network::RabcTerm<double>;
  uni20::tensor_network::RabcContractionPlan<double> const plan(
      std::vector<Term>{{.r_ordinal = 1, .a_ordinal = 2, .b_ordinal = 3, .c_ordinal = 4, .coefficient = 2.0},
                        {.r_ordinal = 0, .a_ordinal = 1, .b_ordinal = 2, .c_ordinal = 3, .coefficient = 5.0},
                        {.r_ordinal = 1, .a_ordinal = 2, .b_ordinal = 3, .c_ordinal = 4, .coefficient = -0.5},
                        {.r_ordinal = 0, .a_ordinal = 1, .b_ordinal = 2, .c_ordinal = 3, .coefficient = -5.0}});

  ASSERT_EQ(plan.term_count(), 1);
  EXPECT_EQ(plan.terms()[0].r_ordinal, 1);
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

  using Term = uni20::tensor_network::RabcTerm<double>;
  uni20::tensor_network::RabcContractionPlan<double> const plan(
      std::vector<Term>{{.r_ordinal = 0, .a_ordinal = 0, .b_ordinal = 0, .c_ordinal = 0, .coefficient = 1.0},
                        {.r_ordinal = 1, .a_ordinal = 1, .b_ordinal = 0, .c_ordinal = 0, .coefficient = 1.0}});

  std::vector<std::size_t> term_group(plan.term_count());
  auto const groups = uni20::tensor_network::detail::make_right_first_groups(plan, term_group);
  ASSERT_EQ(groups.size(), 1);
  EXPECT_EQ(term_group[0], term_group[1]);

  auto selector = uni20::linalg::select_backend(uni20::tensor_network::rabc_contract_op{}, output);
  EXPECT_EQ(
      uni20::linalg::probe_dispatch_kernel(selector, uni20::tensor_network::rabc_contract_op{}, output, plan, a, b, c),
      uni20::linalg::KernelTypeAcceptance::yes);
  uni20::tensor_network::rabc_contract(output, plan, a, b, c);

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

  using Term = uni20::tensor_network::RabcTerm<double>;
  uni20::tensor_network::RabcContractionPlan<double> const plan(
      std::vector<Term>{{.r_ordinal = 0, .a_ordinal = 0, .b_ordinal = 0, .c_ordinal = 0, .coefficient = 1.0}});
  uni20::tensor_network::rabc_contract(output, plan, a, b, c);

  auto unused = output.block_by_ordinal(1);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
      EXPECT_DOUBLE_EQ((unused[row, column]), 0.0);
  }
}

TEST(RabcContraction, RejectsInvalidOrdinalsAndObviousCenterAliasing)
{
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", uni20::U1{1}}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 2}}, "matrix-space");
  auto a = make_matrix_blocks(symmetry, space);
  auto b = make_matrix_blocks(symmetry, space);
  auto c = make_matrix_blocks(symmetry, space);
  auto output = make_matrix_blocks(symmetry, space);

  using Term = uni20::tensor_network::RabcTerm<double>;
  uni20::tensor_network::RabcContractionPlan<double> const invalid_plan(
      std::vector<Term>{{.r_ordinal = 2, .a_ordinal = 0, .b_ordinal = 0, .c_ordinal = 0, .coefficient = 1.0}});
  EXPECT_THROW(uni20::tensor_network::rabc_contract(output, invalid_plan, a, b, c), std::invalid_argument);

  uni20::tensor_network::RabcContractionPlan<double> const aliasing_plan(
      std::vector<Term>{{.r_ordinal = 0, .a_ordinal = 0, .b_ordinal = 0, .c_ordinal = 0, .coefficient = 1.0}});
  EXPECT_THROW(uni20::tensor_network::rabc_contract(b, aliasing_plan, a, b, c), std::invalid_argument);
}

} // namespace
