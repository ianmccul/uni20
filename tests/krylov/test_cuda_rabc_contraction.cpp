#include <uni20/async/debug_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/krylov/block_tensor_vector.hpp>
#include <uni20/linalg/dispatch_diagnostics.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor/copy_into.hpp>
#include <uni20/tensor_network/dmrg_lanczos.hpp>
#include <uni20/tensor_network/rabc_contraction.hpp>
#include <uni20/tensor_network/site_types.hpp>
#include <uni20/tensor_network/two_site_dmrg.hpp>
#include <uni20/tensor_network/two_site_effective_hamiltonian.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string_view>
#include <vector>

namespace
{

using host_storage = uni20::PackedCompleteBlockStorage<>;
using cuda_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
using domain_type = uni20::Domain<uni20::BlockSpace>;
using codomain_type = uni20::Codomain<uni20::BlockSpace>;

template <class Storage> using matrix_blocks = uni20::BlockTensor<double, domain_type, codomain_type, Storage>;

using host_matrix_blocks = matrix_blocks<host_storage>;
using cuda_matrix_blocks = matrix_blocks<cuda_storage>;
using key_type = typename host_matrix_blocks::key_type;
using plan_type = uni20::tensor_network::RabcContractionPlan<double, key_type, key_type, key_type, key_type>;
using term_type = uni20::tensor_network::RabcTerm<double>;

static_assert(uni20::BlockTensorView<cuda_matrix_blocks>);
static_assert(uni20::MutableBlockTensorView<cuda_matrix_blocks>);
static_assert(!uni20::ImmediateBlockTensorView<cuda_matrix_blocks>);

template <class Storage>
auto make_matrix_blocks(uni20::Symmetry const& symmetry, uni20::BlockSpace const& space) -> matrix_blocks<Storage>
{
  return matrix_blocks<Storage>(symmetry, domain_type{space}, codomain_type{space});
}

void set_identity(auto block, double factor)
{
  for (uni20::index_type row = 0; row < block.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      block[row, column] = row == column ? factor : 0.0;
  }
}

template <class Output, class Input> void copy_blocks(Output& output, Input const& input)
{
  ASSERT_TRUE(std::ranges::equal(output.stored_keys(), input.stored_keys()));
  for (std::size_t ordinal = 0; ordinal < output.stored_block_count(); ++ordinal)
  {
    auto output_block = output.block_by_ordinal(ordinal);
    auto input_block = input.block_by_ordinal(ordinal);
    uni20::copy(output_block, input_block);
  }
}

template <uni20::BlockTensorView Tensor> void expect_cuda_blocks_on_device(Tensor const& tensor, int device)
{
  for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
  {
    auto block = tensor.block_by_ordinal(ordinal);
    auto descriptor = uni20::mdspec_of(block);
    EXPECT_EQ(descriptor.data_descriptor().buffer().device().ordinal(), device);
  }
}

auto selected_backend(std::vector<uni20::linalg::dispatch_diagnostics::event> const& events,
                      std::string_view operation) -> std::string_view
{
  auto const found = std::ranges::find_if(
      events, [&](auto const& event) { return event.operation == operation && event.selected_backend().has_value(); });
  return found == events.end() ? std::string_view{} : *found->selected_backend();
}

class CudaRabcContractionTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess) GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

TEST_F(CudaRabcContractionTest, PreparedRightFirstContractionStaysInCudaStorage)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", uni20::U1{1}}});
  uni20::BlockSpace const space(symmetry, {{q0, 2}, {q1, 2}}, "matrix-space");

  auto host_a = make_matrix_blocks<host_storage>(symmetry, space);
  auto host_b = make_matrix_blocks<host_storage>(symmetry, space);
  auto host_c = make_matrix_blocks<host_storage>(symmetry, space);
  auto expected = make_matrix_blocks<host_storage>(symmetry, space);
  set_identity(host_a.block_by_ordinal(0), 1.0);
  set_identity(host_a.block_by_ordinal(1), 2.0);
  set_identity(host_c.block_by_ordinal(0), 1.0);
  auto host_b0 = host_b.block_by_ordinal(0);
  host_b0[0, 0] = 1.0;
  host_b0[0, 1] = 2.0;
  host_b0[1, 0] = 3.0;
  host_b0[1, 1] = 4.0;

  std::vector<key_type> const keys(host_a.stored_keys().begin(), host_a.stored_keys().end());
  plan_type const plan(
      keys, keys, keys, keys,
      std::vector<term_type>{
          {.r_key_index = 0, .a_key_index = 0, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0},
          {.r_key_index = 1, .a_key_index = 1, .b_key_index = 0, .c_key_index = 0, .coefficient = 1.0}});
  uni20::tensor_network::rabc_contract(expected, plan, host_a, host_b, host_c);

  auto device_a = make_matrix_blocks<cuda_storage>(symmetry, space);
  auto device_b = make_matrix_blocks<cuda_storage>(symmetry, space);
  auto device_c = make_matrix_blocks<cuda_storage>(symmetry, space);
  auto device_output = make_matrix_blocks<cuda_storage>(symmetry, space);
  copy_blocks(device_a, host_a);
  copy_blocks(device_b, host_b);
  copy_blocks(device_c, host_c);

  EXPECT_DOUBLE_EQ(uni20::inner_product_host(device_a, device_a), 10.0);
  EXPECT_DOUBLE_EQ(uni20::norm_host(device_a), std::sqrt(10.0));

  auto prepared = uni20::tensor_network::prepare_rabc_contract(device_output, plan, device_a, device_b, device_c);
  EXPECT_EQ(prepared.intermediate_count(), 1);

  std::vector<uni20::linalg::dispatch_diagnostics::event> events;
  uni20::linalg::dispatch_diagnostics::scoped_sink capture([&](auto const& event) { events.push_back(event); });
  prepared(device_output, device_a, device_b, device_c);

  EXPECT_EQ(selected_backend(events, "contract"), "direct_gemm_contraction");
  EXPECT_EQ(selected_backend(events, "gemm"), "cublas");
  EXPECT_EQ(device_output.storage().buffer().device().ordinal(), 0);

  auto actual = make_matrix_blocks<host_storage>(symmetry, space);
  copy_blocks(actual, device_output);
  for (std::size_t ordinal = 0; ordinal < actual.stored_block_count(); ++ordinal)
  {
    auto expected_block = expected.block_by_ordinal(ordinal);
    auto actual_block = actual.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < actual_block.extent(1); ++column)
      for (uni20::index_type row = 0; row < actual_block.extent(0); ++row)
        EXPECT_DOUBLE_EQ((actual_block[row, column]), (expected_block[row, column]));
  }
}

TEST_F(CudaRabcContractionTest, FixedStepDmrgLanczosKeepsLocalVectorsAndMatvecsOnCuda)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 4});
  uni20::cuda::ScopedDevice scoped_device(0);

  using center_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
  using environment_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
  using center_type = uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace,
                                                           uni20::LocalSpace, uni20::BlockSpace, center_storage>;
  using environment_type = uni20::tensor_network::MpoEnvironment<double, uni20::BlockSpace, uni20::LocalSpace,
                                                                 uni20::BlockSpace, environment_storage>;
  using mpo_type = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                  uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
  using host_center_type =
      uni20::tensor_network::TwoSiteCenter<double, uni20::BlockSpace, uni20::LocalSpace, uni20::LocalSpace,
                                           uni20::BlockSpace, uni20::PackedCompleteBlockStorage<>>;
  using host_environment_type =
      uni20::tensor_network::MpoEnvironment<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace,
                                            uni20::PackedCompleteBlockStorage<>>;

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const left_bond(symmetry, {{q0, 2}}, "left-bond");
  uni20::BlockSpace const right_bond(symmetry, {{q0, 2}}, "right-bond");
  uni20::LocalSpace const left_physical(symmetry, {q0}, "left-physical");
  uni20::LocalSpace const right_physical(symmetry, {q0}, "right-physical");
  uni20::LocalSpace const left_auxiliary(symmetry, {q0}, "left-auxiliary");
  uni20::LocalSpace const middle_auxiliary(symmetry, {q0}, "middle-auxiliary");
  uni20::LocalSpace const right_auxiliary(symmetry, {q0}, "right-auxiliary");

  host_center_type host_initial(symmetry, uni20::Domain{left_bond, left_physical, right_physical},
                                uni20::Codomain{right_bond});
  auto host_initial_block = host_initial.block_by_ordinal(0);
  host_initial_block[0, 0] = 1.0;
  host_initial_block[1, 0] = 1.0;
  host_initial_block[0, 1] = 1.0;
  host_initial_block[1, 1] = 1.0;

  host_environment_type host_left(symmetry, uni20::Domain{left_bond, left_auxiliary}, uni20::Codomain{left_bond});
  host_environment_type host_right(symmetry, uni20::Domain{right_bond, right_auxiliary}, uni20::Codomain{right_bond});
  auto host_left_block = host_left.block_by_ordinal(0);
  host_left_block[0, 0] = 1.0;
  host_left_block[1, 0] = 0.0;
  host_left_block[0, 1] = 0.0;
  host_left_block[1, 1] = 2.0;
  auto host_right_block = host_right.block_by_ordinal(0);
  host_right_block[0, 0] = 3.0;
  host_right_block[1, 0] = 0.0;
  host_right_block[0, 1] = 0.0;
  host_right_block[1, 1] = 4.0;

  using mpo_key = typename mpo_type::key_type;
  mpo_key const key{{0, 0, 0, 0}};
  mpo_type first_mpo(symmetry, uni20::Domain{left_auxiliary, left_physical},
                     uni20::Codomain{middle_auxiliary, left_physical}, {key});
  mpo_type second_mpo(symmetry, uni20::Domain{middle_auxiliary, right_physical},
                      uni20::Codomain{right_auxiliary, right_physical}, {key});
  first_mpo.block(key)[] = 1.0;
  second_mpo.block(key)[] = 1.0;

  center_type initial(symmetry, host_initial.domain(), host_initial.codomain());
  environment_type left_environment(symmetry, host_left.domain(), host_left.codomain());
  environment_type right_environment(symmetry, host_right.domain(), host_right.codomain());
  copy_blocks(initial, host_initial);
  copy_blocks(left_environment, host_left);
  copy_blocks(right_environment, host_right);

  auto effective_hamiltonian = uni20::tensor_network::make_two_site_effective_hamiltonian(
      initial, left_environment, first_mpo, second_mpo, right_environment);
  uni20::krylov::BlockTensorMatrixFreeOps ops(initial, std::move(effective_hamiltonian));
  auto result = uni20::tensor_network::dmrg_lanczos_ground_state<double>(ops, initial, {.matvec_iterations = 4});

  EXPECT_EQ(result.matvec_count, 4);
  EXPECT_NEAR(result.energy, 3.0, 1.0e-12);
  EXPECT_NEAR(ops.norm(result.vector), 1.0, 1.0e-12);
  EXPECT_EQ(result.vector.storage().buffer().device().ordinal(), 0);

  host_center_type host_result(symmetry, host_initial.domain(), host_initial.codomain());
  copy_blocks(host_result, result.vector);
  auto result_block = host_result.block_by_ordinal(0);
  EXPECT_NEAR(std::abs(result_block[0, 0]), 1.0, 1.0e-12);
  EXPECT_NEAR((result_block[1, 0]), 0.0, 1.0e-12);
  EXPECT_NEAR((result_block[0, 1]), 0.0, 1.0e-12);
  EXPECT_NEAR((result_block[1, 1]), 0.0, 1.0e-12);
}

TEST_F(CudaRabcContractionTest, WideBlockSvdRetainsFactorsOnCudaAndReconstructsTheSector)
{
  std::vector<int> devices{0};
  if (device_count_ > 1) devices.push_back(1);
  auto runtime = uni20::cuda::initialize({.device_ordinals = devices, .default_device = 0, .streams_per_device = 2});
  int const operand_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice scoped_device(operand_device);

  using device_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
  using device_matrix =
      uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>, device_storage>;
  using host_matrix = uni20::BlockTensor<double, uni20::Domain<uni20::BlockSpace>, uni20::Codomain<uni20::BlockSpace>,
                                         uni20::PackedCompleteBlockStorage<>>;

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const domain_space(symmetry, {{q0, 3}}, "domain");
  uni20::BlockSpace const codomain_space(symmetry, {{q0, 2}}, "codomain");
  host_matrix host(symmetry, uni20::Domain{domain_space}, uni20::Codomain{codomain_space});
  auto host_block = host.block_by_ordinal(0);
  host_block[0, 0] = 1.0;
  host_block[1, 0] = 2.0;
  host_block[2, 0] = -1.0;
  host_block[0, 1] = 3.0;
  host_block[1, 1] = 0.5;
  host_block[2, 1] = 4.0;

  auto& context = runtime.device_resources(operand_device);
  device_matrix device(symmetry, uni20::Domain{domain_space}, uni20::Codomain{codomain_space}, context);
  uni20::copy(device.block_by_ordinal(0), host_block);
  auto decomposition = uni20::block_svd(device);
  EXPECT_EQ(&decomposition.allocation_context(), &context);
  ASSERT_EQ(decomposition.sectors().size(), 1U);
  auto const& sector = decomposition.sectors()[0];
  EXPECT_EQ(sector.left_singular_vectors.storage().device().ordinal(), operand_device);
  EXPECT_EQ(sector.singular_values.storage().device().ordinal(), operand_device);
  EXPECT_EQ(sector.right_singular_vectors_adjoint.storage().device().ordinal(), operand_device);
  EXPECT_EQ(sector.left_singular_vectors.extent(0), 2);
  EXPECT_EQ(sector.left_singular_vectors.extent(1), 2);
  EXPECT_EQ(sector.right_singular_vectors_adjoint.extent(0), 2);
  EXPECT_EQ(sector.right_singular_vectors_adjoint.extent(1), 3);

  auto selection = uni20::select_svd_states(decomposition.spectrum());
  auto materialized = uni20::materialize_svd(decomposition, selection);
  EXPECT_EQ(&materialized.left_singular_vectors.allocation_context(), &context);
  EXPECT_EQ(&materialized.right_singular_vectors_adjoint.allocation_context(), &context);
  expect_cuda_blocks_on_device(materialized.left_singular_vectors, operand_device);
  expect_cuda_blocks_on_device(materialized.right_singular_vectors_adjoint, operand_device);

  uni20::ColumnMajorTensor<double, 2> left(2, 2);
  uni20::ColumnMajorTensor<double, 1> values(2);
  uni20::ColumnMajorTensor<double, 2> right(2, 3);
  uni20::copy(left, sector.left_singular_vectors);
  uni20::copy(values, sector.singular_values);
  uni20::copy(right, sector.right_singular_vectors_adjoint);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
    {
      double reconstructed = 0.0;
      for (uni20::index_type inner = 0; inner < 2; ++inner)
        reconstructed += left[row, inner] * values[inner] * right[inner, column];
      EXPECT_NEAR(reconstructed, (host_block[column, row]), 1.0e-12);
    }
  }
}

TEST_F(CudaRabcContractionTest, EmptyMappedBlockSvdRetainsCudaAllocationContext)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const domain_space(symmetry, {{q0, 1}}, "domain");
  uni20::BlockSpace const codomain_space(symmetry, {{q1, 1}}, "codomain");
  cuda_matrix_blocks empty(symmetry, domain_type{domain_space}, codomain_type{codomain_space},
                           runtime.device_resources(0));
  ASSERT_EQ(empty.stored_block_count(), 0U);

  auto mapped = uni20::as_block_tensor_view(empty);
  EXPECT_EQ(&mapped.allocation_context(), &runtime.device_resources(0));
  auto decomposition = uni20::block_svd(mapped);
  EXPECT_TRUE(decomposition.sectors().empty());
  EXPECT_TRUE(decomposition.spectrum().empty());
}

TEST_F(CudaRabcContractionTest, DefaultContractionPreservesCudaAllocationContext)
{
  std::vector<int> devices{0};
  if (device_count_ > 1) devices.push_back(1);
  auto runtime = uni20::cuda::initialize({.device_ordinals = devices, .default_device = 0, .streams_per_device = 2});
  int const operand_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice scoped_device(operand_device);

  using complete_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
  using matrix_type = matrix_blocks<complete_storage>;
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::BlockSpace const space(symmetry, {{q0, 1}}, "space");
  auto& context = runtime.device_resources(operand_device);
  matrix_type left(symmetry, domain_type{space}, codomain_type{space}, context);
  matrix_type right(symmetry, domain_type{space}, codomain_type{space}, context);
  uni20::ColumnMajorTensor<double, 2> host_left(1, 1);
  uni20::ColumnMajorTensor<double, 2> host_right(1, 1);
  host_left[0, 0] = 2.0;
  host_right[0, 0] = 3.0;
  uni20::copy(left.block_by_ordinal(0), host_left);
  uni20::copy(right.block_by_ordinal(0), host_right);

  auto adjacent_result = uni20::contract_adjacent<1>(left, right);
  static_assert(std::same_as<typename decltype(adjacent_result)::storage_policy,
                             uni20::PackedSparseBlockStorage<uni20::CudaStorage>>);
  EXPECT_EQ(&adjacent_result.allocation_context(), &context);
  EXPECT_EQ(adjacent_result.storage().buffer().device().ordinal(), operand_device);

  auto result = uni20::contract<1, 0>(left, right);
  static_assert(
      std::same_as<typename decltype(result)::storage_policy, uni20::PackedSparseBlockStorage<uni20::CudaStorage>>);
  EXPECT_EQ(&result.allocation_context(), &context);
  EXPECT_EQ(result.storage().buffer().device().ordinal(), operand_device);
  uni20::ColumnMajorTensor<double, 2> host_result(1, 1);
  uni20::copy(host_result, result.block_by_ordinal(0));
  EXPECT_DOUBLE_EQ((host_result[0, 0]), 6.0);
}

TEST_F(CudaRabcContractionTest, SparseMaterializationPreservesCudaAllocationContext)
{
  std::vector<int> devices{0};
  if (device_count_ > 1) devices.push_back(1);
  auto runtime = uni20::cuda::initialize({.device_ordinals = devices, .default_device = 0, .streams_per_device = 2});
  int const operand_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice scoped_device(operand_device);

  using sparse_storage = uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>;
  using sparse_matrix = matrix_blocks<sparse_storage>;
  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 1}, {q1, 1}}, "space");
  auto& context = runtime.device_resources(operand_device);
  sparse_matrix input(symmetry, domain_type{space}, codomain_type{space}, {key_type{{0, 0}}}, context);

  auto widened = uni20::tensor_network::detail::materialize_local_block_tensor<sparse_storage, true>(input);
  EXPECT_EQ(widened.stored_block_count(), 2U);
  EXPECT_EQ(&widened.allocation_context(), &context);
  expect_cuda_blocks_on_device(widened, operand_device);
}

TEST_F(CudaRabcContractionTest, TwoSiteDmrgStepKeepsStateAndEnvironmentsOnCuda)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 4});
  uni20::cuda::ScopedDevice scoped_device(0);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);

  using host_site_type =
      uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
  using site_storage = uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>;
  using site_type =
      uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace, site_storage>;
  using mpo_site_type = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                       uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
  using mps_type = uni20::tensor_network::FiniteMps<double, uni20::BlockSpace, uni20::LocalSpace, site_storage>;
  using mpo_type =
      uni20::tensor_network::FiniteMpo<double, uni20::LocalSpace, uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
  using site_key = typename site_type::key_type;
  using mpo_key = typename mpo_site_type::key_type;
  using dmrg_center_storage = uni20::ParallelPackedCompleteBlockStorage<uni20::CudaStorage>;
  using environment_storage = uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>;

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  auto const qminus1 = uni20::make_qnum(symmetry, {{"N", -1}});
  uni20::BlockSpace const left_bond(symmetry, {{q0, 1}}, "left-boundary");
  uni20::BlockSpace const middle_bond(symmetry, {{q0, 1}, {q1, 1}}, "middle-bond");
  uni20::BlockSpace const right_bond(symmetry, {{q1, 1}}, "right-boundary");
  uni20::LocalSpace const physical(symmetry, {q0, q1}, "physical");
  uni20::LocalSpace const left_auxiliary(symmetry, {q0}, "left-mpo");
  uni20::LocalSpace const middle_auxiliary(symmetry, {q0, qminus1, q1}, "middle-mpo");
  uni20::LocalSpace const right_auxiliary(symmetry, {q0}, "right-mpo");

  host_site_type host_first_site(symmetry, uni20::Domain{left_bond, physical}, uni20::Codomain{middle_bond},
                                 {site_key{{0, 0, 0}}});
  host_site_type host_second_site(symmetry, uni20::Domain{middle_bond, physical}, uni20::Codomain{right_bond},
                                  {site_key{{0, 1, 0}}, site_key{{1, 0, 0}}});
  host_first_site.block_by_ordinal(0)[0, 0] = 1.0;
  host_second_site.block_by_ordinal(0)[0, 0] = 1.0;
  host_second_site.block_by_ordinal(1)[0, 0] = 1.0;
  auto first_site = uni20::tensor_network::detail::materialize_local_block_tensor<site_storage, false>(host_first_site);
  auto second_site =
      uni20::tensor_network::detail::materialize_local_block_tensor<site_storage, false>(host_second_site);
  std::vector<site_type> sites;
  sites.reserve(2);
  sites.push_back(std::move(first_site));
  sites.push_back(std::move(second_site));
  mps_type mps(std::move(sites));

  mpo_site_type first_mpo(symmetry, uni20::Domain{left_auxiliary, physical},
                          uni20::Codomain{middle_auxiliary, physical},
                          {mpo_key{{0, 0, 0, 0}}, mpo_key{{0, 0, 1, 1}}, mpo_key{{0, 1, 0, 1}}, mpo_key{{0, 1, 2, 0}}});
  first_mpo.block(mpo_key{{0, 0, 0, 0}})[] = 0.5;
  first_mpo.block(mpo_key{{0, 1, 0, 1}})[] = -0.5;
  first_mpo.block(mpo_key{{0, 0, 1, 1}})[] = 0.5;
  first_mpo.block(mpo_key{{0, 1, 2, 0}})[] = 0.5;

  mpo_site_type second_mpo(
      symmetry, uni20::Domain{middle_auxiliary, physical}, uni20::Codomain{right_auxiliary, physical},
      {mpo_key{{0, 0, 0, 0}}, mpo_key{{0, 1, 0, 1}}, mpo_key{{1, 1, 0, 0}}, mpo_key{{2, 0, 0, 1}}});
  second_mpo.block(mpo_key{{0, 0, 0, 0}})[] = 0.5;
  second_mpo.block(mpo_key{{0, 1, 0, 1}})[] = -0.5;
  second_mpo.block(mpo_key{{1, 1, 0, 0}})[] = 1.0;
  second_mpo.block(mpo_key{{2, 0, 0, 1}})[] = 1.0;
  mpo_type mpo(std::vector<mpo_site_type>{std::move(first_mpo), std::move(second_mpo)});

  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, environment_storage> cache(mps, mpo, 0, 0);
  auto result = uni20::tensor_network::optimize_two_site_dmrg_bond(
      mps, mpo, cache, 0, uni20::tensor_network::MpsSweepDirection::left_to_right,
      uni20::tensor_network::TwoSiteDmrgOptions<double>{}, dmrg_center_storage{});

  EXPECT_NEAR(result.local_energy, -0.75, 1.0e-12);
  EXPECT_EQ(result.matvec_count, 2);
  EXPECT_EQ(result.bond.truncation.available_rank, 2);
  EXPECT_EQ(result.bond.truncation.retained_rank, 2);
  EXPECT_TRUE(cache.left_cached(1));
  auto device_center = uni20::contract_adjacent<1, dmrg_center_storage>(mps.site(0), mps.site(1));
  auto center = uni20::tensor_network::detail::materialize_local_block_tensor<uni20::PackedSparseBlockStorage<>, false>(
      device_center);
  double const up_down = center.block(typename decltype(center)::key_type{{0, 0, 1, 0}})[0, 0];
  double const down_up = center.block(typename decltype(center)::key_type{{0, 1, 0, 0}})[0, 0];
  EXPECT_NEAR(up_down + down_up, 0.0, 1.0e-12);
  EXPECT_NEAR(std::abs(up_down), std::sqrt(0.5), 1.0e-12);
}

TEST_F(CudaRabcContractionTest, DirectionalDmrgSweepsKeepMpsAndEnvironmentBlocksOnCuda)
{
  std::vector<int> devices{0};
  if (device_count_ > 1) devices.push_back(1);
  auto runtime = uni20::cuda::initialize({.device_ordinals = devices, .default_device = 0, .streams_per_device = 4});
  int const operand_device = device_count_ > 1 ? 1 : 0;
  uni20::cuda::ScopedDevice scoped_device(operand_device);
  uni20::async::DebugScheduler scheduler;
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);

  using host_site_type =
      uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace>;
  using site_storage = uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>;
  using site_type =
      uni20::tensor_network::MpsSite<double, uni20::BlockSpace, uni20::LocalSpace, uni20::BlockSpace, site_storage>;
  using mpo_site_type = uni20::tensor_network::MpoSite<double, uni20::LocalSpace, uni20::LocalSpace, uni20::LocalSpace,
                                                       uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
  using mps_type = uni20::tensor_network::FiniteMps<double, uni20::BlockSpace, uni20::LocalSpace, site_storage>;
  using mpo_type =
      uni20::tensor_network::FiniteMpo<double, uni20::LocalSpace, uni20::LocalSpace, uni20::PackedSparseBlockStorage<>>;
  using environment_storage = uni20::ParallelPackedSparseBlockStorage<uni20::CudaStorage>;
  using center_storage = uni20::ParallelPackedCompleteBlockStorage<uni20::CudaStorage>;
  using site_key = typename host_site_type::key_type;
  using mpo_key = typename mpo_site_type::key_type;

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  uni20::LocalSpace const physical(symmetry, {q0}, "physical");
  std::array bonds{uni20::BlockSpace(symmetry, {{q0, 1}}, "b0"), uni20::BlockSpace(symmetry, {{q0, 1}}, "b1"),
                   uni20::BlockSpace(symmetry, {{q0, 1}}, "b2"), uni20::BlockSpace(symmetry, {{q0, 1}}, "b3")};
  std::array auxiliaries{uni20::LocalSpace(symmetry, {q0}, "a0"), uni20::LocalSpace(symmetry, {q0}, "a1"),
                         uni20::LocalSpace(symmetry, {q0}, "a2"), uni20::LocalSpace(symmetry, {q0}, "a3")};

  auto make_host_site = [&](std::size_t site, double value) {
    host_site_type result(symmetry, uni20::Domain{bonds[site], physical}, uni20::Codomain{bonds[site + 1]},
                          {site_key{{0, 0, 0}}});
    result.block_by_ordinal(0)[0, 0] = value;
    return result;
  };
  std::vector<site_type> sites;
  sites.reserve(3);
  auto& context = runtime.device_resources(operand_device);
  for (std::size_t site = 0; site < 3; ++site)
  {
    auto host_site = make_host_site(site, static_cast<double>(site + 2));
    std::vector<typename site_type::key_type> keys(host_site.stored_keys().begin(), host_site.stored_keys().end());
    site_type device_site(symmetry, host_site.domain(), host_site.codomain(), std::move(keys), context);
    copy_blocks(device_site, host_site);
    sites.push_back(std::move(device_site));
  }
  mps_type mps(std::move(sites));

  std::vector<mpo_site_type> mpo_sites;
  mpo_sites.reserve(3);
  for (std::size_t site = 0; site < 3; ++site)
  {
    mpo_site_type mpo_site(symmetry, uni20::Domain{auxiliaries[site], physical},
                           uni20::Codomain{auxiliaries[site + 1], physical}, {mpo_key{{0, 0, 0, 0}}});
    mpo_site.block_by_ordinal(0)[] = 1.0;
    mpo_sites.push_back(std::move(mpo_site));
  }
  mpo_type mpo(std::move(mpo_sites));
  uni20::tensor_network::MpoEnvironmentCache<mps_type, mpo_type, environment_storage> cache(mps, mpo, 0, 0);

  auto rightward = uni20::tensor_network::sweep_two_site_dmrg(
      mps, mpo, cache, uni20::tensor_network::MpsSweepDirection::left_to_right,
      uni20::tensor_network::TwoSiteDmrgOptions<double>{}, center_storage{});
  ASSERT_EQ(rightward.size(), 2);
  EXPECT_EQ(rightward[0].first_site, 0);
  EXPECT_EQ(rightward[1].first_site, 1);

  auto leftward = uni20::tensor_network::sweep_two_site_dmrg(
      mps, mpo, cache, uni20::tensor_network::MpsSweepDirection::right_to_left,
      uni20::tensor_network::TwoSiteDmrgOptions<double>{}, center_storage{});
  ASSERT_EQ(leftward.size(), 2);
  EXPECT_EQ(leftward[0].first_site, 1);
  EXPECT_EQ(leftward[1].first_site, 0);

  for (std::size_t site = 0; site < mps.size(); ++site)
    expect_cuda_blocks_on_device(mps.site(site), operand_device);
  cache.build_all();
  for (std::size_t bond = 0; bond <= mps.size(); ++bond)
  {
    EXPECT_EQ(&cache.left_environment(bond).allocation_context(), &context);
    EXPECT_EQ(&cache.right_environment(bond).allocation_context(), &context);
    expect_cuda_blocks_on_device(cache.left_environment(bond), operand_device);
    expect_cuda_blocks_on_device(cache.right_environment(bond), operand_device);
  }
  EXPECT_EQ(mps.site(0).codomain().template space<0>().label(), "b1");
  EXPECT_EQ(mps.site(1).codomain().template space<0>().label(), "b2");
}

} // namespace
