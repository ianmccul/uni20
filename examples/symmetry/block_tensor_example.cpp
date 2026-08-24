#include <uni20/symmetry/block_tensor_contract.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_repartition.hpp>

#include <uni20/async/debug_scheduler.hpp>

#include <cstddef>
#include <iostream>
#include <utility>

using namespace uni20;

template <class Tensor> void print_summary(char const* name, Tensor const& tensor)
{
  std::cout << name << ": order=" << tensor.order() << ", key-coordinates=" << tensor.key_coordinate_count()
            << ", dense-block-order=" << tensor.dense_block_order() << ", stored=" << tensor.stored_block_count()
            << ", legal=" << tensor.legal_block_count() << '\n';
}

int main()
{
  Symmetry const sym{"N:U(1)"};
  auto const q0 = make_qnum(sym, {{"N", 0}});
  auto const q1 = make_qnum(sym, {{"N", 1}});
  auto const q2 = make_qnum(sym, {{"N", 2}});

  BlockSpace const left(sym, {{q0, 2}, {q1, 3}}, "left");
  BlockSpace const right(sym, {{q0, 4}, {q1, 5}, {q2, 6}}, "right");
  LocalSpace const physical(sym, {q0, q1}, "physical");
  LocalSpace const auxiliary(sym, {q0, q1}, "auxiliary");

  using Matrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
  Matrix matrix(sym, Domain{left}, Codomain{left}, {Matrix::key_type{{0, 0}}, Matrix::key_type{{1, 1}}});
  matrix.block(Matrix::key_type{{1, 1}})[2, 2] = 1.0;
  auto matrix_squared = contract<1, 0>(matrix, matrix);

  using AsyncMatrix = BlockTensor<double, Domain<BlockSpace>, Codomain<BlockSpace>, AsyncSeparateSparseBlockStorage<>>;
  AsyncMatrix async_matrix(sym, Domain{left}, Codomain{left}, {AsyncMatrix::key_type{{1, 1}}});
  using AsyncDenseBlock = typename AsyncMatrix::storage_type::block_value_type;
  AsyncDenseBlock async_dense_block(3, 3);
  async_dense_block[2, 2] = 2.0;
  async_matrix.async_block(AsyncMatrix::key_type{{1, 1}}) = std::move(async_dense_block);
  async::DebugScheduler scheduler;
  async::ScopedScheduler scoped_scheduler(&scheduler);
  auto async_matrix_squared = contract<1, 0>(async_matrix, async_matrix);
  auto const& async_result_block = async_matrix_squared.async_block(AsyncMatrix::key_type{{1, 1}}).get_wait(scheduler);

  using MpsSite =
      BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
  MpsSite mps_site(sym, Domain{left, physical}, Codomain{right},
                   {MpsSite::key_type{{0, 1, 1}}, MpsSite::key_type{{1, 1, 2}}});
  mps_site.block(MpsSite::key_type{{1, 1, 2}})[2, 5] = 2.0;
  auto permuted_mps = permute<1, 0, 2>(mps_site);
  auto bent_mps = repartition<MorphismSide::Domain, BoundaryEnd::Right>(mps_site);

  using MpoSite =
      BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace, LocalSpace>, PackedSparseBlockStorage<>>;
  MpoSite mpo_site(sym, Domain{auxiliary, physical}, Codomain{auxiliary, physical},
                   {MpoSite::key_type{{0, 0, 0, 0}}, MpoSite::key_type{{0, 1, 0, 1}}});
  mpo_site.block(MpoSite::key_type{{0, 1, 0, 1}})[] = 3.0;

  print_summary("matrix", matrix);
  print_summary("matrix squared", matrix_squared);
  print_summary("async matrix squared", async_matrix_squared);
  print_summary("MPS site", mps_site);
  print_summary("permuted MPS view", permuted_mps);
  print_summary("bent MPS view", bent_mps);
  print_summary("MPO site", mpo_site);
  auto const source_mps_block = mps_site.block(MpsSite::key_type{{1, 1, 2}});
  auto const permuted_mps_block = permuted_mps.block(MpsSite::key_type{{1, 1, 2}});
  auto const bent_mps_block = bent_mps.block(MpsSite::key_type{{1, 2, 1}});
  std::cout << "permuted MPS reuses payload: " << std::boolalpha
            << (source_mps_block.mdspan().data_handle() == permuted_mps_block.mdspan().data_handle()) << '\n';
  std::cout << "bent MPS reuses payload: " << std::boolalpha
            << (source_mps_block.mdspan().data_handle() == bent_mps_block.mdspan().data_handle()) << '\n';
  using SeparateScalarBlock = ColumnMajorTensor<MpoSite::element_type, 0, VectorStorage>;
  std::cout << "rank-zero block ABI bytes: payload=" << sizeof(MpoSite::element_type)
            << ", separate-owner-object=" << sizeof(SeparateScalarBlock)
            << ", transient-view=" << sizeof(MpoSite::mutable_block_type) << '\n';
  std::cout << "packed MPO metadata ABI bytes: key=" << sizeof(MpoSite::key_type) << ", offset=" << sizeof(std::size_t)
            << '\n';
  std::cout << "async block GEMM result[2,2]: " << async_result_block[2, 2] << '\n';
}
