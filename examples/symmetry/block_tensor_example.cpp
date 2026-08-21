#include <uni20/symmetry/block_tensor_repartition.hpp>

#include <cstddef>
#include <iostream>

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

  using MpsSite =
      BlockTensor<double, Domain<BlockSpace, LocalSpace>, Codomain<BlockSpace>, SeparateSparseBlockStorage<>>;
  MpsSite mps_site(sym, Domain{left, physical}, Codomain{right},
                   {MpsSite::key_type{{0, 1, 1}}, MpsSite::key_type{{1, 1, 2}}});
  mps_site.block(MpsSite::key_type{{1, 1, 2}})[2, 5] = 2.0;
  auto bent_mps = repartition<MorphismSide::Domain, BoundaryEnd::Right>(mps_site);

  using MpoSite =
      BlockTensor<double, Domain<LocalSpace, LocalSpace>, Codomain<LocalSpace, LocalSpace>, PackedSparseBlockStorage<>>;
  MpoSite mpo_site(sym, Domain{auxiliary, physical}, Codomain{auxiliary, physical},
                   {MpoSite::key_type{{0, 0, 0, 0}}, MpoSite::key_type{{0, 1, 0, 1}}});
  mpo_site.block(MpoSite::key_type{{0, 1, 0, 1}})[] = 3.0;

  print_summary("matrix", matrix);
  print_summary("MPS site", mps_site);
  print_summary("bent MPS view", bent_mps);
  print_summary("MPO site", mpo_site);
  auto const source_mps_block = mps_site.block(MpsSite::key_type{{1, 1, 2}});
  auto const bent_mps_block = bent_mps.block(MpsSite::key_type{{1, 2, 1}});
  std::cout << "bent MPS reuses payload: " << std::boolalpha
            << (source_mps_block.data_handle() == bent_mps_block.data_handle()) << '\n';
  using SeparateScalarBlock = ColumnMajorTensor<MpoSite::element_type, 0, VectorStorage>;
  std::cout << "rank-zero block ABI bytes: payload=" << sizeof(MpoSite::element_type)
            << ", separate-owner-object=" << sizeof(SeparateScalarBlock)
            << ", transient-view=" << sizeof(MpoSite::mutable_block_type) << '\n';
  std::cout << "packed MPO metadata ABI bytes: key=" << sizeof(MpoSite::key_type) << ", offset=" << sizeof(std::size_t)
            << '\n';
}
