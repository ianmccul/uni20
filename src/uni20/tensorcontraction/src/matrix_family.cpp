#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace uni20::tensorcontraction
{

struct MatrixFamily::Impl
{
    std::vector<Block> blocks;
    std::vector<std::vector<double>> storage;
    std::vector<tensor::Matrix> matrices;
};

namespace
{

int checked_extent(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::length_error("TensorContraction matrix dimensions must fit in int");
  }
  return static_cast<int>(value);
}

std::size_t checked_block_size(MatrixFamily::Block block)
{
  if (block.rows != 0 && block.cols > std::numeric_limits<std::size_t>::max() / block.rows)
  {
    throw std::length_error("TensorContraction matrix block size overflows size_t");
  }
  return block.rows * block.cols;
}

} // namespace

MatrixFamily::MatrixFamily() : impl_(std::make_unique<Impl>()) {}

MatrixFamily::MatrixFamily(std::span<Block const> blocks) : MatrixFamily()
{
  impl_->blocks.assign(blocks.begin(), blocks.end());
  impl_->storage.reserve(blocks.size());
  impl_->matrices.reserve(blocks.size());

  for (Block const block : blocks)
  {
    auto& values = impl_->storage.emplace_back(checked_block_size(block));
    impl_->matrices.emplace_back(values.data(), checked_extent(block.rows), checked_extent(block.cols));
  }
}

MatrixFamily::MatrixFamily(MatrixFamily&&) noexcept = default;
MatrixFamily& MatrixFamily::operator=(MatrixFamily&&) noexcept = default;
MatrixFamily::~MatrixFamily() = default;

std::size_t MatrixFamily::size() const noexcept { return impl_->blocks.size(); }

bool MatrixFamily::empty() const noexcept { return impl_->blocks.empty(); }

std::span<MatrixFamily::Block const> MatrixFamily::blocks() const noexcept { return impl_->blocks; }

MatrixFamily::Block MatrixFamily::block(std::size_t index) const { return impl_->blocks.at(index); }

std::span<double> MatrixFamily::values(std::size_t index) { return impl_->storage.at(index); }

std::span<double const> MatrixFamily::values(std::size_t index) const { return impl_->storage.at(index); }

void MatrixFamily::assign(std::size_t index, std::span<double const> values)
{
  auto dst = this->values(index);
  if (dst.size() != values.size())
  {
    throw std::invalid_argument("TensorContraction matrix block assignment has the wrong size");
  }
  std::copy(values.begin(), values.end(), dst.begin());
}

void MatrixFamily::assign(MatrixFamily const& other)
{
  if (this->blocks().size() != other.blocks().size())
  {
    throw std::invalid_argument("TensorContraction matrix family assignment has the wrong block count");
  }

  for (std::size_t i = 0; i < this->blocks().size(); ++i)
  {
    if (this->block(i) != other.block(i))
    {
      throw std::invalid_argument("TensorContraction matrix family assignment has incompatible block shapes");
    }
    this->assign(i, other.values(i));
  }
}

void MatrixFamily::fill(double value)
{
  for (auto& block_storage : impl_->storage)
  {
    std::fill(block_storage.begin(), block_storage.end(), value);
  }
}

std::vector<tensor::Matrix>& raw_matrices(MatrixFamily& family) { return family.impl_->matrices; }

std::vector<tensor::Matrix> const& raw_matrices(MatrixFamily const& family) { return family.impl_->matrices; }

void broadcast_values_from_rank_zero(MatrixFamily& family)
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized == 0)
  {
    return;
  }

  int finalized = 0;
  MPI_Finalized(&finalized);
  if (finalized != 0)
  {
    return;
  }

  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size <= 1)
  {
    return;
  }

  for (std::size_t block = 0; block < family.size(); ++block)
  {
    auto values = family.values(block);
    std::size_t offset = 0;
    while (offset < values.size())
    {
      auto const remaining = values.size() - offset;
      auto const count = std::min<std::size_t>(remaining, static_cast<std::size_t>(std::numeric_limits<int>::max()));
      MPI_Bcast(values.data() + offset, static_cast<int>(count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      offset += count;
    }
  }
}

} // namespace uni20::tensorcontraction
