#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"

#include <cuda_runtime_api.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>

namespace uni20::tensorcontraction
{

namespace
{

struct PinnedHostSlab
{
    double* data = nullptr;
    std::size_t count = 0;

    PinnedHostSlab() = default;
    explicit PinnedHostSlab(std::size_t value_count) : count(value_count)
    {
      if (count == 0)
      {
        return;
      }
      if (count > std::numeric_limits<std::size_t>::max() / sizeof(double))
      {
        throw std::length_error("TensorContraction pinned host slab size overflows size_t");
      }

      void* raw = nullptr;
      cudaError_t const status = cudaHostAlloc(&raw, count * sizeof(double), cudaHostAllocPortable);
      if (status != cudaSuccess)
      {
        throw std::bad_alloc();
      }
      data = static_cast<double*>(raw);
    }

    PinnedHostSlab(PinnedHostSlab const&) = delete;
    PinnedHostSlab& operator=(PinnedHostSlab const&) = delete;

    PinnedHostSlab(PinnedHostSlab&& other) noexcept : data(other.data), count(other.count)
    {
      other.data = nullptr;
      other.count = 0;
    }

    PinnedHostSlab& operator=(PinnedHostSlab&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        data = other.data;
        count = other.count;
        other.data = nullptr;
        other.count = 0;
      }
      return *this;
    }

    ~PinnedHostSlab() { this->release(); }

    void release() noexcept
    {
      if (data != nullptr)
      {
        (void)cudaFreeHost(data);
      }
      data = nullptr;
      count = 0;
    }
};

} // namespace

struct MatrixFamily::Impl
{
    std::vector<Block> blocks;
    std::vector<std::size_t> offsets;
    PinnedHostSlab storage;
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

std::size_t checked_total_size(std::span<MatrixFamily::Block const> blocks)
{
  std::size_t total = 0;
  for (MatrixFamily::Block const block : blocks)
  {
    std::size_t const block_size = checked_block_size(block);
    if (block_size > std::numeric_limits<std::size_t>::max() - total)
    {
      throw std::length_error("TensorContraction matrix family storage size overflows size_t");
    }
    total += block_size;
  }
  return total;
}

} // namespace

MatrixFamily::MatrixFamily() : impl_(std::make_unique<Impl>()) {}

MatrixFamily::MatrixFamily(std::span<Block const> blocks) : MatrixFamily()
{
  impl_->blocks.assign(blocks.begin(), blocks.end());
  impl_->offsets.reserve(blocks.size());
  impl_->matrices.reserve(blocks.size());
  impl_->storage = PinnedHostSlab(checked_total_size(blocks));

  std::size_t offset = 0;
  for (Block const block : blocks)
  {
    std::size_t const block_size = checked_block_size(block);
    impl_->offsets.push_back(offset);
    double* const block_data = block_size == 0 ? nullptr : impl_->storage.data + offset;
    auto& matrix = impl_->matrices.emplace_back(block_data, checked_extent(block.rows), checked_extent(block.cols));
    matrix.setHostMemoryKind(tensor::HostMemoryKind::Pinned);
    offset += block_size;
  }
}

MatrixFamily::MatrixFamily(MatrixFamily&&) noexcept = default;
MatrixFamily& MatrixFamily::operator=(MatrixFamily&&) noexcept = default;
MatrixFamily::~MatrixFamily() = default;

std::size_t MatrixFamily::size() const noexcept { return impl_->blocks.size(); }

bool MatrixFamily::empty() const noexcept { return impl_->blocks.empty(); }

std::span<MatrixFamily::Block const> MatrixFamily::blocks() const noexcept { return impl_->blocks; }

MatrixFamily::Block MatrixFamily::block(std::size_t index) const { return impl_->blocks.at(index); }

std::span<double> MatrixFamily::values(std::size_t index)
{
  std::size_t const offset = impl_->offsets.at(index);
  std::size_t const block_size = checked_block_size(impl_->blocks.at(index));
  if (block_size == 0)
  {
    return {};
  }
  return {impl_->storage.data + offset, block_size};
}

std::span<double const> MatrixFamily::values(std::size_t index) const
{
  std::size_t const offset = impl_->offsets.at(index);
  std::size_t const block_size = checked_block_size(impl_->blocks.at(index));
  if (block_size == 0)
  {
    return {};
  }
  return {impl_->storage.data + offset, block_size};
}

std::span<double> MatrixFamily::coalesced_values() noexcept { return {impl_->storage.data, impl_->storage.count}; }

std::span<double const> MatrixFamily::coalesced_values() const noexcept
{
  return {impl_->storage.data, impl_->storage.count};
}

std::size_t MatrixFamily::value_offset(std::size_t index) const { return impl_->offsets.at(index); }

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
  if (impl_->storage.count == 0)
  {
    return;
  }
  std::fill(impl_->storage.data, impl_->storage.data + impl_->storage.count, value);
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
