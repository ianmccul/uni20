#include <uni20/tensorcontraction/matrix_family.hpp>

#include "Matrix.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace uni20::tensorcontraction {

struct MatrixFamily::Impl {
  std::vector<Block> blocks;
  std::vector<std::vector<double>> storage;
  std::vector<tensor::Matrix> matrices;
};

namespace {

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

}  // namespace

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

MatrixFamily::Block MatrixFamily::block(std::size_t index) const
{
  return impl_->blocks.at(index);
}

std::span<double> MatrixFamily::values(std::size_t index)
{
  return impl_->storage.at(index);
}

std::span<double const> MatrixFamily::values(std::size_t index) const
{
  return impl_->storage.at(index);
}

void MatrixFamily::assign(std::size_t index, std::span<double const> values)
{
  auto dst = this->values(index);
  if (dst.size() != values.size())
  {
    throw std::invalid_argument("TensorContraction matrix block assignment has the wrong size");
  }
  std::copy(values.begin(), values.end(), dst.begin());
}

void MatrixFamily::fill(double value)
{
  for (auto& block_storage : impl_->storage)
  {
    std::fill(block_storage.begin(), block_storage.end(), value);
  }
}

std::vector<tensor::Matrix>& raw_matrices(MatrixFamily& family)
{
  return family.impl_->matrices;
}

std::vector<tensor::Matrix> const& raw_matrices(MatrixFamily const& family)
{
  return family.impl_->matrices;
}

}  // namespace uni20::tensorcontraction
