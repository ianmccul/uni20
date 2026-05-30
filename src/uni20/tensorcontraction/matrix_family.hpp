#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace tensor
{
class Matrix;
}

namespace uni20::tensorcontraction
{

class MatrixFamily {
  public:
    struct Block
    {
        std::size_t rows = 0;
        std::size_t cols = 0;

        bool operator==(Block const&) const = default;
    };

    MatrixFamily();
    explicit MatrixFamily(std::span<Block const> blocks);
    MatrixFamily(MatrixFamily&&) noexcept;
    MatrixFamily& operator=(MatrixFamily&&) noexcept;
    MatrixFamily(MatrixFamily const&) = delete;
    MatrixFamily& operator=(MatrixFamily const&) = delete;
    ~MatrixFamily();

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::span<Block const> blocks() const noexcept;
    [[nodiscard]] Block block(std::size_t index) const;
    [[nodiscard]] std::span<double> values(std::size_t index);
    [[nodiscard]] std::span<double const> values(std::size_t index) const;

    void assign(std::size_t index, std::span<double const> values);
    void assign(MatrixFamily const& other);
    void fill(double value);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend std::vector<tensor::Matrix>& raw_matrices(MatrixFamily& family);
    friend std::vector<tensor::Matrix> const& raw_matrices(MatrixFamily const& family);
};

std::vector<tensor::Matrix>& raw_matrices(MatrixFamily& family);
std::vector<tensor::Matrix> const& raw_matrices(MatrixFamily const& family);

} // namespace uni20::tensorcontraction
