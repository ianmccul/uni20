/**
 * \file finite_mps.hpp
 * \brief Minimal in-memory finite MPS helpers for the first DMRG prototype.
 */

#pragma once

#include <uni20/operator/local_space.hpp>
#include <uni20/symmetry/block_space.hpp>
#include <uni20/tensorcontraction/matrix_family.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20
{

class MpsSiteTensor {
  public:
    MpsSiteTensor(LocalSpace physical_space, BlockSpace left_bond_space, BlockSpace right_bond_space)
        : physical_space_(std::move(physical_space)), left_bond_space_(std::move(left_bond_space)),
          right_bond_space_(std::move(right_bond_space))
    {
      this->verify_spaces();
      blocks_.resize(physical_space_.size(), std::vector<double>(this->left_dim() * this->right_dim(), 0.0));
    }

    [[nodiscard]] LocalSpace const& physical_space() const noexcept { return physical_space_; }
    [[nodiscard]] BlockSpace const& left_bond_space() const noexcept { return left_bond_space_; }
    [[nodiscard]] BlockSpace const& right_bond_space() const noexcept { return right_bond_space_; }
    [[nodiscard]] std::size_t physical_dim() const noexcept { return physical_space_.size(); }
    [[nodiscard]] std::size_t left_dim() const noexcept { return left_bond_space_.total_dim(); }
    [[nodiscard]] std::size_t right_dim() const noexcept { return right_bond_space_.total_dim(); }

    [[nodiscard]] std::span<double> values(std::size_t physical_index) { return blocks_.at(physical_index); }
    [[nodiscard]] std::span<double const> values(std::size_t physical_index) const
    {
      return blocks_.at(physical_index);
    }

    void assign(std::size_t physical_index, std::span<double const> values)
    {
      auto dst = this->values(physical_index);
      if (dst.size() != values.size())
      {
        throw std::invalid_argument("MpsSiteTensor physical block assignment has the wrong size");
      }
      std::copy(values.begin(), values.end(), dst.begin());
    }

  private:
    void verify_spaces() const
    {
      auto const sym = physical_space_.symmetry();
      if (left_bond_space_.symmetry() != sym || right_bond_space_.symmetry() != sym)
      {
        throw std::invalid_argument("MpsSiteTensor spaces must share one symmetry");
      }
      if (physical_space_.empty())
      {
        throw std::invalid_argument("MpsSiteTensor requires a non-empty physical space");
      }
    }

    LocalSpace physical_space_;
    BlockSpace left_bond_space_;
    BlockSpace right_bond_space_;
    std::vector<std::vector<double>> blocks_;
};

class FiniteMPS {
  public:
    using container_type = std::vector<MpsSiteTensor>;
    using size_type = container_type::size_type;

    FiniteMPS() = default;
    explicit FiniteMPS(container_type sites) : sites_(std::move(sites)) { this->check_structure(); }

    [[nodiscard]] size_type size() const noexcept { return sites_.size(); }
    [[nodiscard]] bool empty() const noexcept { return sites_.empty(); }
    [[nodiscard]] MpsSiteTensor const& operator[](size_type index) const { return sites_.at(index); }
    [[nodiscard]] MpsSiteTensor& operator[](size_type index) { return sites_.at(index); }
    [[nodiscard]] auto begin() const { return sites_.begin(); }
    [[nodiscard]] auto end() const { return sites_.end(); }

    void check_structure() const
    {
      for (size_type i = 1; i < sites_.size(); ++i)
      {
        if (sites_[i - 1].right_bond_space() != sites_[i].left_bond_space())
        {
          throw std::invalid_argument("FiniteMPS adjacent bond spaces do not match");
        }
        if (sites_[i - 1].physical_space().symmetry() != sites_[i].physical_space().symmetry())
        {
          throw std::invalid_argument("FiniteMPS sites do not share one symmetry");
        }
      }
    }

  private:
    container_type sites_;
};

class TwoSiteWavefunction {
  public:
    TwoSiteWavefunction(std::size_t left_site, LocalSpace left_physical_space, LocalSpace right_physical_space,
                        BlockSpace left_bond_space, BlockSpace shared_bond_space, BlockSpace right_bond_space,
                        tensorcontraction::MatrixFamily values)
        : left_site_(left_site), left_physical_space_(std::move(left_physical_space)),
          right_physical_space_(std::move(right_physical_space)), left_bond_space_(std::move(left_bond_space)),
          shared_bond_space_(std::move(shared_bond_space)), right_bond_space_(std::move(right_bond_space)),
          values_(std::move(values))
    {}

    [[nodiscard]] std::size_t left_site() const noexcept { return left_site_; }
    [[nodiscard]] std::size_t right_site() const noexcept { return left_site_ + 1; }
    [[nodiscard]] LocalSpace const& left_physical_space() const noexcept { return left_physical_space_; }
    [[nodiscard]] LocalSpace const& right_physical_space() const noexcept { return right_physical_space_; }
    [[nodiscard]] BlockSpace const& left_bond_space() const noexcept { return left_bond_space_; }
    [[nodiscard]] BlockSpace const& shared_bond_space() const noexcept { return shared_bond_space_; }
    [[nodiscard]] BlockSpace const& right_bond_space() const noexcept { return right_bond_space_; }
    [[nodiscard]] tensorcontraction::MatrixFamily::Block block() const { return values_.block(0); }
    [[nodiscard]] std::span<double const> values() const { return values_.values(0); }
    [[nodiscard]] tensorcontraction::MatrixFamily const& matrix_family() const noexcept { return values_; }

    void assign_to(tensorcontraction::MatrixFamily& output) const { output.assign(values_); }

  private:
    std::size_t left_site_ = 0;
    LocalSpace left_physical_space_;
    LocalSpace right_physical_space_;
    BlockSpace left_bond_space_;
    BlockSpace shared_bond_space_;
    BlockSpace right_bond_space_;
    tensorcontraction::MatrixFamily values_;
};

inline auto make_two_site_wavefunction(FiniteMPS const& psi, std::size_t left_site) -> TwoSiteWavefunction
{
  if (left_site + 1 >= psi.size())
  {
    throw std::out_of_range("make_two_site_wavefunction requires two adjacent MPS sites");
  }

  auto const& left = psi[left_site];
  auto const& right = psi[left_site + 1];
  if (left.right_bond_space() != right.left_bond_space())
  {
    throw std::invalid_argument("make_two_site_wavefunction adjacent bond spaces do not match");
  }

  auto const row_dim = left.left_dim() * left.physical_dim();
  auto const col_dim = right.physical_dim() * right.right_dim();
  std::array blocks{tensorcontraction::MatrixFamily::Block{row_dim, col_dim}};
  tensorcontraction::MatrixFamily values(blocks);
  auto dst = values.values(0);

  for (std::size_t left_bond = 0; left_bond < left.left_dim(); ++left_bond)
  {
    for (std::size_t left_phys = 0; left_phys < left.physical_dim(); ++left_phys)
    {
      auto const left_values = left.values(left_phys);
      auto const row = left_bond * left.physical_dim() + left_phys;
      for (std::size_t right_phys = 0; right_phys < right.physical_dim(); ++right_phys)
      {
        auto const right_values = right.values(right_phys);
        for (std::size_t right_bond = 0; right_bond < right.right_dim(); ++right_bond)
        {
          auto const col = right_phys * right.right_dim() + right_bond;
          double value = 0.0;
          for (std::size_t shared = 0; shared < left.right_dim(); ++shared)
          {
            value += left_values[left_bond * left.right_dim() + shared] *
                     right_values[shared * right.right_dim() + right_bond];
          }
          dst[row * col_dim + col] = value;
        }
      }
    }
  }

  return TwoSiteWavefunction(left_site, left.physical_space(), right.physical_space(), left.left_bond_space(),
                             left.right_bond_space(), right.right_bond_space(), std::move(values));
}

} // namespace uni20
