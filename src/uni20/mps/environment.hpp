/**
 * \file environment.hpp
 * \brief Minimal dense MPO environment construction for the first DMRG prototype.
 */

#pragma once

#include <uni20/mps/finite_mps.hpp>
#include <uni20/operator/finite_triangular_mpo.hpp>
#include <uni20/tensorcontraction/effective_hamiltonian_plan.hpp>

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20
{

class MpoEnvironment {
  public:
    MpoEnvironment(LocalSpace virtual_space, BlockSpace bond_space)
        : virtual_space_(std::move(virtual_space)), bond_space_(std::move(bond_space))
    {
      auto const block_size = this->bond_dim() * this->bond_dim();
      blocks_.resize(virtual_space_.size(), std::vector<double>(block_size, 0.0));
    }

    [[nodiscard]] LocalSpace const& virtual_space() const noexcept { return virtual_space_; }
    [[nodiscard]] BlockSpace const& bond_space() const noexcept { return bond_space_; }
    [[nodiscard]] std::size_t virtual_dim() const noexcept { return virtual_space_.size(); }
    [[nodiscard]] std::size_t bond_dim() const noexcept { return bond_space_.total_dim(); }

    [[nodiscard]] std::span<double> values(std::size_t virtual_index) { return blocks_.at(virtual_index); }
    [[nodiscard]] std::span<double const> values(std::size_t virtual_index) const { return blocks_.at(virtual_index); }

    void set_identity(std::size_t virtual_index)
    {
      auto values = this->values(virtual_index);
      std::fill(values.begin(), values.end(), 0.0);
      auto const dim = this->bond_dim();
      for (std::size_t i = 0; i < dim; ++i)
      {
        values[i * dim + i] = 1.0;
      }
    }

  private:
    LocalSpace virtual_space_;
    BlockSpace bond_space_;
    std::vector<std::vector<double>> blocks_;
};

inline void validate_site_component_spaces(MpsSiteTensor const& site, OperatorComponent const& component)
{
  if (component.local_bra_space() != site.physical_space() || component.local_ket_space() != site.physical_space())
  {
    throw std::invalid_argument("MPO component local spaces do not match the MPS site physical space");
  }
}

inline void validate_environment_site_pair(MpoEnvironment const& env, MpsSiteTensor const& site,
                                           BlockSpace const& expected_bond, LocalSpace const& expected_virtual,
                                           char const* side)
{
  if (env.bond_space() != expected_bond)
  {
    throw std::invalid_argument(std::string("MPO ") + side + " environment bond space does not match the MPS site");
  }
  if (env.virtual_space() != expected_virtual)
  {
    throw std::invalid_argument(std::string("MPO ") + side +
                                " environment virtual space does not match the MPO component");
  }
  if (env.bond_dim() == 0)
  {
    throw std::invalid_argument("MPO environment requires a non-empty bond space");
  }
  if (site.physical_dim() == 0)
  {
    throw std::invalid_argument("MPO environment update requires a non-empty physical space");
  }
}

namespace detail
{

inline bool use_host_environment_backend()
{
  auto const* environment_backend = std::getenv("UNI20_MPS_ENVIRONMENT_BACKEND");
  if (environment_backend != nullptr && std::string(environment_backend) == "host")
  {
    return true;
  }

  auto const* backend = std::getenv("UNI20_TENSORCONTRACTION_BACKEND");
  if (backend != nullptr && (std::string(backend) == "host" || std::string(backend) == "cpu"))
  {
    return true;
  }

  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (environment_backend != nullptr && std::string(environment_backend) == "tensorcontraction" && mpi_initialized == 0)
  {
    throw std::runtime_error("TensorContraction environment backend requires MPI to be initialized");
  }
  return mpi_initialized == 0;
}

inline auto repeated_blocks(std::size_t count, std::size_t rows,
                            std::size_t cols) -> std::vector<tensorcontraction::MatrixFamily::Block>
{
  return std::vector<tensorcontraction::MatrixFamily::Block>(count, tensorcontraction::MatrixFamily::Block{rows, cols});
}

inline void assign_transposed_site_block(tensorcontraction::MatrixFamily& family, std::size_t index,
                                         std::span<double const> values, std::size_t rows, std::size_t cols)
{
  auto target = family.values(index);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      target[col * rows + row] = values[row * cols + col];
    }
  }
}

inline auto left_environment_host(MpoEnvironment const& left_env, MpsSiteTensor const& site,
                                  OperatorComponent const& component) -> MpoEnvironment
{
  validate_site_component_spaces(site, component);
  validate_environment_site_pair(left_env, site, site.left_bond_space(), component.left_virtual_space(), "left");

  MpoEnvironment next(component.right_virtual_space(), site.right_bond_space());
  auto const left_bond_dim = site.left_dim();
  auto const right_bond_dim = site.right_dim();

  for (std::size_t mpo_left = 0; mpo_left < component.rows(); ++mpo_left)
  {
    auto const left_values = left_env.values(mpo_left);
    std::vector<double> left_times_ket(left_bond_dim * right_bond_dim, 0.0);
    for (auto const& mpo_entry : component.data().row(mpo_left))
    {
      auto const mpo_right = mpo_entry.column;
      auto next_values = next.values(mpo_right);
      auto const& local_op = mpo_entry.value;
      for (std::size_t bra_phys = 0; bra_phys < local_op.rows(); ++bra_phys)
      {
        auto const bra_site = site.values(bra_phys);
        for (auto const& op_entry : local_op.coefficients().row(bra_phys))
        {
          auto const ket_phys = op_entry.column;
          auto const ket_site = site.values(ket_phys);
          auto const coefficient = op_entry.value;
          if (coefficient == 0.0)
          {
            continue;
          }
          std::fill(left_times_ket.begin(), left_times_ket.end(), 0.0);
          for (std::size_t left_bra = 0; left_bra < left_bond_dim; ++left_bra)
          {
            for (std::size_t left_ket = 0; left_ket < left_bond_dim; ++left_ket)
            {
              double const env_value = left_values[left_bra * left_bond_dim + left_ket];
              if (env_value == 0.0)
              {
                continue;
              }
              double const weighted_env = env_value * coefficient;
              for (std::size_t right_ket = 0; right_ket < right_bond_dim; ++right_ket)
              {
                left_times_ket[left_bra * right_bond_dim + right_ket] +=
                    weighted_env * ket_site[left_ket * right_bond_dim + right_ket];
              }
            }
            for (std::size_t right_bra = 0; right_bra < right_bond_dim; ++right_bra)
            {
              double const bra_value = bra_site[left_bra * right_bond_dim + right_bra];
              if (bra_value == 0.0)
              {
                continue;
              }
              for (std::size_t right_ket = 0; right_ket < right_bond_dim; ++right_ket)
              {
                next_values[right_bra * right_bond_dim + right_ket] +=
                    bra_value * left_times_ket[left_bra * right_bond_dim + right_ket];
              }
            }
          }
        }
      }
    }
  }

  return next;
}

inline auto right_environment_host(MpoEnvironment const& right_env, MpsSiteTensor const& site,
                                   OperatorComponent const& component) -> MpoEnvironment
{
  validate_site_component_spaces(site, component);
  validate_environment_site_pair(right_env, site, site.right_bond_space(), component.right_virtual_space(), "right");

  MpoEnvironment previous(component.left_virtual_space(), site.left_bond_space());
  auto const left_bond_dim = site.left_dim();
  auto const right_bond_dim = site.right_dim();

  for (std::size_t mpo_left = 0; mpo_left < component.rows(); ++mpo_left)
  {
    auto previous_values = previous.values(mpo_left);
    for (auto const& mpo_entry : component.data().row(mpo_left))
    {
      auto const mpo_right = mpo_entry.column;
      auto const right_values = right_env.values(mpo_right);
      auto const& local_op = mpo_entry.value;
      std::vector<double> bra_times_right(left_bond_dim * right_bond_dim, 0.0);
      for (std::size_t bra_phys = 0; bra_phys < local_op.rows(); ++bra_phys)
      {
        auto const bra_site = site.values(bra_phys);
        for (auto const& op_entry : local_op.coefficients().row(bra_phys))
        {
          auto const ket_phys = op_entry.column;
          auto const ket_site = site.values(ket_phys);
          auto const coefficient = op_entry.value;
          if (coefficient == 0.0)
          {
            continue;
          }
          std::fill(bra_times_right.begin(), bra_times_right.end(), 0.0);
          for (std::size_t left_bra = 0; left_bra < left_bond_dim; ++left_bra)
          {
            for (std::size_t right_bra = 0; right_bra < right_bond_dim; ++right_bra)
            {
              double const bra_value = bra_site[left_bra * right_bond_dim + right_bra];
              if (bra_value == 0.0)
              {
                continue;
              }
              for (std::size_t right_ket = 0; right_ket < right_bond_dim; ++right_ket)
              {
                bra_times_right[left_bra * right_bond_dim + right_ket] +=
                    bra_value * right_values[right_bra * right_bond_dim + right_ket];
              }
            }
            for (std::size_t left_ket = 0; left_ket < left_bond_dim; ++left_ket)
            {
              double value = 0.0;
              for (std::size_t right_ket = 0; right_ket < right_bond_dim; ++right_ket)
              {
                value += bra_times_right[left_bra * right_bond_dim + right_ket] *
                         ket_site[left_ket * right_bond_dim + right_ket];
              }
              previous_values[left_bra * left_bond_dim + left_ket] += coefficient * value;
            }
          }
        }
      }
    }
  }

  return previous;
}

inline auto left_environment_tensorcontraction(MpoEnvironment const& left_env, MpsSiteTensor const& site,
                                               OperatorComponent const& component) -> MpoEnvironment
{
  validate_site_component_spaces(site, component);
  validate_environment_site_pair(left_env, site, site.left_bond_space(), component.left_virtual_space(), "left");

  auto const left_bond_dim = site.left_dim();
  auto const right_bond_dim = site.right_dim();
  MpoEnvironment next(component.right_virtual_space(), site.right_bond_space());

  tensorcontraction::MatrixFamily r_mats(repeated_blocks(next.virtual_dim(), right_bond_dim, right_bond_dim));
  tensorcontraction::MatrixFamily a_mats(
      repeated_blocks(site.physical_dim(), right_bond_dim, left_bond_dim)); // site tensor transposes
  tensorcontraction::MatrixFamily b_mats(
      repeated_blocks(left_env.virtual_dim(), left_bond_dim, left_bond_dim)); // previous environment
  tensorcontraction::MatrixFamily c_mats(repeated_blocks(site.physical_dim(), left_bond_dim, right_bond_dim));

  for (std::size_t phys = 0; phys < site.physical_dim(); ++phys)
  {
    assign_transposed_site_block(a_mats, phys, site.values(phys), left_bond_dim, right_bond_dim);
    c_mats.assign(phys, site.values(phys));
  }
  for (std::size_t virtual_index = 0; virtual_index < left_env.virtual_dim(); ++virtual_index)
  {
    b_mats.assign(virtual_index, left_env.values(virtual_index));
  }

  std::vector<tensorcontraction::EffectiveHamiltonianPlan::Term> terms;
  for (std::size_t mpo_left = 0; mpo_left < component.rows(); ++mpo_left)
  {
    for (auto const& mpo_entry : component.data().row(mpo_left))
    {
      auto const mpo_right = mpo_entry.column;
      auto const& local_op = mpo_entry.value;
      for (std::size_t bra_phys = 0; bra_phys < local_op.rows(); ++bra_phys)
      {
        for (auto const& op_entry : local_op.coefficients().row(bra_phys))
        {
          terms.push_back(tensorcontraction::EffectiveHamiltonianPlan::Term{
              .r = mpo_right, .a = bra_phys, .b = mpo_left, .c = op_entry.column, .coefficient = op_entry.value});
        }
      }
    }
  }

  tensorcontraction::EffectiveHamiltonianPlan plan(std::move(r_mats), std::move(a_mats), std::move(b_mats),
                                                   std::move(c_mats), terms);
  plan.apply();
  for (std::size_t virtual_index = 0; virtual_index < next.virtual_dim(); ++virtual_index)
  {
    auto values = next.values(virtual_index);
    auto result = plan.r_values(virtual_index);
    std::copy(result.begin(), result.end(), values.begin());
  }
  return next;
}

inline auto right_environment_tensorcontraction(MpoEnvironment const& right_env, MpsSiteTensor const& site,
                                                OperatorComponent const& component) -> MpoEnvironment
{
  validate_site_component_spaces(site, component);
  validate_environment_site_pair(right_env, site, site.right_bond_space(), component.right_virtual_space(), "right");

  auto const left_bond_dim = site.left_dim();
  auto const right_bond_dim = site.right_dim();
  MpoEnvironment previous(component.left_virtual_space(), site.left_bond_space());

  tensorcontraction::MatrixFamily r_mats(repeated_blocks(previous.virtual_dim(), left_bond_dim, left_bond_dim));
  tensorcontraction::MatrixFamily a_mats(repeated_blocks(site.physical_dim(), left_bond_dim, right_bond_dim));
  tensorcontraction::MatrixFamily b_mats(repeated_blocks(right_env.virtual_dim(), right_bond_dim, right_bond_dim));
  tensorcontraction::MatrixFamily c_mats(
      repeated_blocks(site.physical_dim(), right_bond_dim, left_bond_dim)); // site tensor transposes

  for (std::size_t phys = 0; phys < site.physical_dim(); ++phys)
  {
    a_mats.assign(phys, site.values(phys));
    assign_transposed_site_block(c_mats, phys, site.values(phys), left_bond_dim, right_bond_dim);
  }
  for (std::size_t virtual_index = 0; virtual_index < right_env.virtual_dim(); ++virtual_index)
  {
    b_mats.assign(virtual_index, right_env.values(virtual_index));
  }

  std::vector<tensorcontraction::EffectiveHamiltonianPlan::Term> terms;
  for (std::size_t mpo_left = 0; mpo_left < component.rows(); ++mpo_left)
  {
    for (auto const& mpo_entry : component.data().row(mpo_left))
    {
      auto const mpo_right = mpo_entry.column;
      auto const& local_op = mpo_entry.value;
      for (std::size_t bra_phys = 0; bra_phys < local_op.rows(); ++bra_phys)
      {
        for (auto const& op_entry : local_op.coefficients().row(bra_phys))
        {
          terms.push_back(tensorcontraction::EffectiveHamiltonianPlan::Term{
              .r = mpo_left, .a = bra_phys, .b = mpo_right, .c = op_entry.column, .coefficient = op_entry.value});
        }
      }
    }
  }

  tensorcontraction::EffectiveHamiltonianPlan plan(std::move(r_mats), std::move(a_mats), std::move(b_mats),
                                                   std::move(c_mats), terms);
  plan.apply();
  for (std::size_t virtual_index = 0; virtual_index < previous.virtual_dim(); ++virtual_index)
  {
    auto values = previous.values(virtual_index);
    auto result = plan.r_values(virtual_index);
    std::copy(result.begin(), result.end(), values.begin());
  }
  return previous;
}

} // namespace detail

inline auto extend_left_environment(MpoEnvironment const& left_env, MpsSiteTensor const& site,
                                    OperatorComponent const& component) -> MpoEnvironment
{
  if (detail::use_host_environment_backend())
  {
    return detail::left_environment_host(left_env, site, component);
  }
  return detail::left_environment_tensorcontraction(left_env, site, component);
}

inline auto extend_right_environment(MpoEnvironment const& right_env, MpsSiteTensor const& site,
                                     OperatorComponent const& component) -> MpoEnvironment
{
  if (detail::use_host_environment_backend())
  {
    return detail::right_environment_host(right_env, site, component);
  }
  return detail::right_environment_tensorcontraction(right_env, site, component);
}

inline auto make_left_boundary_environment(FiniteMPS const& psi, FiniteTriangularMPO const& mpo) -> MpoEnvironment
{
  if (psi.empty() || mpo.empty() || psi.size() != mpo.size())
  {
    throw std::invalid_argument("left boundary environment requires non-empty MPS and MPO chains of equal length");
  }
  MpoEnvironment env(mpo.left_boundary_virtual_space(), psi[0].left_bond_space());
  env.set_identity(0);
  return env;
}

inline auto make_right_boundary_environment(FiniteMPS const& psi, FiniteTriangularMPO const& mpo) -> MpoEnvironment
{
  if (psi.empty() || mpo.empty() || psi.size() != mpo.size())
  {
    throw std::invalid_argument("right boundary environment requires non-empty MPS and MPO chains of equal length");
  }
  MpoEnvironment env(mpo.right_boundary_virtual_space(), psi[psi.size() - 1].right_bond_space());
  env.set_identity(env.virtual_dim() - 1);
  return env;
}

inline auto build_left_environments(FiniteMPS const& psi, FiniteTriangularMPO const& mpo) -> std::vector<MpoEnvironment>
{
  std::vector<MpoEnvironment> environments;
  environments.reserve(psi.size() + 1);
  environments.push_back(make_left_boundary_environment(psi, mpo));
  for (std::size_t site = 0; site < psi.size(); ++site)
  {
    environments.push_back(extend_left_environment(environments.back(), psi[site], mpo[site]));
  }
  return environments;
}

inline auto build_right_environments(FiniteMPS const& psi,
                                     FiniteTriangularMPO const& mpo) -> std::vector<MpoEnvironment>
{
  std::vector<MpoEnvironment> reversed;
  reversed.reserve(psi.size() + 1);
  reversed.push_back(make_right_boundary_environment(psi, mpo));
  for (std::size_t offset = 0; offset < psi.size(); ++offset)
  {
    auto const site = psi.size() - 1 - offset;
    reversed.push_back(extend_right_environment(reversed.back(), psi[site], mpo[site]));
  }
  std::reverse(reversed.begin(), reversed.end());
  return reversed;
}

} // namespace uni20
