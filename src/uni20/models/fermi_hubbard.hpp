/**
 * \file fermi_hubbard.hpp
 * \brief Defines U(1)xU(1)-symmetric Fermi-Hubbard local-site and MPO helpers.
 * \details See `docs/models.md` for the current model-layer scope.
 */

#pragma once

#include <uni20/operator/finite_triangular_mpo.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Bundle of U(1)xU(1)-symmetric spinful fermion local data.
/// \details The default symmetry is `N:U(1),Sz:U(1)`, matching the Matrix
///          Product Toolkit Hubbard convention. Local states are ordered as
///          `|0>`, `|up down>`, `|down>`, `|up>`.
struct FermiHubbardSite
{
    Symmetry symmetry;
    LocalSpace space;
    QNum empty;
    QNum doubly_occupied;
    QNum down;
    QNum up;
    LocalOperator identity;
    LocalOperator parity;
    LocalOperator n;
    LocalOperator n_up;
    LocalOperator n_down;
    LocalOperator sz;
    LocalOperator p_double;
    LocalOperator hu;
    LocalOperator ch_up;
    LocalOperator c_up;
    LocalOperator ch_down;
    LocalOperator c_down;
    LocalOperator ch_up_parity;
    LocalOperator c_up_parity;
    LocalOperator ch_down_parity;
    LocalOperator c_down_parity;
};

namespace detail
{

/// \brief Return a scaled copy of a Hubbard local operator.
/// \param op Source local operator.
/// \param factor Scalar multiplier.
/// \return Sparse local operator with every coefficient multiplied by `factor`.
inline auto scale_hubbard_local_operator(LocalOperator const& op, double factor) -> LocalOperator
{
  LocalOperator result(op.bra_space(), op.ket_space(), op.transforms_as());
  for (LocalOperator::index_type row = 0; row < op.rows(); ++row)
  {
    for (auto const& entry : op.coefficients().row(row))
    {
      result.insert_or_assign(row, entry.column, factor * entry.value);
    }
  }
  return result;
}

/// \brief Right-multiply a local operator by a diagonal scalar operator.
/// \details This is used to build the nearest-neighbor Jordan-Wigner convention
///          `O_i P_i` explicitly for adjacent fermion hopping terms.
/// \param op Source local operator.
/// \param diagonal Diagonal coefficients indexed by ket-state coordinate.
/// \return Sparse local operator representing `op * diagonal`.
inline auto right_multiply_hubbard_local_operator_by_diagonal(LocalOperator const& op,
                                                              std::span<double const> diagonal) -> LocalOperator
{
  if (diagonal.size() != op.cols())
  {
    throw std::invalid_argument("diagonal size does not match local operator ket space");
  }

  LocalOperator result(op.bra_space(), op.ket_space(), op.transforms_as());
  for (LocalOperator::index_type row = 0; row < op.rows(); ++row)
  {
    for (auto const& entry : op.coefficients().row(row))
    {
      double const value = entry.value * diagonal[entry.column];
      if (value != 0.0)
      {
        result.insert_or_assign(row, entry.column, value);
      }
    }
  }
  return result;
}

} // namespace detail

/// \brief Construct the standard U(1)xU(1)-symmetric spinful fermion site bundle.
/// \details The default charges and operator signs match `models/fermion-u1u1.h`
///          in the Matrix Product Toolkit. The two symmetry factors are total
///          particle number and spin `S^z`.
/// \param particle_name Name of the particle-number U(1) component.
/// \param spin_name Name of the spin-projection U(1) component.
/// \return Hubbard local space together with symmetry-pure local operators.
inline auto make_fermi_hubbard_u1u1_site(std::string_view particle_name = "N",
                                         std::string_view spin_name = "Sz") -> FermiHubbardSite
{
  auto const spec = std::string{particle_name} + ":U(1)," + std::string{spin_name} + ":U(1)";
  Symmetry const symmetry{spec};

  QNum const empty = make_qnum(symmetry, {{particle_name, U1{0}}, {spin_name, U1{0}}});
  QNum const doubly_occupied = make_qnum(symmetry, {{particle_name, U1{2}}, {spin_name, U1{0}}});
  QNum const down = make_qnum(symmetry, {{particle_name, U1{1}}, {spin_name, U1{-0.5}}});
  QNum const up = make_qnum(symmetry, {{particle_name, U1{1}}, {spin_name, U1{0.5}}});

  QNum const scalar = QNum::identity(symmetry);
  QNum const create_up = make_qnum(symmetry, {{particle_name, U1{1}}, {spin_name, U1{0.5}}});
  QNum const annihilate_up = make_qnum(symmetry, {{particle_name, U1{-1}}, {spin_name, U1{-0.5}}});
  QNum const create_down = make_qnum(symmetry, {{particle_name, U1{1}}, {spin_name, U1{-0.5}}});
  QNum const annihilate_down = make_qnum(symmetry, {{particle_name, U1{-1}}, {spin_name, U1{0.5}}});

  LocalSpace const space(symmetry, {empty, doubly_occupied, down, up});

  LocalOperator identity(space, space, scalar);
  identity.insert_or_assign(0, 0, 1.0);
  identity.insert_or_assign(1, 1, 1.0);
  identity.insert_or_assign(2, 2, 1.0);
  identity.insert_or_assign(3, 3, 1.0);

  LocalOperator parity(space, space, scalar);
  parity.insert_or_assign(0, 0, 1.0);
  parity.insert_or_assign(1, 1, 1.0);
  parity.insert_or_assign(2, 2, -1.0);
  parity.insert_or_assign(3, 3, -1.0);

  LocalOperator n(space, space, scalar);
  n.insert_or_assign(1, 1, 2.0);
  n.insert_or_assign(2, 2, 1.0);
  n.insert_or_assign(3, 3, 1.0);

  LocalOperator n_up(space, space, scalar);
  n_up.insert_or_assign(1, 1, 1.0);
  n_up.insert_or_assign(3, 3, 1.0);

  LocalOperator n_down(space, space, scalar);
  n_down.insert_or_assign(1, 1, 1.0);
  n_down.insert_or_assign(2, 2, 1.0);

  LocalOperator sz(space, space, scalar);
  sz.insert_or_assign(2, 2, -0.5);
  sz.insert_or_assign(3, 3, 0.5);

  LocalOperator p_double(space, space, scalar);
  p_double.insert_or_assign(1, 1, 1.0);

  LocalOperator hu(space, space, scalar);
  hu.insert_or_assign(0, 0, 0.25);
  hu.insert_or_assign(1, 1, 0.25);
  hu.insert_or_assign(2, 2, -0.25);
  hu.insert_or_assign(3, 3, -0.25);

  LocalOperator ch_up(space, space, create_up);
  ch_up.insert_or_assign(3, 0, 1.0);
  ch_up.insert_or_assign(1, 2, 1.0);

  LocalOperator c_up(space, space, annihilate_up);
  c_up.insert_or_assign(0, 3, 1.0);
  c_up.insert_or_assign(2, 1, 1.0);

  LocalOperator ch_down(space, space, create_down);
  ch_down.insert_or_assign(2, 0, 1.0);
  ch_down.insert_or_assign(1, 3, -1.0);

  LocalOperator c_down(space, space, annihilate_down);
  c_down.insert_or_assign(0, 2, 1.0);
  c_down.insert_or_assign(3, 1, -1.0);

  std::array<double, 4> const parity_diagonal{1.0, 1.0, -1.0, -1.0};
  auto ch_up_parity = detail::right_multiply_hubbard_local_operator_by_diagonal(ch_up, parity_diagonal);
  auto c_up_parity = detail::right_multiply_hubbard_local_operator_by_diagonal(c_up, parity_diagonal);
  auto ch_down_parity = detail::right_multiply_hubbard_local_operator_by_diagonal(ch_down, parity_diagonal);
  auto c_down_parity = detail::right_multiply_hubbard_local_operator_by_diagonal(c_down, parity_diagonal);

  return FermiHubbardSite{
      .symmetry = symmetry,
      .space = space,
      .empty = empty,
      .doubly_occupied = doubly_occupied,
      .down = down,
      .up = up,
      .identity = std::move(identity),
      .parity = std::move(parity),
      .n = std::move(n),
      .n_up = std::move(n_up),
      .n_down = std::move(n_down),
      .sz = std::move(sz),
      .p_double = std::move(p_double),
      .hu = std::move(hu),
      .ch_up = std::move(ch_up),
      .c_up = std::move(c_up),
      .ch_down = std::move(ch_down),
      .c_down = std::move(c_down),
      .ch_up_parity = std::move(ch_up_parity),
      .c_up_parity = std::move(c_up_parity),
      .ch_down_parity = std::move(ch_down_parity),
      .c_down_parity = std::move(c_down_parity),
  };
}

/// \brief Construct the bulk virtual space for the nearest-neighbor Hubbard MPO.
/// \details The channel order is `[0, Cup, CHup, Cdown, CHdown, 0]`.
/// \param site Spinful fermion site bundle whose symmetry labels define the channels.
/// \return Virtual `LocalSpace` for the bulk Hubbard MPO component.
inline auto make_fermi_hubbard_virtual_space(FermiHubbardSite const& site) -> LocalSpace
{
  return LocalSpace(site.symmetry,
                    {QNum::identity(site.symmetry), site.c_up.transforms_as(), site.ch_up.transforms_as(),
                     site.c_down.transforms_as(), site.ch_down.transforms_as(), QNum::identity(site.symmetry)});
}

/// \brief Construct the repeated bulk component for the nearest-neighbor Hubbard MPO.
/// \details The hopping channels encode the adjacent-site Jordan-Wigner sign by
///          storing `O_i P_i` on the left site and the complementary fermion
///          operator on the right site. This matches the Matrix Product Toolkit
///          convention `-dot(CH(0), C(1)) + dot(C(0), CH(1))`.
/// \param site Spinful fermion site bundle providing the local operators.
/// \param hopping Nearest-neighbor hopping coefficient `t`.
/// \param onsite_u On-site interaction coefficient multiplying `n_up n_down`.
/// \return Upper-triangular MPO site component.
inline auto make_fermi_hubbard_bulk_component(FermiHubbardSite const& site, double hopping = 1.0,
                                              double onsite_u = 0.0) -> OperatorComponent
{
  LocalSpace const virtual_space = make_fermi_hubbard_virtual_space(site);
  OperatorComponent component(site.space, site.space, virtual_space, virtual_space);

  component.insert_or_assign(0, 0, site.identity);
  component.insert_or_assign(0, 1, detail::scale_hubbard_local_operator(site.ch_up_parity, -hopping));
  component.insert_or_assign(0, 2, detail::scale_hubbard_local_operator(site.c_up_parity, hopping));
  component.insert_or_assign(0, 3, detail::scale_hubbard_local_operator(site.ch_down_parity, -hopping));
  component.insert_or_assign(0, 4, detail::scale_hubbard_local_operator(site.c_down_parity, hopping));
  if (onsite_u != 0.0)
  {
    component.insert_or_assign(0, 5, detail::scale_hubbard_local_operator(site.p_double, onsite_u));
  }
  component.insert_or_assign(1, 5, site.c_up);
  component.insert_or_assign(2, 5, site.ch_up);
  component.insert_or_assign(3, 5, site.c_down);
  component.insert_or_assign(4, 5, site.ch_down);
  component.insert_or_assign(5, 5, site.identity);

  return component;
}

/// \brief Construct a uniform finite triangular MPO for the nearest-neighbor Hubbard chain.
/// \details The current builder repeats one bulk component at every site and keeps
///          the bulk virtual space at the boundaries, matching the first
///          Heisenberg MPO convention.
/// \param length Number of lattice sites.
/// \param site Spinful fermion site bundle providing the local operators.
/// \param hopping Nearest-neighbor hopping coefficient `t`.
/// \param onsite_u On-site interaction coefficient multiplying `n_up n_down`.
/// \return Finite triangular MPO with `length` identical bulk components.
inline auto make_fermi_hubbard_mpo(std::size_t length, FermiHubbardSite const& site, double hopping = 1.0,
                                   double onsite_u = 0.0) -> FiniteTriangularMPO
{
  std::vector<OperatorComponent> components;
  components.reserve(length);

  OperatorComponent const bulk = make_fermi_hubbard_bulk_component(site, hopping, onsite_u);
  for (std::size_t i = 0; i < length; ++i)
  {
    components.push_back(bulk);
  }

  return FiniteTriangularMPO(std::move(components));
}

} // namespace uni20
