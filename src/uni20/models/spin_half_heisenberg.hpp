/**
 * \file spin_half_heisenberg.hpp
 * \ingroup models
 * \brief Constructs U(1) spin-half product states and Heisenberg MPOs.
 */

#pragma once

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/u1.hpp>
#include <uni20/tensor_network/finite_chain.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uni20::models
{

/// \brief Physical state selected at the first site of a spin-half product pattern.
enum class SpinHalfState
{
  up,
  down
};

/// \brief U(1)-symmetric spin-half local space and its two state charges.
/// \details States are ordered as `|up>` then `|down>`, with charges `+1/2`
///          and `-1/2` under the selected U(1) component.
struct SpinHalfU1Site
{
    /// \brief U(1) symmetry context shared by every model object.
    Symmetry symmetry;
    /// \brief Physical state space in up-then-down order.
    LocalSpace space;
    /// \brief Quantum number of `|up>`.
    QNum up;
    /// \brief Quantum number of `|down>`.
    QNum down;

    /// \brief Return the physical-space coordinate of one local state.
    /// \param state Local spin state.
    /// \return Zero for up and one for down.
    static constexpr auto coordinate(SpinHalfState state) noexcept -> std::size_t
    {
      return state == SpinHalfState::up ? 0 : 1;
    }

    /// \brief Return the quantum number of one local state.
    /// \param state Local spin state.
    /// \return Stored up or down quantum number.
    auto qnum(SpinHalfState state) const -> QNum const& { return state == SpinHalfState::up ? up : down; }
};

/// \brief Finite MPS type produced by the spin-half model builders.
/// \tparam Scalar Real or complex tensor element type.
/// \tparam Storage Block storage policy used by every MPS site.
template <RealOrComplex Scalar, BlockTensorStorage Storage = PackedSparseBlockStorage<>>
using SpinHalfU1Mps = tensor_network::FiniteMps<Scalar, BlockSpace, LocalSpace, Storage>;

/// \brief Packed finite MPO type produced by the spin-half model builders.
template <RealOrComplex Scalar>
using SpinHalfU1Mpo = tensor_network::FiniteMpo<Scalar, LocalSpace, LocalSpace, PackedSparseBlockStorage<>>;

/// \brief Construct the standard U(1)-symmetric spin-half local space.
/// \param charge_name Name of the conserved U(1) component, normally `"Sz"`.
/// \return Local state space and charges in up-then-down order.
/// \throws std::invalid_argument If \p charge_name is empty.
inline auto make_spin_half_u1_site(std::string_view charge_name = "Sz") -> SpinHalfU1Site
{
  if (charge_name.empty()) throw std::invalid_argument("spin-half U(1) charge name must not be empty");

  Symmetry const symmetry(std::string(charge_name) + ":U(1)");
  QNum const up = make_qnum(symmetry, {{charge_name, U1{half_int{0.5}}}});
  QNum const down = make_qnum(symmetry, {{charge_name, U1{half_int{-0.5}}}});
  return {.symmetry = symmetry, .space = LocalSpace(symmetry, {up, down}, "spin-1/2"), .up = up, .down = down};
}

namespace detail
{

inline auto opposite(SpinHalfState state) noexcept -> SpinHalfState
{
  return state == SpinHalfState::up ? SpinHalfState::down : SpinHalfState::up;
}

inline auto mps_bond_label(std::size_t bond) -> std::string { return "neel-mps-bond-" + std::to_string(bond); }

inline auto mpo_bond_label(std::size_t bond) -> std::string { return "heisenberg-mpo-bond-" + std::to_string(bond); }

template <RealOrComplex Scalar> struct MpoCoefficient
{
    using site_type = typename SpinHalfU1Mpo<Scalar>::site_type;

    typename site_type::key_type key;
    Scalar value;
};

template <RealOrComplex Scalar>
void append_mpo_coefficient(std::vector<MpoCoefficient<Scalar>>& coefficients, std::size_t left, std::size_t ket,
                            std::size_t right, std::size_t bra, Scalar value)
{
  if (value == Scalar{}) return;
  coefficients.push_back(
      {.key = typename MpoCoefficient<Scalar>::site_type::key_type{{left, ket, right, bra}}, .value = value});
}

template <RealOrComplex Scalar>
void append_identity(std::vector<MpoCoefficient<Scalar>>& coefficients, std::size_t left, std::size_t right)
{
  append_mpo_coefficient(coefficients, left, 0, right, 0, Scalar{1});
  append_mpo_coefficient(coefficients, left, 1, right, 1, Scalar{1});
}

template <RealOrComplex Scalar>
void append_sz(std::vector<MpoCoefficient<Scalar>>& coefficients, std::size_t left, std::size_t right, Scalar scale)
{
  append_mpo_coefficient(coefficients, left, 0, right, 0, scale * Scalar{0.5});
  append_mpo_coefficient(coefficients, left, 1, right, 1, scale * Scalar{-0.5});
}

template <RealOrComplex Scalar>
void append_sp(std::vector<MpoCoefficient<Scalar>>& coefficients, std::size_t left, std::size_t right, Scalar scale)
{
  append_mpo_coefficient(coefficients, left, 1, right, 0, scale);
}

template <RealOrComplex Scalar>
void append_sm(std::vector<MpoCoefficient<Scalar>>& coefficients, std::size_t left, std::size_t right, Scalar scale)
{
  append_mpo_coefficient(coefficients, left, 0, right, 1, scale);
}

template <RealOrComplex Scalar>
auto make_mpo_site(SpinHalfU1Site const& local, LocalSpace const& left, LocalSpace const& right,
                   std::vector<MpoCoefficient<Scalar>> const& coefficients) -> typename SpinHalfU1Mpo<Scalar>::site_type
{
  using site_type = typename SpinHalfU1Mpo<Scalar>::site_type;
  std::vector<typename site_type::key_type> keys;
  keys.reserve(coefficients.size());
  for (auto const& coefficient : coefficients)
    keys.push_back(coefficient.key);

  site_type result(local.symmetry, Domain{left, local.space}, Codomain{right, local.space}, std::move(keys));
  for (auto const& coefficient : coefficients)
    result.block(coefficient.key)[] = coefficient.value;
  return result;
}

template <RealOrComplex Scalar>
auto first_heisenberg_site(SpinHalfU1Site const& local, LocalSpace const& left, LocalSpace const& right,
                           Scalar longitudinal_field) -> typename SpinHalfU1Mpo<Scalar>::site_type
{
  std::vector<MpoCoefficient<Scalar>> coefficients;
  append_identity(coefficients, 0, 0);
  append_sp(coefficients, 0, 1, Scalar{1});
  append_sm(coefficients, 0, 2, Scalar{1});
  append_sz(coefficients, 0, 3, Scalar{1});
  append_sz(coefficients, 0, 4, longitudinal_field);
  return make_mpo_site(local, left, right, coefficients);
}

template <RealOrComplex Scalar>
auto bulk_heisenberg_site(SpinHalfU1Site const& local, LocalSpace const& left, LocalSpace const& right, Scalar exchange,
                          Scalar longitudinal_field) -> typename SpinHalfU1Mpo<Scalar>::site_type
{
  std::vector<MpoCoefficient<Scalar>> coefficients;
  append_identity(coefficients, 0, 0);
  append_sp(coefficients, 0, 1, Scalar{1});
  append_sm(coefficients, 0, 2, Scalar{1});
  append_sz(coefficients, 0, 3, Scalar{1});
  append_sz(coefficients, 0, 4, longitudinal_field);
  append_sm(coefficients, 1, 4, exchange * Scalar{0.5});
  append_sp(coefficients, 2, 4, exchange * Scalar{0.5});
  append_sz(coefficients, 3, 4, exchange);
  append_identity(coefficients, 4, 4);
  return make_mpo_site(local, left, right, coefficients);
}

template <RealOrComplex Scalar>
auto last_heisenberg_site(SpinHalfU1Site const& local, LocalSpace const& left, LocalSpace const& right, Scalar exchange,
                          Scalar longitudinal_field) -> typename SpinHalfU1Mpo<Scalar>::site_type
{
  std::vector<MpoCoefficient<Scalar>> coefficients;
  append_sz(coefficients, 0, 0, longitudinal_field);
  append_sm(coefficients, 1, 0, exchange * Scalar{0.5});
  append_sp(coefficients, 2, 0, exchange * Scalar{0.5});
  append_sz(coefficients, 3, 0, exchange);
  append_identity(coefficients, 4, 0);
  return make_mpo_site(local, left, right, coefficients);
}

template <RealOrComplex Scalar>
auto single_heisenberg_site(SpinHalfU1Site const& local, LocalSpace const& left, LocalSpace const& right,
                            Scalar longitudinal_field) -> typename SpinHalfU1Mpo<Scalar>::site_type
{
  std::vector<MpoCoefficient<Scalar>> coefficients;
  append_sz(coefficients, 0, 0, longitudinal_field);
  return make_mpo_site(local, left, right, coefficients);
}

} // namespace detail

/// \brief Construct a normalized alternating spin-half product MPS.
/// \details Every bond contains one cumulative-charge sector of dimension one,
///          and every site contains one scalar block with value one. The
///          resulting product state is canonical from either direction.
/// \tparam Scalar Real or complex MPS element type.
/// \tparam Storage Block storage policy used by every MPS site.
/// \param length Positive number of sites.
/// \param local U(1) spin-half local space.
/// \param first State placed on site zero; later sites alternate.
/// \return Finite MPS in the corresponding total-charge sector.
/// \throws std::invalid_argument If \p length is zero.
template <RealOrComplex Scalar = double, BlockTensorStorage Storage = PackedSparseBlockStorage<>>
[[nodiscard]] auto make_neel_product_mps(std::size_t length, SpinHalfU1Site const& local,
                                         SpinHalfState first = SpinHalfState::up) -> SpinHalfU1Mps<Scalar, Storage>
{
  if (length == 0) throw std::invalid_argument("Neel product MPS requires at least one site");

  std::vector<BlockSpace> bonds;
  bonds.reserve(length + 1);
  QNum cumulative = QNum::identity(local.symmetry);
  bonds.emplace_back(local.symmetry, std::initializer_list<BlockSector>{{cumulative, 1}}, detail::mps_bond_label(0));
  for (std::size_t site = 0; site < length; ++site)
  {
    SpinHalfState const state = site % 2 == 0 ? first : detail::opposite(first);
    cumulative = cumulative + local.qnum(state);
    bonds.emplace_back(local.symmetry, std::initializer_list<BlockSector>{{cumulative, 1}},
                       detail::mps_bond_label(site + 1));
  }

  using mps_type = SpinHalfU1Mps<Scalar, Storage>;
  using site_type = typename mps_type::site_type;
  std::vector<site_type> sites;
  sites.reserve(length);
  for (std::size_t site = 0; site < length; ++site)
  {
    SpinHalfState const state = site % 2 == 0 ? first : detail::opposite(first);
    typename site_type::key_type const key{{0, SpinHalfU1Site::coordinate(state), 0}};
    site_type value(local.symmetry, Domain{bonds[site], local.space}, Codomain{bonds[site + 1]}, {key});
    value.block(key)[0, 0] = Scalar{1};
    sites.push_back(std::move(value));
  }
  return mps_type(std::move(sites));
}

/// \brief Construct the open spin-half Heisenberg Hamiltonian as a finite MPO.
/// \details The Hamiltonian is
///          `exchange * sum_i S_i.S_(i+1) + longitudinal_field * sum_i S_i^z`.
///          Interior auxiliaries use `[start, pending S-, pending S+, pending
///          Sz, finish]` with charges `[0,-1,+1,0,0]`. Both boundary
///          auxiliaries contain only the scalar charge, so boundary index zero
///          is the complete open-chain contraction.
/// \tparam Scalar Real or complex MPO element type.
/// \param length Positive number of sites.
/// \param local U(1) spin-half local space.
/// \param exchange Nearest-neighbor exchange coefficient.
/// \param longitudinal_field On-site longitudinal field coefficient.
/// \return Packed finite MPO with reduced one-state boundaries.
/// \throws std::invalid_argument If \p length is zero.
template <RealOrComplex Scalar = double>
[[nodiscard]] auto make_spin_half_heisenberg_mpo(std::size_t length, SpinHalfU1Site const& local,
                                                 make_real_t<Scalar> exchange = make_real_t<Scalar>{1},
                                                 make_real_t<Scalar> longitudinal_field = make_real_t<Scalar>{})
    -> SpinHalfU1Mpo<Scalar>
{
  if (length == 0) throw std::invalid_argument("spin-half Heisenberg MPO requires at least one site");

  QNum const scalar = QNum::identity(local.symmetry);
  std::string const& charge_name = local.symmetry.factors().front().name;
  QNum const minus_one = make_qnum(local.symmetry, {{charge_name, U1{-1}}});
  QNum const plus_one = make_qnum(local.symmetry, {{charge_name, U1{1}}});
  std::vector<LocalSpace> auxiliaries;
  auxiliaries.reserve(length + 1);
  auxiliaries.emplace_back(local.symmetry, std::initializer_list<QNum>{scalar}, detail::mpo_bond_label(0));
  for (std::size_t bond = 1; bond < length; ++bond)
  {
    auxiliaries.emplace_back(local.symmetry, std::initializer_list<QNum>{scalar, minus_one, plus_one, scalar, scalar},
                             detail::mpo_bond_label(bond));
  }
  auxiliaries.emplace_back(local.symmetry, std::initializer_list<QNum>{scalar}, detail::mpo_bond_label(length));

  using mpo_type = SpinHalfU1Mpo<Scalar>;
  using site_type = typename mpo_type::site_type;
  std::vector<site_type> sites;
  sites.reserve(length);
  if (length == 1)
  {
    sites.push_back(
        detail::single_heisenberg_site<Scalar>(local, auxiliaries[0], auxiliaries[1], Scalar{longitudinal_field}));
  }
  else
  {
    sites.push_back(
        detail::first_heisenberg_site<Scalar>(local, auxiliaries[0], auxiliaries[1], Scalar{longitudinal_field}));
    for (std::size_t site = 1; site + 1 < length; ++site)
    {
      sites.push_back(detail::bulk_heisenberg_site<Scalar>(local, auxiliaries[site], auxiliaries[site + 1],
                                                           Scalar{exchange}, Scalar{longitudinal_field}));
    }
    sites.push_back(detail::last_heisenberg_site<Scalar>(local, auxiliaries[length - 1], auxiliaries[length],
                                                         Scalar{exchange}, Scalar{longitudinal_field}));
  }
  return mpo_type(std::move(sites));
}

} // namespace uni20::models
