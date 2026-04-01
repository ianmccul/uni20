/**
 * \file heisenberg.hpp
 * \brief Defines the first finite triangular MPO builders for the spin-1/2 Heisenberg chain.
 * \details See `docs/models.md` for the current model-layer scope.
 */

#pragma once

#include <uni20/models/spin_half.hpp>
#include <uni20/operator/finite_triangular_mpo.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Construct the bulk virtual space used by the first spin-1/2 Heisenberg MPO.
/// \details The channel order is `[0, -1, +1, 0, 0]`, representing the start channel,
///          the pending `S^-` channel, the pending `S^+` channel, the pending `S^z`
///          channel, and the finish channel.
/// \param site Spin-1/2 site bundle whose symmetry labels define the channels.
/// \return Virtual `LocalSpace` for the bulk Heisenberg MPO component.
inline auto make_spin_half_heisenberg_virtual_space(SpinHalfSite const& site) -> LocalSpace
{
    return LocalSpace(site.symmetry,
                      {QNum::identity(site.symmetry), site.sm.transforms_as(), site.sp.transforms_as(),
                       QNum::identity(site.symmetry), QNum::identity(site.symmetry)});
}

/// \brief Construct the repeated bulk component for the first spin-1/2 Heisenberg MPO.
/// \details The current first-pass builder keeps the same bulk virtual space at the
///          boundaries rather than reducing bond dimensions there.
/// \param site Spin-1/2 site bundle providing the local operators.
/// \param j Heisenberg exchange coupling.
/// \param hz Optional longitudinal field coefficient multiplying `S^z`.
/// \return Upper-triangular MPO site component.
inline auto make_spin_half_heisenberg_bulk_component(SpinHalfSite const& site, double j = 1.0, double hz = 0.0)
    -> OperatorComponent
{
    LocalSpace const virtual_space = make_spin_half_heisenberg_virtual_space(site);
    OperatorComponent component(site.space, site.space, virtual_space, virtual_space);

    component.insert_or_assign(0, 0, site.identity);
    component.insert_or_assign(0, 1, site.sp);
    component.insert_or_assign(0, 2, site.sm);
    component.insert_or_assign(0, 3, site.sz);
    if (hz != 0.0)
    {
        component.insert_or_assign(0, 4, detail::scale_local_operator(site.sz, hz));
    }
    component.insert_or_assign(1, 4, detail::scale_local_operator(site.sm, 0.5 * j));
    component.insert_or_assign(2, 4, detail::scale_local_operator(site.sp, 0.5 * j));
    component.insert_or_assign(3, 4, detail::scale_local_operator(site.sz, j));
    component.insert_or_assign(4, 4, site.identity);

    return component;
}

/// \brief Construct a uniform finite triangular MPO for the first spin-1/2 Heisenberg chain.
/// \details The current builder simply repeats one bulk component at every site.
///          Boundary-spanning channels remain present at the ends; reduced boundary
///          dimensions are deferred.
/// \param length Number of lattice sites.
/// \param site Spin-1/2 site bundle providing the local operators.
/// \param j Heisenberg exchange coupling.
/// \param hz Optional longitudinal field coefficient multiplying `S^z`.
/// \return Finite triangular MPO with `length` identical bulk components.
inline auto make_spin_half_heisenberg_mpo(std::size_t length, SpinHalfSite const& site, double j = 1.0,
                                          double hz = 0.0) -> FiniteTriangularMPO
{
    std::vector<OperatorComponent> components;
    components.reserve(length);

    OperatorComponent const bulk = make_spin_half_heisenberg_bulk_component(site, j, hz);
    for (std::size_t i = 0; i < length; ++i)
    {
        components.push_back(bulk);
    }

    return FiniteTriangularMPO(std::move(components));
}

} // namespace uni20
