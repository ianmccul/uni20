/**
 * \file spin_half.hpp
 * \brief Defines the first concrete U(1)-symmetric spin-1/2 local model helpers.
 * \details See `docs/models.md` for the current model-layer scope.
 */

#pragma once

#include <uni20/operator/local_operator.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace uni20
{

/// \brief Bundle of U(1)-symmetric spin-1/2 local data and standard symmetry-pure operators.
/// \details The local states are ordered as `|up>` then `|down>`, with U(1) charges
///          `+1/2` and `-1/2` under the named symmetry component.
struct SpinHalfSite
{
    Symmetry symmetry;
    LocalSpace space;
    QNum up;
    QNum down;
    LocalOperator identity;
    LocalOperator sz;
    LocalOperator sp;
    LocalOperator sm;
    LocalOperator sigma_z;
};

namespace detail
{

/// \brief Return a scaled copy of a local operator.
/// \param op Source local operator.
/// \param factor Scalar multiplier.
/// \return Sparse local operator with every coefficient multiplied by `factor`.
inline auto scale_local_operator(LocalOperator const& op, double factor) -> LocalOperator
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

} // namespace detail

/// \brief Construct the standard U(1)-symmetric spin-1/2 site bundle.
/// \param charge_name Name of the U(1) symmetry component, usually `"Sz"`.
/// \return Spin-1/2 local space together with standard symmetry-pure local operators.
inline auto make_spin_half_u1_site(std::string_view charge_name = "Sz") -> SpinHalfSite
{
    auto const spec = std::string{charge_name} + ":U(1)";
    Symmetry const symmetry{spec};

    QNum const up = make_qnum(symmetry, {{charge_name, U1{0.5}}});
    QNum const down = make_qnum(symmetry, {{charge_name, U1{-0.5}}});
    QNum const scalar = QNum::identity(symmetry);
    QNum const plus_one = make_qnum(symmetry, {{charge_name, U1{1}}});
    QNum const minus_one = make_qnum(symmetry, {{charge_name, U1{-1}}});

    LocalSpace const space(symmetry, {up, down});

    LocalOperator identity(space, space, scalar);
    identity.insert_or_assign(0, 0, 1.0);
    identity.insert_or_assign(1, 1, 1.0);

    LocalOperator sz(space, space, scalar);
    sz.insert_or_assign(0, 0, 0.5);
    sz.insert_or_assign(1, 1, -0.5);

    LocalOperator sp(space, space, plus_one);
    sp.insert_or_assign(0, 1, 1.0);

    LocalOperator sm(space, space, minus_one);
    sm.insert_or_assign(1, 0, 1.0);

    LocalOperator sigma_z(space, space, scalar);
    sigma_z.insert_or_assign(0, 0, 1.0);
    sigma_z.insert_or_assign(1, 1, -1.0);

    return SpinHalfSite{
        .symmetry = symmetry,
        .space = space,
        .up = up,
        .down = down,
        .identity = std::move(identity),
        .sz = std::move(sz),
        .sp = std::move(sp),
        .sm = std::move(sm),
        .sigma_z = std::move(sigma_z),
    };
}

} // namespace uni20
