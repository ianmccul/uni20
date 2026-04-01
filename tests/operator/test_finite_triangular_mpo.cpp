#include <uni20/operator/finite_triangular_mpo.hpp>

#include <gtest/gtest.h>

using namespace uni20;

namespace
{

auto make_identity_local(LocalSpace const& space, Symmetry sym) -> LocalOperator
{
    LocalOperator op(space, space, QNum::identity(sym));
    for (std::size_t i = 0; i < space.size(); ++i)
    {
        op.insert_or_assign(i, i, 1.0);
    }
    return op;
}

} // namespace

TEST(OperatorComponentTest, UpperTriangularAllowsRectangularComponents)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}})});
    LocalSpace const left(sym, {QNum::identity(sym)});
    LocalSpace const right(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}}), make_qnum(sym, {{"N", U1{2}}})});

    OperatorComponent component(local, local, left, right);
    component.insert_or_assign(0, 0, make_identity_local(local, sym));
    component.insert_or_assign(0, 2, make_identity_local(local, sym));

    EXPECT_TRUE(is_upper_triangular(component));
}

TEST(OperatorComponentTest, UpperTriangularRejectsBelowDiagonalEntries)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}})});
    LocalSpace const aux(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}})});

    OperatorComponent component(local, local, aux, aux);
    component.insert_or_assign(1, 0, make_identity_local(local, sym));

    EXPECT_FALSE(is_upper_triangular(component));
}

TEST(FiniteTriangularMPOTest, ConstructsCompatibleUpperTriangularChain)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const left0(sym, {QNum::identity(sym)});
    LocalSpace const mid(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const right2(sym, {QNum::identity(sym)});

    OperatorComponent site0(local, local, left0, mid);
    OperatorComponent site1(local, local, mid, right2);

    site0.insert_or_assign(0, 0, make_identity_local(local, sym));
    site0.insert_or_assign(0, 1, make_identity_local(local, sym));
    site1.insert_or_assign(0, 0, make_identity_local(local, sym));

    FiniteTriangularMPO mpo({site0, site1});

    EXPECT_EQ(mpo.size(), 2);
    EXPECT_EQ(mpo.symmetry(), sym);
    EXPECT_EQ(mpo.left_boundary_virtual_space(), left0);
    EXPECT_EQ(mpo.right_boundary_virtual_space(), right2);
    EXPECT_TRUE(is_upper_triangular(mpo));
}

TEST(FiniteTriangularMPOTest, RejectsVirtualMismatchOrNonTriangularSites)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}})});
    LocalSpace const left(sym, {QNum::identity(sym)});
    LocalSpace const mid(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const other_mid(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{2}}})});

    OperatorComponent compatible(local, local, left, mid);
    compatible.insert_or_assign(0, 0, make_identity_local(local, sym));

    OperatorComponent wrong_virtual(local, local, other_mid, left);
    wrong_virtual.insert_or_assign(0, 0, make_identity_local(local, sym));

    EXPECT_THROW((void)FiniteTriangularMPO({compatible, wrong_virtual}), std::invalid_argument);

    OperatorComponent non_triangular(local, local, mid, left);
    non_triangular.insert_or_assign(1, 0, make_identity_local(local, sym));
    EXPECT_THROW((void)FiniteTriangularMPO({compatible, non_triangular}), std::invalid_argument);
}
