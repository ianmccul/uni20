#include <uni20/operator/operator_component.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(OperatorComponentTest, ConstructsEmptySparseMatrixOverAuxiliarySpaces)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const left(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const right(sym, {QNum::identity(sym)});

    OperatorComponent component(local, local, left, right);

    EXPECT_EQ(component.symmetry(), sym);
    EXPECT_EQ(component.local_bra_space(), local);
    EXPECT_EQ(component.local_ket_space(), local);
    EXPECT_EQ(component.left_virtual_space(), left);
    EXPECT_EQ(component.right_virtual_space(), right);
    EXPECT_EQ(component.rows(), 2);
    EXPECT_EQ(component.cols(), 1);
    EXPECT_EQ(component.nnz(), 0);
}

TEST(OperatorComponentTest, StoresCompatibleLocalOperators)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const local(sym, {make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const aux(sym, {QNum::identity(sym), make_qnum(sym, {{"N", U1{1}}})});

    OperatorComponent component(local, local, aux, aux);

    LocalOperator creation(local, local, make_qnum(sym, {{"N", U1{1}}}));
    creation.insert_or_assign(1, 0, 1.0);

    component.insert_or_assign(0, 1, creation);

    ASSERT_TRUE(component.contains(0, 1));
    EXPECT_EQ(component.at(0, 1).transforms_as(), make_qnum(sym, {{"N", U1{1}}}));
    EXPECT_EQ(component.at(0, 1).at(1, 0), 1.0);
    EXPECT_EQ(component.nnz(), 1);
}

TEST(OperatorComponentTest, RejectsIncompatibleEntriesOrShapes)
{
    Symmetry const n_sym{"N:U(1)"};
    Symmetry const sz_sym{"Sz:U(1)"};

    LocalSpace const local(n_sym, {make_qnum(n_sym, {{"N", U1{0}}})});
    LocalSpace const aux(n_sym, {QNum::identity(n_sym)});
    OperatorComponent component(local, local, aux, aux);

    LocalSpace const wrong_local(sz_sym, {make_qnum(sz_sym, {{"Sz", U1{0}}})});
    LocalOperator wrong_op(wrong_local, wrong_local, QNum::identity(sz_sym));
    EXPECT_THROW(component.insert_or_assign(0, 0, wrong_op), std::invalid_argument);

    LocalSpace const wrong_aux(sz_sym, {QNum::identity(sz_sym)});
    EXPECT_THROW((void)OperatorComponent(local, local, aux, wrong_aux), std::invalid_argument);

    SparseMatrix<LocalOperator> wrong_shape(2, 2);
    EXPECT_THROW((void)OperatorComponent(local, local, aux, aux, std::move(wrong_shape)), std::invalid_argument);
}
