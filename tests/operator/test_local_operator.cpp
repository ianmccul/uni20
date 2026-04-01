#include <uni20/operator/local_operator.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(LocalOperatorTest, ConstructsWithSparseCoefficientShape)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const bra(sym, {make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    LocalSpace const ket(sym, {make_qnum(sym, {{"N", U1{-1}}}), make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    QNum const transforms_as = make_qnum(sym, {{"N", U1{1}}});

    LocalOperator op(bra, ket, transforms_as);

    EXPECT_EQ(op.symmetry(), sym);
    EXPECT_EQ(op.transforms_as(), transforms_as);
    EXPECT_EQ(op.rows(), 2);
    EXPECT_EQ(op.cols(), 3);
    EXPECT_EQ(op.nnz(), 0);
    EXPECT_EQ(op.bra_space(), bra);
    EXPECT_EQ(op.ket_space(), ket);
}

TEST(LocalOperatorTest, StoresSparseCoefficients)
{
    Symmetry const sym{"N:U(1)"};
    LocalSpace const space(sym, {make_qnum(sym, {{"N", U1{0}}}), make_qnum(sym, {{"N", U1{1}}})});
    LocalOperator op(space, space, QNum::identity(sym));

    op.insert_or_assign(0, 1, 2.5);
    op.insert_or_assign(1, 0, -3.0);

    EXPECT_TRUE(op.contains(0, 1));
    EXPECT_EQ(op.at(0, 1), 2.5);
    EXPECT_EQ(op.at(1, 0), -3.0);
    EXPECT_EQ(op.nnz(), 2);

    EXPECT_TRUE(op.erase(0, 1));
    EXPECT_FALSE(op.contains(0, 1));
    EXPECT_EQ(op.nnz(), 1);

    op.clear();
    EXPECT_EQ(op.nnz(), 0);
}

TEST(LocalOperatorTest, RejectsMismatchedSymmetryOrShape)
{
    Symmetry const n_sym{"N:U(1)"};
    Symmetry const sz_sym{"Sz:U(1)"};

    LocalSpace const bra(n_sym, {make_qnum(n_sym, {{"N", U1{0}}})});
    LocalSpace const ket(sz_sym, {make_qnum(sz_sym, {{"Sz", U1{0}}})});

    EXPECT_THROW((void)LocalOperator(bra, ket, QNum::identity(n_sym)), std::invalid_argument);

    LocalSpace const ket_n(n_sym, {make_qnum(n_sym, {{"N", U1{0}}})});
    EXPECT_THROW((void)LocalOperator(bra, ket_n, QNum::identity(sz_sym)), std::invalid_argument);

    SparseMatrix<double> wrong_shape(2, 2);
    EXPECT_THROW((void)LocalOperator(bra, ket_n, QNum::identity(n_sym), std::move(wrong_shape)), std::invalid_argument);
}
