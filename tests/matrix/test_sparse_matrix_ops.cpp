#include <uni20/matrix/sparse_matrix_ops.hpp>

#include <gtest/gtest.h>

#include <string>
#include <stdexcept>

using namespace uni20;

TEST(SparseMatrixOpsTest, AddMergesSparsePatterns)
{
    SparseMatrix<int> lhs(2, 3);
    SparseMatrix<int> rhs(2, 3);

    lhs.insert_or_assign(0, 0, 1);
    lhs.insert_or_assign(0, 2, 2);
    rhs.insert_or_assign(0, 2, 5);
    rhs.insert_or_assign(1, 1, 7);

    auto const sum = add(lhs, rhs);

    EXPECT_EQ(sum.nnz(), 3);
    EXPECT_EQ(sum.at(0, 0), 1);
    EXPECT_EQ(sum.at(0, 2), 7);
    EXPECT_EQ(sum.at(1, 1), 7);
}

TEST(SparseMatrixOpsTest, AddSupportsCustomOverlapFunctor)
{
    SparseMatrix<int> lhs(1, 3);
    SparseMatrix<int> rhs(1, 3);

    lhs.insert_or_assign(0, 0, 1);
    lhs.insert_or_assign(0, 1, 3);
    rhs.insert_or_assign(0, 1, 2);
    rhs.insert_or_assign(0, 2, 4);

    auto const sum = add(lhs, rhs, [](int x, int y) { return std::max(x, y); });

    EXPECT_EQ(sum.at(0, 0), 1);
    EXPECT_EQ(sum.at(0, 1), 3);
    EXPECT_EQ(sum.at(0, 2), 4);
}

TEST(SparseMatrixOpsTest, ScaleAppliesScalarActionToEachStoredEntry)
{
    SparseMatrix<double> matrix(2, 2);
    matrix.insert_or_assign(0, 1, 1.5);
    matrix.insert_or_assign(1, 0, -2.0);

    auto const scaled = scale(matrix, 3.0);

    EXPECT_EQ(scaled.at(0, 1), 4.5);
    EXPECT_EQ(scaled.at(1, 0), -6.0);
}

TEST(SparseMatrixOpsTest, ScaleSupportsCustomFunctor)
{
    SparseMatrix<int> matrix(1, 2);
    matrix.insert_or_assign(0, 0, 5);
    matrix.insert_or_assign(0, 1, 8);

    auto const shifted = scale(matrix, 2, [](int value, int scalar) { return value + scalar; });

    EXPECT_EQ(shifted.at(0, 0), 7);
    EXPECT_EQ(shifted.at(0, 1), 10);
}

TEST(SparseMatrixOpsTest, MultiplyAccumulatesMatchingInnerContributions)
{
    SparseMatrix<int> lhs(2, 3);
    SparseMatrix<int> rhs(3, 2);

    lhs.insert_or_assign(0, 0, 1);
    lhs.insert_or_assign(0, 2, 2);
    lhs.insert_or_assign(1, 1, 3);

    rhs.insert_or_assign(0, 1, 4);
    rhs.insert_or_assign(1, 0, 5);
    rhs.insert_or_assign(2, 0, 6);
    rhs.insert_or_assign(2, 1, 7);

    auto const product = multiply(lhs, rhs);

    EXPECT_EQ(product.rows(), 2);
    EXPECT_EQ(product.cols(), 2);
    EXPECT_EQ(product.at(0, 0), 12);
    EXPECT_EQ(product.at(0, 1), 18);
    EXPECT_EQ(product.at(1, 0), 15);
}

TEST(SparseMatrixOpsTest, MultiplySupportsCustomNestedOperations)
{
    SparseMatrix<std::string> lhs(1, 2);
    SparseMatrix<std::string> rhs(2, 1);

    lhs.insert_or_assign(0, 0, "a");
    lhs.insert_or_assign(0, 1, "b");
    rhs.insert_or_assign(0, 0, "x");
    rhs.insert_or_assign(1, 0, "y");

    auto const product =
        multiply(lhs, rhs, [](std::string const& x, std::string const& y) { return x + y; },
                 [](std::string const& x, std::string const& y) { return x + "|" + y; });

    EXPECT_EQ(product.at(0, 0), "ax|by");
}

TEST(SparseMatrixOpsTest, KroneckerFormsTensorProductPattern)
{
    SparseMatrix<int> lhs(2, 2);
    SparseMatrix<int> rhs(2, 2);

    lhs.insert_or_assign(0, 1, 2);
    lhs.insert_or_assign(1, 0, 3);
    rhs.insert_or_assign(0, 0, 5);
    rhs.insert_or_assign(1, 1, 7);

    auto const product = kronecker(lhs, rhs);

    EXPECT_EQ(product.rows(), 4);
    EXPECT_EQ(product.cols(), 4);
    EXPECT_EQ(product.nnz(), 4);
    EXPECT_EQ(product.at(0, 2), 10);
    EXPECT_EQ(product.at(1, 3), 14);
    EXPECT_EQ(product.at(2, 0), 15);
    EXPECT_EQ(product.at(3, 1), 21);
}

TEST(SparseMatrixOpsTest, ShapeMismatchThrows)
{
    SparseMatrix<int> lhs(2, 2);
    SparseMatrix<int> rhs_add(2, 3);
    SparseMatrix<int> rhs_mul(3, 2);

    EXPECT_THROW((void)add(lhs, rhs_add), std::invalid_argument);
    EXPECT_THROW((void)multiply(lhs, rhs_mul), std::invalid_argument);
}
