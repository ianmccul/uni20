#include <uni20/matrix/sparse_matrix.hpp>

#include <gtest/gtest.h>

using namespace uni20;

TEST(SparseMatrixTest, TracksShapeAndEntryCount)
{
    SparseMatrix<double> matrix(3, 4);

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    EXPECT_EQ(matrix.shape(), (std::pair<std::size_t, std::size_t>{3, 4}));
    EXPECT_FALSE(matrix.empty());
    EXPECT_EQ(matrix.nnz(), 0);
    EXPECT_EQ(matrix.row_size(0), 0);
}

TEST(SparseMatrixTest, InsertOrAssignKeepsRowsSorted)
{
    SparseMatrix<int> matrix(2, 5);

    matrix.insert_or_assign(1, 3, 30);
    matrix.insert_or_assign(1, 1, 10);
    matrix.insert_or_assign(1, 4, 40);

    auto const row = matrix.row(1);
    ASSERT_EQ(row.size(), 3);
    EXPECT_EQ(row[0].column, 1);
    EXPECT_EQ(row[0].value, 10);
    EXPECT_EQ(row[1].column, 3);
    EXPECT_EQ(row[1].value, 30);
    EXPECT_EQ(row[2].column, 4);
    EXPECT_EQ(row[2].value, 40);
    EXPECT_EQ(matrix.nnz(), 3);
}

TEST(SparseMatrixTest, OverwriteFindAndAtOperateOnExistingEntries)
{
    SparseMatrix<double> matrix(3, 3);

    matrix.insert_or_assign(0, 2, 1.5);
    matrix.insert_or_assign(0, 2, 2.5);

    ASSERT_TRUE(matrix.contains(0, 2));
    ASSERT_NE(matrix.find(0, 2), nullptr);
    EXPECT_EQ(*matrix.find(0, 2), 2.5);
    EXPECT_EQ(matrix.at(0, 2), 2.5);
    EXPECT_EQ(matrix.nnz(), 1);

    ASSERT_EQ(matrix.find(2, 1), nullptr);
    EXPECT_THROW((void)matrix.at(2, 1), std::out_of_range);
    EXPECT_THROW((void)matrix.find(3, 1), std::out_of_range);
}

TEST(SparseMatrixTest, EraseAndClearUpdateCounts)
{
    SparseMatrix<int> matrix(3, 3);

    matrix.insert_or_assign(0, 0, 1);
    matrix.insert_or_assign(0, 2, 2);
    matrix.insert_or_assign(2, 1, 3);

    EXPECT_TRUE(matrix.erase(0, 2));
    EXPECT_FALSE(matrix.contains(0, 2));
    EXPECT_EQ(matrix.nnz(), 2);
    EXPECT_FALSE(matrix.erase(0, 2));

    matrix.clear_row(0);
    EXPECT_EQ(matrix.row_size(0), 0);
    EXPECT_EQ(matrix.nnz(), 1);

    matrix.clear();
    EXPECT_EQ(matrix.nnz(), 0);
    EXPECT_EQ(matrix.row_size(2), 0);
    EXPECT_THROW(matrix.clear_row(3), std::out_of_range);
}

TEST(SparseMatrixTest, TransposeSwapsRowsAndColumns)
{
    SparseMatrix<double> matrix(2, 3);
    matrix.insert_or_assign(0, 1, 1.0);
    matrix.insert_or_assign(1, 0, 2.0);
    matrix.insert_or_assign(1, 2, 3.0);

    auto const transposed = matrix.transpose();

    EXPECT_EQ(transposed.rows(), 3);
    EXPECT_EQ(transposed.cols(), 2);
    EXPECT_EQ(transposed.nnz(), 3);
    EXPECT_EQ(transposed.at(1, 0), 1.0);
    EXPECT_EQ(transposed.at(0, 1), 2.0);
    EXPECT_EQ(transposed.at(2, 1), 3.0);
}
