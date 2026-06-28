#include <uni20/krylov/tridiagonal.hpp>

#include <gtest/gtest.h>

#include <array>
#include <span>
#include <vector>

namespace
{

TEST(SymmetricTridiagonalView, IndexesDiagonalOffdiagonalAndZeros)
{
  std::array<double, 4> diagonal{1.0, 2.0, 3.0, 4.0};
  std::array<double, 3> offdiagonal{0.5, 0.25, 0.125};

  auto matrix =
      uni20::krylov::symmetric_tridiagonal(std::span<double const>(diagonal), std::span<double const>(offdiagonal));

  EXPECT_EQ(matrix.size(), 4);
  EXPECT_FALSE(matrix.empty());
  EXPECT_EQ((matrix[0, 0]), 1.0);
  EXPECT_EQ((matrix[3, 3]), 4.0);
  EXPECT_EQ((matrix[0, 1]), 0.5);
  EXPECT_EQ((matrix[1, 0]), 0.5);
  EXPECT_EQ((matrix[2, 3]), 0.125);
  EXPECT_EQ((matrix[3, 2]), 0.125);
  EXPECT_EQ((matrix[0, 2]), 0.0);
  EXPECT_EQ((matrix[3, 0]), 0.0);
}

TEST(SymmetricTridiagonalView, RejectsInvalidShape)
{
  std::array<double, 3> diagonal{1.0, 2.0, 3.0};
  std::array<double, 3> too_long{0.5, 0.25, 0.125};
  std::array<double, 1> too_short{0.5};
  std::array<double, 1> nonempty_offdiagonal{0.5};

  EXPECT_DEATH((void)(uni20::krylov::symmetric_tridiagonal(std::span<double const>(diagonal),
                                                           std::span<double const>(too_long))),
               "PRECONDITION");
  EXPECT_DEATH((void)(uni20::krylov::symmetric_tridiagonal(std::span<double const>(diagonal),
                                                           std::span<double const>(too_short))),
               "PRECONDITION");
  EXPECT_DEATH((void)(uni20::krylov::symmetric_tridiagonal(std::span<double const>(),
                                                           std::span<double const>(nonempty_offdiagonal))),
               "PRECONDITION");
}

TEST(SymmetricTridiagonalView, RejectsOutOfRangeIndex)
{
  std::array<double, 2> diagonal{1.0, 2.0};
  std::array<double, 1> offdiagonal{0.5};
  auto matrix =
      uni20::krylov::symmetric_tridiagonal(std::span<double const>(diagonal), std::span<double const>(offdiagonal));

  EXPECT_DEATH((void)(matrix[2, 0]), "PRECONDITION");
  EXPECT_DEATH((void)(matrix[0, 2]), "PRECONDITION");
}

TEST(SymmetricTridiagonalMatrix, OwnsMutableFixedSizeStorage)
{
  uni20::krylov::symmetric_tridiagonal_matrix<double> matrix(3);
  matrix.diagonal[0] = 1.0;
  matrix.diagonal[1] = 2.0;
  matrix.diagonal[2] = 3.0;
  matrix.offdiagonal[0] = 0.5;
  matrix.offdiagonal[1] = 0.25;

  EXPECT_EQ(matrix.size(), 3);
  EXPECT_EQ((matrix[0, 0]), 1.0);
  EXPECT_EQ((matrix[1, 2]), 0.25);
  EXPECT_EQ((matrix.view()[2, 1]), 0.25);
}

TEST(SymmetricTridiagonalMatrix, CopyAndMoveRebindSpansToOwnedStorage)
{
  uni20::krylov::symmetric_tridiagonal_matrix<double> original(2);
  original.diagonal[0] = 1.0;
  original.diagonal[1] = 2.0;
  original.offdiagonal[0] = 0.5;

  uni20::krylov::symmetric_tridiagonal_matrix<double> copied(original);
  original.diagonal[0] = 10.0;
  original.offdiagonal[0] = 7.0;

  EXPECT_EQ((copied[0, 0]), 1.0);
  EXPECT_EQ((copied[0, 1]), 0.5);

  uni20::krylov::symmetric_tridiagonal_matrix<double> moved(std::move(copied));
  EXPECT_EQ((moved[0, 0]), 1.0);
  EXPECT_EQ((moved[0, 1]), 0.5);
  moved.diagonal[0] = 11.0;
  moved.offdiagonal[0] = 8.0;
  EXPECT_EQ((moved[0, 0]), 11.0);
  EXPECT_EQ((moved[1, 0]), 8.0);
}

} // namespace
