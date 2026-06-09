#include <uni20/symmetry/u1.hpp>

#include <gtest/gtest.h>

#include <format>
#include <sstream>
#include <unordered_set>

using namespace uni20;

TEST(U1Test, ValueSemanticsAndBasicOperationsWork)
{
    U1 const zero;
    U1 const half{half_int{0.5}};
    U1 const two{2};

    EXPECT_EQ(zero.value(), half_int{0});
    EXPECT_EQ(half.value(), half_int{0.5});
    EXPECT_EQ(two.value(), half_int{2});
    EXPECT_EQ(dual(half), U1{half_int{-0.5}});
    EXPECT_EQ(two + U1{half_int{0.5}}, U1{half_int{2.5}});
    EXPECT_EQ(two - U1{half_int{0.5}}, U1{half_int{1.5}});
    EXPECT_DOUBLE_EQ(qdim(half), 1.0);
    EXPECT_EQ(degree(half), 1);
}

TEST(U1Test, FormattingSupportsDecimalFractionStreamAndStdFormat)
{
    U1 const value{half_int{2.5}};

    EXPECT_EQ(to_string(value), "2.5");
    EXPECT_EQ(to_string_fraction(value), "5/2");

    std::ostringstream out;
    out << value;
    EXPECT_EQ(out.str(), "2.5");
    EXPECT_EQ(std::format("{}", value), "2.5");
}

TEST(U1Test, HashingAndCanonicalEncodingSupportContainersAndDisplayOrder)
{
    std::unordered_set<U1> seen;
    seen.insert(U1{half_int{0.5}});
    seen.insert(U1{half_int{0.5}});
    seen.insert(U1{half_int{-0.5}});

    EXPECT_EQ(seen.size(), 2);

    EXPECT_EQ(detail::SymmetryFactorTraits<U1>::encode(U1{0}), 0);
    EXPECT_EQ(detail::SymmetryFactorTraits<U1>::encode(U1{half_int{0.5}}), 1);
    EXPECT_EQ(detail::SymmetryFactorTraits<U1>::encode(U1{half_int{-0.5}}), 2);
    EXPECT_EQ(detail::SymmetryFactorTraits<U1>::decode(3), U1{1});
    EXPECT_EQ(detail::SymmetryFactorTraits<U1>::decode(4), U1{-1});
}
