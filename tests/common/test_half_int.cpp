#include <uni20/common/half_int.hpp>

#include <gtest/gtest.h>

#include <format>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

using uni20::basic_half_int;
using uni20::half_int;

TEST(HalfIntTest, ConstructsFromIntegralAndFloatingValues)
{
    half_int const zero;
    half_int const one = 1;
    half_int const half{0.5};
    half_int const rounded{1.26};

    EXPECT_EQ(zero.twice(), 0);
    EXPECT_EQ(one.twice(), 2);
    EXPECT_EQ(half.twice(), 1);
    EXPECT_EQ(rounded.twice(), 3);
}

TEST(HalfIntTest, ArithmeticAndComparisonWork)
{
    half_int const a{1.5};
    half_int const b = 2;

    EXPECT_EQ((a + b).twice(), 7);
    EXPECT_EQ((b - a).twice(), 1);
    EXPECT_EQ((-a).twice(), -3);
    EXPECT_EQ((a * 2).twice(), 6);
    EXPECT_DOUBLE_EQ(a * b, 3.0);
    EXPECT_LT(a, b);
    EXPECT_TRUE(abs(-a) == a);
}

TEST(HalfIntTest, IntegralQueriesAndConversionsWork)
{
    half_int const three = 3;
    half_int const half{0.5};

    EXPECT_TRUE(three.is_integral());
    EXPECT_FALSE(half.is_integral());
    EXPECT_EQ(three.to_int(), 3);
    EXPECT_THROW(static_cast<void>(half.to_int()), std::runtime_error);
    EXPECT_EQ(uni20::to_int_assert(three), 3);
}

TEST(HalfIntTest, ParsingAndStreamingSupportExpectedFormats)
{
    EXPECT_EQ(half_int::parse("3").twice(), 6);
    EXPECT_EQ(half_int::parse("5/2").twice(), 5);
    EXPECT_EQ(half_int::parse("-1.5").twice(), -3);
    EXPECT_EQ(half_int::parse("4.0").twice(), 8);
    EXPECT_THROW(static_cast<void>(half_int::parse("1.25")), std::runtime_error);
    EXPECT_THROW(static_cast<void>(half_int::parse("3/4")), std::runtime_error);

    std::istringstream in{"-3/2 2"};
    half_int a;
    half_int b;
    in >> a >> b;
    EXPECT_EQ(a.twice(), -3);
    EXPECT_EQ(b.twice(), 4);

    std::ostringstream out;
    out << half_int{-0.5} << ' ' << half_int{2.5};
    EXPECT_EQ(out.str(), "-0.5 2.5");
}

TEST(HalfIntTest, FormattingHelpersAndHashingWork)
{
    half_int const a{-0.5};
    half_int const b{2.5};

    EXPECT_EQ(uni20::to_string(a), "-0.5");
    EXPECT_EQ(uni20::to_string_fraction(b), "5/2");
    EXPECT_EQ(std::format("{}", b), "2.5");

    std::unordered_set<half_int> values;
    values.insert(a);
    values.insert(b);
    EXPECT_TRUE(values.contains(a));
    EXPECT_TRUE(values.contains(b));
}

TEST(HalfIntTest, HelpersCoverTriangleAndParity)
{
    EXPECT_EQ(uni20::minus1pow(0), 1);
    EXPECT_EQ(uni20::minus1pow(3), -1);

    EXPECT_TRUE(uni20::is_triangle(half_int{1}, half_int{0.5}, half_int{0.5}));
    EXPECT_FALSE(uni20::is_triangle(half_int{2}, half_int{0.5}, half_int{0.5}));
}

TEST(HalfIntTest, AliasUsesInt64Storage)
{
    static_assert(std::same_as<half_int, basic_half_int<std::int64_t>>);
}
