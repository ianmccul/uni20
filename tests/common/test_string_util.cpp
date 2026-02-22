#include "common/string_util.hpp"
#include "gtest/gtest.h"

#include <stdexcept>

namespace
{
struct StreamReadable
{
    int value{};
};

std::istream& operator>>(std::istream& is, StreamReadable& readable)
{
  int tmp{};
  if (is >> tmp)
  {
    readable.value = tmp;
  }
  else
  {
    is.setstate(std::ios::failbit);
  }
  return is;
}
} // namespace

TEST(StringUtilTest, IEEqualsMatchesCaseInsensitive)
{
  EXPECT_TRUE(iequals("Hello", "heLLo"));
  EXPECT_FALSE(iequals("Hello", "World"));
}

TEST(StringUtilTest, FromStringArithmeticSuccess)
{
  EXPECT_EQ(from_string<int>("42"), 42);
  EXPECT_DOUBLE_EQ(from_string<double>("3.125"), 3.125);
}

TEST(StringUtilTest, FromStringArithmeticInvalidInput) { EXPECT_THROW(from_string<int>("abc"), std::runtime_error); }

TEST(StringUtilTest, FromStringUsesStreamExtractorWhenAvailable)
{
  StreamReadable readable = from_string<StreamReadable>("123");
  EXPECT_EQ(readable.value, 123);
}

TEST(StringUtilTest, FromStringConstructsStdStringDirectly)
{
  std::string_view input = "direct";
  EXPECT_EQ(from_string<std::string>(input), std::string(input));
}
