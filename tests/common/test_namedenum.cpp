#include "common/namedenum.hpp"
#include "gtest/gtest.h"

#include <array>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
struct ExampleEnumTraits
{
    enum Enum
    {
      Alpha,
      Beta,
      Gamma
    };

    inline static constexpr Enum Default = Beta;
    inline static constexpr const char* StaticName = "example enumeration";
    inline static constexpr std::array<const char*, 3> Names = {"alpha", "beta", "gamma"};
};

using ExampleNamedEnumeration = NamedEnumeration<ExampleEnumTraits>;
} // namespace

TEST(NamedEnumerationTest, DefaultConstructionAndOperators)
{
  ExampleNamedEnumeration value;
  EXPECT_TRUE(value == ExampleEnumTraits::Default);

  ++value;
  EXPECT_TRUE(value == ExampleEnumTraits::Gamma);

  ExampleNamedEnumeration other(ExampleEnumTraits::Gamma);
  EXPECT_TRUE(value == other);

  --value;
  EXPECT_TRUE(value == ExampleEnumTraits::Beta);
  EXPECT_TRUE(value != ExampleEnumTraits::Alpha);
}

TEST(NamedEnumerationTest, ListAndEnumerate)
{
  EXPECT_EQ(ExampleNamedEnumeration::ListAll(), "alpha, beta, gamma");

  std::vector<std::string> expected{"alpha", "beta", "gamma"};
  EXPECT_EQ(ExampleNamedEnumeration::EnumerateAll(), expected);
}

TEST(NamedEnumerationTest, CaseInsensitiveConstructionAndError)
{
  ExampleNamedEnumeration uppercase{"GAMMA"};
  EXPECT_TRUE(uppercase == ExampleEnumTraits::Gamma);

  try
  {
    [[maybe_unused]] ExampleNamedEnumeration invalid{"unknown"};
    FAIL() << "Expected runtime_error when constructing with invalid name";
  }
  catch (const std::runtime_error& err)
  {
    EXPECT_NE(std::string_view{err.what()}.find("Unknown initializer for example enumeration"), std::string_view::npos);
  }
}

TEST(NamedEnumerationTest, FormatterUsesFriendlyName)
{
  ExampleNamedEnumeration value{ExampleEnumTraits::Alpha};
  EXPECT_EQ(std::format("{}", value), "alpha");
}
