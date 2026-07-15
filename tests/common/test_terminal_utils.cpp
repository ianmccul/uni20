#include <uni20/common/terminal.hpp>
#include <uni20/config.hpp>

#include "env_var_guard.hpp"

#include "gtest/gtest.h"

#include <cstdio>
#include <string>

namespace
{
using uni20::test::EnvVarGuard;

} // namespace

TEST(TerminalUtilsTest, QuoteShellHandlesPlainTokens)
{
  EXPECT_EQ(terminal::quote_shell("plain"), "plain");
  EXPECT_EQ(terminal::quote_shell("needs space"), "\"needs space\"");
}

TEST(TerminalUtilsTest, QuoteShellEscapesSpecialCharacters)
{
  EXPECT_EQ(terminal::quote_shell("with\"quote"), "\"with\\\"quote\"");
  EXPECT_EQ(terminal::quote_shell("path\\segment"), "\"path\\\\segment\"");
}

TEST(TerminalUtilsTest, CmdlineProducesQuotedCommandLine)
{
  char arg0[] = "prog";
  char arg1[] = "simple";
  char arg2[] = "needs space";
  char arg3[] = "quote\"and\\backslash";
  char* argv[] = {arg0, arg1, arg2, arg3};

  EXPECT_EQ(terminal::cmdline(4, argv), "prog simple \"needs space\" \"quote\\\"and\\\\backslash\"");
}

TEST(TerminalUtilsTest, GetenvOrDefaultIntReturnsConvertedValue)
{
  EnvVarGuard guard("UNI20_TEST_ENV_INT_VALUE");
  guard.set("42");

  EXPECT_EQ(terminal::getenv_or_default<int>("UNI20_TEST_ENV_INT_VALUE", 7), 42);
}

TEST(TerminalUtilsTest, GetenvOrDefaultIntFallsBackWhenMissing)
{
  EnvVarGuard guard("UNI20_TEST_ENV_INT_MISSING");
  guard.unset();

  EXPECT_EQ(terminal::getenv_or_default<int>("UNI20_TEST_ENV_INT_MISSING", 5), 5);
}

TEST(TerminalUtilsTest, GetenvOrDefaultIntIgnoresUnparsableInput)
{
  EnvVarGuard guard("UNI20_TEST_ENV_INT_BAD");
  guard.set("not-a-number");

  EXPECT_EQ(terminal::getenv_or_default<int>("UNI20_TEST_ENV_INT_BAD", 9), 9);
}

TEST(TerminalUtilsTest, ToggleParsesAffirmativeTokens)
{
  EXPECT_TRUE(terminal::toggle("yes", false));
  EXPECT_TRUE(terminal::toggle("true", false));
  EXPECT_TRUE(terminal::toggle("1", false));
}

TEST(TerminalUtilsTest, ToggleParsesNegativeTokens)
{
  EXPECT_FALSE(terminal::toggle("no", true));
  EXPECT_FALSE(terminal::toggle("false", true));
  EXPECT_FALSE(terminal::toggle("0", true));
}

TEST(TerminalUtilsTest, ToggleUsesDefaultForEmptyString)
{
  EXPECT_FALSE(terminal::toggle("", false));
  EXPECT_TRUE(terminal::toggle("", true));
}

TEST(TerminalUtilsTest, ToggleUsesDefaultForUnknownToken)
{
  EXPECT_TRUE(terminal::toggle("maybe", true));
  EXPECT_FALSE(terminal::toggle("maybe", false));
}

TEST(TerminalUtilsTest, ColumnsUsesConfiguredFallbackForRedirectedFile)
{
  EnvVarGuard columns("COLUMNS");
  columns.unset();
  std::FILE* const file = std::tmpfile();
  ASSERT_NE(file, nullptr);

  EXPECT_FALSE(terminal::is_a_terminal(file));
  EXPECT_EQ(terminal::columns(file), UNI20_FALLBACK_TERMINAL_WIDTH);

  std::fclose(file);
}

TEST(TerminalUtilsTest, ColumnsEnvironmentOverridesRedirectedFileFallback)
{
  EnvVarGuard columns("COLUMNS", "137");
  std::FILE* const file = std::tmpfile();
  ASSERT_NE(file, nullptr);

  EXPECT_EQ(terminal::columns(file), 137);

  std::fclose(file);
}

TEST(TerminalUtilsTest, ColumnsRejectsMalformedEnvironmentOverride)
{
  EnvVarGuard columns("COLUMNS", "137junk");
  std::FILE* const file = std::tmpfile();
  ASSERT_NE(file, nullptr);

  EXPECT_EQ(terminal::columns(file), UNI20_FALLBACK_TERMINAL_WIDTH);

  std::fclose(file);
}
