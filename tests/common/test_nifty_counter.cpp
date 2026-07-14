#include <type_traits>
#include <uni20/common/nifty_counter.hpp>

#include <gtest/gtest.h>

namespace
{

int initialization_count = 0;
int finalization_count = 0;

void initialize() noexcept { ++initialization_count; }

void finalize() noexcept { ++finalization_count; }

using TestCounter = uni20::nifty_counter<initialize, finalize>;

static_assert(!std::is_copy_constructible_v<TestCounter>);
static_assert(!std::is_move_constructible_v<TestCounter>);

TEST(NiftyCounter, InitializesFirstAndFinalizesLast)
{
  EXPECT_EQ(initialization_count, 0);
  EXPECT_EQ(finalization_count, 0);

  {
    TestCounter first;
    EXPECT_EQ(initialization_count, 1);
    EXPECT_EQ(finalization_count, 0);

    {
      TestCounter second;
      EXPECT_EQ(initialization_count, 1);
      EXPECT_EQ(finalization_count, 0);
    }

    EXPECT_EQ(finalization_count, 0);
  }

  EXPECT_EQ(initialization_count, 1);
  EXPECT_EQ(finalization_count, 1);
}

} // namespace
