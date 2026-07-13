#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20::async;

namespace
{

struct IntAlias
{
    using async_alias_tag = void;

    explicit IntAlias(int const* value_in) : value(value_in) {}

    [[nodiscard]] int get() const { return *value; }

    int const* value;
};

static_assert(async_value_kind_v<int> == async_value_kind::value);
static_assert(async_value_kind_v<IntAlias> == async_value_kind::shared_alias);

} // namespace

TEST(AsyncAliasTest, CopiesShareStorageOwnerAndEpochQueue)
{
  Async<int> parent = 5;
  auto alias = make_async_alias<IntAlias>(parent, parent.storage().storage_address());

  EXPECT_EQ(parent.storage().use_count(), 2);
  EXPECT_EQ(alias.storage().use_count(), 1);
  EXPECT_EQ(&alias.queue(), &parent.queue());

  auto copy = alias;
  EXPECT_EQ(parent.storage().use_count(), 2);
  EXPECT_EQ(alias.storage().use_count(), 2);
  EXPECT_EQ(copy.storage().control_address(), alias.storage().control_address());
  EXPECT_EQ(&copy.queue(), &parent.queue());
}

TEST(AsyncAliasTest, AliasOutlivesParentHandle)
{
  auto make_alias = [] {
    Async<int> parent = 7;
    return make_async_alias<IntAlias>(parent, parent.storage().storage_address());
  };

  auto alias = make_alias();
  DebugScheduler sched;
  sched.schedule([](ReadBuffer<IntAlias> reader) static -> AsyncTask {
    EXPECT_EQ((co_await reader).get(), 7);
    co_return;
  }(alias.read()));

  sched.run_all();
}

TEST(AsyncAliasTest, ParentWriterPrecedesAliasReader)
{
  Async<int> parent;
  auto alias = make_async_alias<IntAlias>(parent, parent.storage().storage_address());
  DebugScheduler sched;

  sched.schedule([](WriteBuffer<int> writer) static -> AsyncTask {
    (co_await writer).emplace(31);
    co_return;
  }(parent.write()));

  sched.schedule([](ReadBuffer<IntAlias> reader) static -> AsyncTask {
    EXPECT_EQ((co_await reader).get(), 31);
    co_return;
  }(alias.read()));

  sched.run_all();
}
