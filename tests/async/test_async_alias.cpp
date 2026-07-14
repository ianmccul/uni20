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

struct MutableIntAlias
{
    using async_alias_tag = void;
    using async_write_through_tag = void;

    explicit MutableIntAlias(int* value_in) : value(value_in) {}

    MutableIntAlias& operator=(MutableIntAlias const& other)
    {
      *value = *other.value;
      return *this;
    }

    MutableIntAlias& operator=(int new_value)
    {
      *value = new_value;
      return *this;
    }

    [[nodiscard]] int get() const { return *value; }

    int* value;
};

static_assert(async_value_kind_v<int> == async_value_kind::value);
static_assert(async_value_kind_v<IntAlias> == async_value_kind::shared_alias);
static_assert(async_assignment_kind_v<int> == async_assignment_kind::rebind);
static_assert(async_assignment_kind_v<IntAlias> == async_assignment_kind::rebind);
static_assert(async_assignment_kind_v<MutableIntAlias> == async_assignment_kind::write_through);

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

TEST(AsyncAliasTest, MutableAliasAssignmentWritesThroughExistingTimeline)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> parent = 5;
  Async<int> source = 17;
  auto alias = make_async_alias<MutableIntAlias>(parent, parent.storage_address());
  auto* const alias_storage = alias.storage().control_address();
  auto const* const alias_queue = &alias.queue();

  alias = 11;

  EXPECT_EQ(alias.storage().control_address(), alias_storage);
  EXPECT_EQ(&alias.queue(), alias_queue);

  sched.run_all();
  EXPECT_EQ(parent.get_wait(), 11);

  alias = source;

  EXPECT_EQ(alias.storage().control_address(), alias_storage);
  EXPECT_EQ(&alias.queue(), alias_queue);

  sched.run_all();
  EXPECT_EQ(parent.get_wait(), 17);
  EXPECT_EQ(alias.get_wait().get(), 17);
}

TEST(AsyncAliasTest, HeterogeneousAsyncAssignmentRebindsFreshTimeline)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<long> destination = 5;
  Async<int> source = 17;
  auto* const old_storage = destination.storage().control_address();
  auto const* const old_queue = &destination.queue();
  long old_observation = 0;
  long new_observation = 0;

  schedule([](ReadBuffer<long> reader, long& observation) static -> AsyncTask {
    observation = co_await reader;
    co_return;
  }(destination.read(), old_observation));

  destination = source;

  EXPECT_NE(destination.storage().control_address(), old_storage);
  EXPECT_NE(&destination.queue(), old_queue);

  schedule([](ReadBuffer<long> reader, long& observation) static -> AsyncTask {
    observation = co_await reader;
    co_return;
  }(destination.read(), new_observation));

  sched.run_all();
  EXPECT_EQ(old_observation, 5);
  EXPECT_EQ(new_observation, 17);
}

TEST(AsyncAliasTest, ExactAliasAssignmentRebindsDescriptorOwnerAndQueue)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> first_parent = 3;
  Async<int> second_parent = 8;
  auto alias = make_async_alias<MutableIntAlias>(first_parent, first_parent.storage_address());
  auto replacement = make_async_alias<MutableIntAlias>(second_parent, second_parent.storage_address());
  int old_observation = 0;
  int new_observation = 0;

  schedule([](ReadBuffer<MutableIntAlias> reader, int& observation) static -> AsyncTask {
    observation = (co_await reader).get();
    co_return;
  }(alias.read(), old_observation));

  alias = replacement;

  EXPECT_EQ(alias.storage().control_address(), replacement.storage().control_address());
  EXPECT_EQ(&alias.queue(), &second_parent.queue());

  schedule([](ReadBuffer<MutableIntAlias> reader, int& observation) static -> AsyncTask {
    observation = (co_await reader).get();
    co_return;
  }(alias.read(), new_observation));

  sched.run_all();
  EXPECT_EQ(old_observation, 3);
  EXPECT_EQ(new_observation, 8);
}

TEST(AsyncAliasTest, ExplicitAsyncAssignWritesThroughExactAliasType)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> destination_parent = 3;
  Async<int> source_parent = 8;
  auto destination = make_async_alias<MutableIntAlias>(destination_parent, destination_parent.storage_address());
  auto source = make_async_alias<MutableIntAlias>(source_parent, source_parent.storage_address());
  auto* const destination_storage = destination.storage().control_address();
  auto const* const destination_queue = &destination.queue();

  async_assign(destination, source);

  EXPECT_EQ(destination.storage().control_address(), destination_storage);
  EXPECT_EQ(&destination.queue(), destination_queue);

  sched.run_all();
  EXPECT_EQ(destination_parent.get_wait(), 8);
  EXPECT_EQ(source_parent.get_wait(), 8);
}
