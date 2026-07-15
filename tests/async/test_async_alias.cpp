#include <gtest/gtest.h>
#include <memory>
#include <type_traits>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <utility>

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

    explicit MutableIntAlias(int* value_in) : value(value_in) {}

    [[nodiscard]] int get() const { return *value; }

    int* value;
};

void assign_through(MutableIntAlias& target, MutableIntAlias const& source) { *target.value = *source.value; }

void assign_through(MutableIntAlias& target, int const& source) { *target.value = source; }

struct CopyConstructOnly
{
    explicit CopyConstructOnly(int value_in) : value(value_in) {}
    CopyConstructOnly(CopyConstructOnly const&) = default;
    CopyConstructOnly(CopyConstructOnly&&) = delete;
    CopyConstructOnly& operator=(CopyConstructOnly const&) = delete;
    CopyConstructOnly& operator=(CopyConstructOnly&&) = delete;

    int value;
};

struct MoveOnly
{
    explicit MoveOnly(int value_in) : value(value_in) {}
    MoveOnly(MoveOnly const&) = delete;
    MoveOnly(MoveOnly&&) = default;
    MoveOnly& operator=(MoveOnly const&) = delete;
    MoveOnly& operator=(MoveOnly&&) = default;

    int value;
};

template <typename Proxy, typename Source>
concept proxy_assignable_from = requires(Proxy& proxy, Source&& source) { proxy = std::forward<Source>(source); };

template <typename Proxy>
concept proxy_can_emplace_int = requires(Proxy& proxy) { proxy.emplace(1); };

template <typename Buffer>
concept buffer_can_take = requires(Buffer& buffer) { buffer.take(); };

template <typename Buffer>
concept buffer_can_access_storage = requires(Buffer& buffer) { buffer.storage(); };

template <typename Buffer, typename Source>
concept buffer_can_schedule_write =
    requires(Buffer& buffer, Source&& source) { buffer.write(std::forward<Source>(source)); };

using read_only_alias_proxy = WriteAccessProxy<IntAlias>;
using mutable_alias_proxy = WriteAccessProxy<MutableIntAlias>;
using owning_read_only_alias_proxy = OwningWriteAccessProxy<IntAlias>;
using owning_mutable_alias_proxy = OwningWriteAccessProxy<MutableIntAlias>;

static_assert(!is_async_alias_v<int>);
static_assert(is_async_alias_v<IntAlias>);
static_assert(is_async_alias_v<MutableIntAlias>);
static_assert(std::is_copy_constructible_v<Async<IntAlias>>);
static_assert(std::is_move_constructible_v<Async<IntAlias>>);
static_assert(!std::is_copy_assignable_v<Async<IntAlias>>);
static_assert(!std::is_move_assignable_v<Async<IntAlias>>);
static_assert(!std::is_assignable_v<Async<IntAlias>&, int>);
static_assert(!std::is_assignable_v<Async<IntAlias>&, Async<int> const&>);
static_assert(std::is_copy_assignable_v<Async<MutableIntAlias>>);
static_assert(std::is_move_assignable_v<Async<MutableIntAlias>>);
static_assert(std::is_assignable_v<Async<MutableIntAlias>&, int>);
static_assert(std::is_assignable_v<Async<MutableIntAlias>&, Async<int> const&>);
static_assert(async_reader<Async<IntAlias>>);
static_assert(!async_writer<Async<IntAlias>>);
static_assert(!async_writer<Async<MutableIntAlias>>);
static_assert(std::same_as<decltype(std::declval<Async<IntAlias>&>().write()), WriteBuffer<IntAlias>>);
static_assert(std::same_as<decltype(std::declval<Async<MutableIntAlias>&>().write()), WriteBuffer<MutableIntAlias>>);
static_assert(!proxy_assignable_from<read_only_alias_proxy, IntAlias const&>);
static_assert(proxy_assignable_from<mutable_alias_proxy, int const&>);
static_assert(proxy_assignable_from<mutable_alias_proxy, MutableIntAlias const&>);
static_assert(!proxy_can_emplace_int<mutable_alias_proxy>);
static_assert(!proxy_assignable_from<owning_read_only_alias_proxy, IntAlias const&>);
static_assert(proxy_assignable_from<owning_mutable_alias_proxy, int const&>);
static_assert(proxy_assignable_from<owning_mutable_alias_proxy, MutableIntAlias const&>);
static_assert(!proxy_can_emplace_int<owning_mutable_alias_proxy>);
static_assert(!buffer_can_take<WriteBuffer<MutableIntAlias>>);
static_assert(!buffer_can_access_storage<WriteBuffer<MutableIntAlias>>);
static_assert(!buffer_can_schedule_write<WriteBuffer<IntAlias>, int>);
static_assert(buffer_can_schedule_write<WriteBuffer<MutableIntAlias>, int>);
static_assert(buffer_can_schedule_write<WriteBuffer<MutableIntAlias>, Async<int> const&>);
static_assert(!std::is_assignable_v<WriteAssignProxy<IntAlias>&, int>);
static_assert(std::is_assignable_v<WriteAssignProxy<MutableIntAlias>&, int>);
static_assert(std::is_assignable_v<WriteAssignProxy<MutableIntAlias>&, Async<int> const&>);
static_assert(std::same_as<decltype(std::declval<mutable_alias_proxy const&>().get()), MutableIntAlias const&>);
static_assert(std::same_as<decltype(std::declval<owning_mutable_alias_proxy const&>().get()), MutableIntAlias const&>);
static_assert(std::same_as<decltype(std::declval<Async<IntAlias>&>().storage_address()), IntAlias const*>);
static_assert(std::same_as<decltype(std::declval<Async<IntAlias>&>().value_ptr()), std::shared_ptr<IntAlias const>>);
static_assert(
    std::same_as<decltype(std::declval<Async<MutableIntAlias>&>().storage_address()), MutableIntAlias const*>);
static_assert(std::same_as<decltype(std::declval<Async<MutableIntAlias>&>().value_ptr()),
                           std::shared_ptr<MutableIntAlias const>>);
static_assert(std::is_copy_constructible_v<Async<CopyConstructOnly>>);
static_assert(std::is_copy_assignable_v<Async<CopyConstructOnly>>);
static_assert(std::is_move_constructible_v<Async<CopyConstructOnly>>);
static_assert(std::is_move_assignable_v<Async<CopyConstructOnly>>);
static_assert(!std::is_copy_constructible_v<Async<MoveOnly>>);
static_assert(!std::is_copy_assignable_v<Async<MoveOnly>>);
static_assert(std::is_move_constructible_v<Async<MoveOnly>>);
static_assert(std::is_move_assignable_v<Async<MoveOnly>>);

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

TEST(AsyncAliasTest, CapabilityAwareWriteProxyAssignsThroughWithoutExposingDescriptorMutation)
{
  Async<int> parent = 5;
  auto alias = make_async_alias<MutableIntAlias>(parent, parent.storage_address());
  auto* const alias_storage = alias.storage().control_address();
  auto const* const alias_queue = &alias.queue();
  DebugScheduler sched;

  sched.schedule([](WriteBuffer<MutableIntAlias> writer) static -> AsyncTask {
    co_await writer = 23;
    co_return;
  }(alias.write()));

  sched.run_all();
  EXPECT_EQ(parent.get_wait(), 23);
  EXPECT_EQ(alias.storage().control_address(), alias_storage);
  EXPECT_EQ(&alias.queue(), alias_queue);
}

TEST(AsyncAliasTest, FreshTimelineCopyNeedsConstructionButNotPayloadAssignment)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<CopyConstructOnly> source(17);
  Async<CopyConstructOnly> copy(source);
  Async<CopyConstructOnly> assigned(3);
  assigned = source;

  sched.run_all();
  EXPECT_EQ(copy.get_wait().value, 17);
  EXPECT_EQ(assigned.get_wait().value, 17);
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

TEST(AsyncAliasTest, ExactMutableAliasAssignmentWritesThrough)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> first_parent = 3;
  Async<int> second_parent = 8;
  auto alias = make_async_alias<MutableIntAlias>(first_parent, first_parent.storage_address());
  auto replacement = make_async_alias<MutableIntAlias>(second_parent, second_parent.storage_address());
  auto* const alias_storage = alias.storage().control_address();
  auto const* const alias_queue = &alias.queue();
  int old_observation = 0;
  int new_observation = 0;

  schedule([](ReadBuffer<MutableIntAlias> reader, int& observation) static -> AsyncTask {
    observation = (co_await reader).get();
    co_return;
  }(alias.read(), old_observation));

  alias = replacement;

  EXPECT_EQ(alias.storage().control_address(), alias_storage);
  EXPECT_EQ(&alias.queue(), alias_queue);

  schedule([](ReadBuffer<MutableIntAlias> reader, int& observation) static -> AsyncTask {
    observation = (co_await reader).get();
    co_return;
  }(alias.read(), new_observation));

  sched.run_all();
  EXPECT_EQ(old_observation, 3);
  EXPECT_EQ(new_observation, 8);
  EXPECT_EQ(first_parent.get_wait(), 8);
  EXPECT_EQ(second_parent.get_wait(), 8);
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
