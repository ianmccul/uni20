#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <uni20/async/shared_storage.hpp>
#include <utility>
#include <vector>

using namespace uni20::async;

namespace
{
struct Counting
{
    static void reset()
    {
      constructions = 0;
      destructions = 0;
    }

    explicit Counting(int value_in) : value(value_in) { ++constructions; }
    ~Counting() { ++destructions; }

    int value;
    inline static int constructions = 0;
    inline static int destructions = 0;
};

struct CountedDefaultConstructible
{
    static void reset()
    {
      default_constructions = 0;
      value_constructions = 0;
      destructions = 0;
    }

    CountedDefaultConstructible() : value(7) { ++default_constructions; }

    explicit CountedDefaultConstructible(int value_in) : value(value_in) { ++value_constructions; }

    ~CountedDefaultConstructible() { ++destructions; }

    int value = 7;
    inline static int default_constructions = 0;
    inline static int value_constructions = 0;
    inline static int destructions = 0;
};

struct LifetimeOwner
{
    explicit LifetimeOwner(std::shared_ptr<std::vector<std::string>> events_in) : events(std::move(events_in)) {}

    ~LifetimeOwner() { events->push_back("destroy owner"); }

    std::shared_ptr<std::vector<std::string>> events;
    int value = 17;
};

struct LifetimeAlias
{
    LifetimeAlias(std::shared_ptr<std::vector<std::string>> events_in, LifetimeOwner const* owner_in)
        : events(std::move(events_in)), owner(owner_in)
    {}

    ~LifetimeAlias() { events->push_back("destroy alias"); }

    [[nodiscard]] int value() const { return owner->value; }

    std::shared_ptr<std::vector<std::string>> events;
    LifetimeOwner const* owner;
};
} // namespace

TEST(SharedStorageTest, DefaultConstructionEmplaceAndRefCount)
{
  Counting::reset();

  {
    auto storage = make_unconstructed_shared_storage<Counting>();
    EXPECT_TRUE(storage.valid());
    EXPECT_FALSE(storage.constructed());
    EXPECT_EQ(storage.use_count(), 1); // initial control block ref
    EXPECT_EQ(storage.get(), nullptr);

    auto& obj = storage.emplace(7);
    EXPECT_TRUE(storage.constructed());
    EXPECT_EQ(obj.value, 7);
    EXPECT_EQ(Counting::constructions, 1);

    auto copy = storage;
    EXPECT_EQ(storage.use_count(), 2);
    EXPECT_EQ(copy.use_count(), 2);

    auto moved = std::move(copy);
    EXPECT_FALSE(copy.valid());
    EXPECT_TRUE(moved.constructed());
    EXPECT_EQ(moved.use_count(), 2);
  }

  EXPECT_EQ(Counting::constructions, 1);
  EXPECT_EQ(Counting::destructions, 1);
}

TEST(SharedStorageTest, DestroyAllowsReemplace)
{
  Counting::reset();

  {
    auto storage = make_shared_storage<Counting>(10);
    EXPECT_TRUE(storage.constructed());
    EXPECT_EQ((*storage).value, 10);
    EXPECT_EQ(Counting::constructions, 1);

    storage.destroy();
    EXPECT_FALSE(storage.constructed());
    EXPECT_EQ(Counting::destructions, 1);

    auto& rebuilt = storage.emplace(25);
    EXPECT_TRUE(storage.constructed());
    EXPECT_EQ(rebuilt.value, 25);
    EXPECT_EQ(Counting::constructions, 2);
  }

  EXPECT_EQ(Counting::destructions, 2);
}

TEST(SharedStorageTest, UnconstructedStorageDoesNotDefaultConstructDefaultConstructibleType)
{
  CountedDefaultConstructible::reset();

  {
    auto storage = make_unconstructed_shared_storage<CountedDefaultConstructible>();
    EXPECT_TRUE(storage.valid());
    EXPECT_FALSE(storage.constructed());
    EXPECT_EQ(storage.get(), nullptr);
    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);

    auto& obj = storage.emplace(11);
    EXPECT_TRUE(storage.constructed());
    EXPECT_EQ(obj.value, 11);
    EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 1);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 0);
  }

  EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);
  EXPECT_EQ(CountedDefaultConstructible::value_constructions, 1);
  EXPECT_EQ(CountedDefaultConstructible::destructions, 1);
}

TEST(SharedStorageTest, EmplaceReplacesExistingObject)
{
  CountedDefaultConstructible::reset();

  {
    auto storage = make_unconstructed_shared_storage<CountedDefaultConstructible>();

    auto& first = storage.emplace(1);
    EXPECT_EQ(first.value, 1);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 1);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 0);

    auto& second = storage.emplace(2);
    EXPECT_EQ(second.value, 2);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 2);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 1);

    auto& third = storage.emplace(3);
    EXPECT_EQ(third.value, 3);
    EXPECT_EQ(CountedDefaultConstructible::value_constructions, 3);
    EXPECT_EQ(CountedDefaultConstructible::destructions, 2);
  }

  EXPECT_EQ(CountedDefaultConstructible::default_constructions, 0);
  EXPECT_EQ(CountedDefaultConstructible::value_constructions, 3);
  EXPECT_EQ(CountedDefaultConstructible::destructions, 3);
}

TEST(SharedStorageTest, PreconstructedStorageSupportsConstAccess)
{
  Counting::reset();

  {
    const auto storage = make_shared_storage<Counting>(5);
    EXPECT_TRUE(storage.constructed());
    EXPECT_EQ(storage.use_count(), 1);
    EXPECT_EQ(storage->value, 5);
  }

  EXPECT_EQ(Counting::constructions, 1);
  EXPECT_EQ(Counting::destructions, 1);
}

TEST(SharedStorageTest, AliasUsesShardedReferenceCount)
{
  auto events = std::make_shared<std::vector<std::string>>();
  auto owner = make_shared_storage<LifetimeOwner>(events);

  {
    auto alias = make_shared_storage_alias<LifetimeAlias>(owner, events, owner.storage_address());
    EXPECT_TRUE(alias.has_lifetime_owner());
    EXPECT_EQ(owner.use_count(), 2);
    EXPECT_EQ(alias.use_count(), 1);

    {
      auto alias_copy = alias;
      EXPECT_EQ(alias.use_count(), 2);
      EXPECT_EQ(alias_copy.use_count(), 2);
      EXPECT_EQ(owner.use_count(), 2);
    }

    EXPECT_EQ(alias.use_count(), 1);
    owner.reset();
    EXPECT_TRUE(alias.constructed());
    EXPECT_EQ(alias->value(), 17);
    EXPECT_TRUE(events->empty());
  }

  EXPECT_EQ(*events, (std::vector<std::string>{"destroy alias", "destroy owner"}));
}

TEST(SharedStorageTest, AliasReadinessTracksUnconstructedOwner)
{
  auto owner = make_unconstructed_shared_storage<int>();
  auto alias = make_shared_storage_alias<int const*>(owner, owner.storage_address());

  EXPECT_FALSE(owner.constructed());
  EXPECT_FALSE(alias.constructed());
  EXPECT_EQ(alias.get(), nullptr);

  owner.emplace(23);
  ASSERT_TRUE(alias.constructed());
  ASSERT_NE(alias.get(), nullptr);
  EXPECT_EQ(**alias, 23);

  owner.destroy();
  EXPECT_FALSE(alias.constructed());
  EXPECT_EQ(alias.get(), nullptr);
}
