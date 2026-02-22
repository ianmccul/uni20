#include "async/shared_storage.hpp"
#include <gtest/gtest.h>
#include <utility>

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
