#include <uni20/linalg/backend_selector.hpp>

#include <gtest/gtest.h>

#include <string_view>
#include <type_traits>
#include <utility>

namespace backend_selector_test
{

using uni20::linalg::backend_list;

struct StorageBackend
{
    static constexpr std::string_view name = "storage";
    int configuration = 0;

    friend constexpr bool operator==(StorageBackend const&, StorageBackend const&) = default;
};

struct LibraryBackend
{
    static constexpr std::string_view name = "library_default";
    backend_list<StorageBackend> execution;

    friend constexpr bool operator==(LibraryBackend const&, LibraryBackend const&) = default;
};

struct UserBackend
{
    static constexpr std::string_view name = "user_override";
    friend constexpr bool operator==(UserBackend const&, UserBackend const&) = default;
};

struct StoragePolicy
{
    [[nodiscard]] static constexpr auto backend_selector() { return backend_list{StorageBackend{.configuration = 17}}; }
};

struct NeutralStoragePolicy
{
    [[nodiscard]] static constexpr auto backend_selector() { return backend_list{StorageBackend{.configuration = 23}}; }
};

template <class Policy> struct Tensor
{
    using storage_policy = Policy;
};

struct GeneralOperation
{};

struct DefaultedOperation
{};

struct OverriddenOperation
{};

struct NeutralDefaultedOperation
{};

} // namespace backend_selector_test

template <>
struct uni20::linalg::backend_selector_default<backend_selector_test::DefaultedOperation,
                                               backend_selector_test::StoragePolicy>
{
    static constexpr auto select(backend_selector_test::DefaultedOperation const&,
                                 backend_list<backend_selector_test::StorageBackend> execution)
    {
      return backend_list{backend_selector_test::LibraryBackend{.execution = std::move(execution)}};
    }
};

template <>
struct uni20::linalg::backend_selector_default<backend_selector_test::OverriddenOperation,
                                               backend_selector_test::StoragePolicy>
{
    static constexpr auto select(backend_selector_test::OverriddenOperation const&,
                                 backend_list<backend_selector_test::StorageBackend> execution)
    {
      return backend_list{backend_selector_test::LibraryBackend{.execution = std::move(execution)}};
    }
};

template <>
struct uni20::linalg::backend_selector_override<backend_selector_test::OverriddenOperation,
                                                backend_selector_test::StoragePolicy>
{
    static constexpr auto select(backend_selector_test::OverriddenOperation const&)
    {
      return backend_list{backend_selector_test::UserBackend{}};
    }
};

template <>
struct uni20::linalg::backend_selector_default<backend_selector_test::NeutralDefaultedOperation,
                                               backend_selector_test::NeutralStoragePolicy>
{
    static constexpr auto select(backend_selector_test::NeutralDefaultedOperation const&,
                                 backend_list<backend_selector_test::StorageBackend> execution)
    {
      return backend_list{backend_selector_test::LibraryBackend{.execution = std::move(execution)}};
    }
};

template <>
inline constexpr bool uni20::linalg::enable_backend_neutral_storage<backend_selector_test::NeutralStoragePolicy> = true;

namespace
{

using namespace backend_selector_test;

TEST(BackendSelectorTest, GeneralOperationUsesStorageSelector)
{
  auto selector = uni20::linalg::select_backend_for<Tensor<StoragePolicy>>(GeneralOperation{});

  static_assert(std::same_as<decltype(selector), backend_list<StorageBackend>>);
  EXPECT_EQ(selector, backend_list{StorageBackend{.configuration = 17}});
}

TEST(BackendSelectorTest, LibraryDefaultComposesStorageSelector)
{
  auto selector = uni20::linalg::select_backend_for<Tensor<StoragePolicy>>(DefaultedOperation{});

  static_assert(std::same_as<decltype(selector), backend_list<LibraryBackend>>);
  EXPECT_EQ(selector, backend_list{LibraryBackend{.execution = backend_list{StorageBackend{.configuration = 17}}}});
}

TEST(BackendSelectorTest, UserOverrideCompletelyReplacesLibraryDefault)
{
  auto selector = uni20::linalg::select_backend_for<Tensor<StoragePolicy>>(OverriddenOperation{});

  static_assert(std::same_as<decltype(selector), backend_list<UserBackend>>);
  EXPECT_EQ(selector, backend_list{UserBackend{}});
}

TEST(BackendSelectorTest, LibraryDefaultAlsoAppliesToBackendNeutralStorage)
{
  auto selector = uni20::linalg::select_backend_for<Tensor<NeutralStoragePolicy>>(NeutralDefaultedOperation{});

  static_assert(std::same_as<decltype(selector), backend_list<LibraryBackend>>);
  EXPECT_EQ(selector, backend_list{LibraryBackend{.execution = backend_list{StorageBackend{.configuration = 23}}}});
}

} // namespace
