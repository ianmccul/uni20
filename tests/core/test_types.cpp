#include "core/types.hpp"
#include <gtest/gtest.h>

// Simulated proxy type
template <typename T> struct MyProxy
{
    T value;
};

// Specialization of the customization point uni20::remove_proxy_reference for MyProxy
template <typename T> struct uni20::remove_proxy_reference<MyProxy<T>>
{
    using type = T;
};

TEST(RemoveProxyReferenceTest, ProxyAndNonProxyCases)
{
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int>>, int>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int>&>, int>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int> const>, int>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int> const&>, int>);

  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int const>>, int const>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int const>&>, int const>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int const> const>, int const>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<MyProxy<int const> const&>, int const>);

  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<int>, int>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<int&>, int>);
  static_assert(std::is_same_v<uni20::remove_proxy_reference_t<int const&>, int const>);

  static_assert(uni20::is_proxy_v<MyProxy<int>>);
  static_assert(uni20::is_proxy_v<MyProxy<int> const&>);
  static_assert(!uni20::is_proxy_v<int>);
  static_assert(!uni20::is_proxy_v<int&>);
}
