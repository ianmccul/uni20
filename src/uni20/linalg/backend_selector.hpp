#pragma once

/**
 * \file backend_selector.hpp
 * \ingroup linalg
 * \brief Ordered backend selector values shared by tensor storage and linalg dispatch.
 */

#include <concepts>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{

/// \brief Backend value for BLAS dense linalg kernels.
struct BlasBackend
{
    static constexpr std::string_view name = "blas";
    friend constexpr bool operator==(BlasBackend const&, BlasBackend const&) = default;
};

/// \brief Backend value for the generic host CPU dense linalg oracle.
struct CpuGenericBackend
{
    static constexpr std::string_view name = "cpu_generic";
    friend constexpr bool operator==(CpuGenericBackend const&, CpuGenericBackend const&) = default;
};

/// \brief Ordered list of backend values tried by the dispatch walk.
template <class... Backends> struct backend_list
{
    std::tuple<Backends...> entries;

    constexpr explicit backend_list(Backends... backends) : entries(std::move(backends)...) {}

    friend constexpr bool operator==(backend_list const&, backend_list const&) = default;
};

template <class... Backends> backend_list(Backends...) -> backend_list<std::remove_cvref_t<Backends>...>;

template <class T> struct is_backend_list : std::false_type
{};

template <class... Backends> struct is_backend_list<backend_list<Backends...>> : std::true_type
{};

template <class T> inline constexpr bool is_backend_list_v = is_backend_list<std::remove_cvref_t<T>>::value;

/// \brief Optional global backend-selector override for an operation and storage policy.
/// \details Specializations may define a static `select(operation, operands...)`
///          function. When no such function is available, tensor dispatch uses
///          the storage policy's selector exposed by the first operand.
template <class Operation, class StoragePolicy> struct backend_selector_override
{};

namespace detail
{
template <class Tensor>
concept HasStoragePolicy = requires { typename std::remove_cvref_t<Tensor>::storage_policy; };

template <class FirstTensor, class... RestTensors>
[[nodiscard]] constexpr auto storage_default_backend_selector(FirstTensor const& first, RestTensors const&... rest)
{
  using selector_type = std::remove_cvref_t<decltype(first.backend_selector())>;
  static_assert((std::same_as<selector_type, std::remove_cvref_t<decltype(rest.backend_selector())>> && ...),
                "tensor operands must provide compatible default backend selector types");
  return first.backend_selector();
}
} // namespace detail

/// \brief Select the default backend list for an operation on tensor operands.
/// \details Operands with storage policies must use one common policy. A global
///          `backend_selector_override<Operation, StoragePolicy>` specialization
///          may replace the storage-provided default. Tensor adaptors without a
///          storage-policy type fall back directly to their member selectors.
template <class Operation, class FirstTensor, class... RestTensors>
[[nodiscard]] constexpr auto select_backend(Operation const& operation, FirstTensor const& first,
                                            RestTensors const&... rest)
{
  if constexpr (detail::HasStoragePolicy<FirstTensor> && (detail::HasStoragePolicy<RestTensors> && ...))
  {
    using storage_policy = typename std::remove_cvref_t<FirstTensor>::storage_policy;
    static_assert((std::same_as<storage_policy, typename std::remove_cvref_t<RestTensors>::storage_policy> && ...),
                  "tensor operands with different storage policies require an explicit selector or transfer");
    using override_type = backend_selector_override<std::remove_cvref_t<Operation>, storage_policy>;
    if constexpr (requires { override_type::select(operation, first, rest...); })
    {
      return override_type::select(operation, first, rest...);
    }
    else
    {
      return detail::storage_default_backend_selector(first, rest...);
    }
  }
  else
  {
    return detail::storage_default_backend_selector(first, rest...);
  }
}

/// \brief Return an explicit backend list unchanged.
template <class... Backends>
[[nodiscard]] constexpr auto normalize_backend_selector(backend_list<Backends...> selector) -> backend_list<Backends...>
{
  return selector;
}

/// \brief Treat a single backend value as a one-entry backend list.
template <class Backend>
  requires(!is_backend_list_v<Backend>)
[[nodiscard]] constexpr auto normalize_backend_selector(Backend&& backend)
{
  return backend_list{std::forward<Backend>(backend)};
}

} // namespace uni20::linalg
