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

/// \brief Backend value for configured dense LAPACK kernels.
struct LapackBackend
{
    static constexpr std::string_view name = "lapack";
    friend constexpr bool operator==(LapackBackend const&, LapackBackend const&) = default;
};

/// \brief Backend value for BLAS dense linalg kernels.
struct BlasBackend
{
    static constexpr std::string_view name = "blas";
    friend constexpr bool operator==(BlasBackend const&, BlasBackend const&) = default;
};

/// \brief Backend value for cuBLAS dense linalg kernels.
struct CublasBackend
{
    static constexpr std::string_view name = "cublas";
    friend constexpr bool operator==(CublasBackend const&, CublasBackend const&) = default;
};

/// \brief Backend value for generic CUDA runtime kernels and transfers.
struct CudaReferenceBackend
{
    static constexpr std::string_view name = "cuda_reference";
    friend constexpr bool operator==(CudaReferenceBackend const&, CudaReferenceBackend const&) = default;
};

/// \brief Backend value for the reference host CPU dense linalg oracle.
struct CpuReferenceBackend
{
    static constexpr std::string_view name = "cpu_reference";
    friend constexpr bool operator==(CpuReferenceBackend const&, CpuReferenceBackend const&) = default;
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

/// \brief Named backend value accepted as one dispatch candidate.
template <class T>
concept KernelBackend = requires {
  { std::remove_cvref_t<T>::name } -> std::convertible_to<std::string_view>;
};

/// \brief Single named backend or ordered backend list accepted by dispatch front ends.
template <class T>
concept KernelBackendSelector = KernelBackend<T> || is_backend_list_v<T>;

/// \brief Optional global backend-selector override for an operation and storage policy.
/// \details Specializations may define a static `select(operation)` function.
///          When no such function is available, Tensor dispatch uses the
///          storage policy's static selector. Selector resolution does not
///          inspect Tensor values.
template <class Operation, class StoragePolicy> struct backend_selector_override
{};

/// \brief Opt-in marker for storage policies that do not constrain backend selection.
/// \details Backend-neutral operands may be evaluated by any backend that
///          accepts their resolved accessor. When concrete storage operands
///          are also present, their common storage policy selects the default
///          backend list.
template <class StoragePolicy> inline constexpr bool enable_backend_neutral_storage = false;

namespace detail
{
template <class Tensor>
concept HasStoragePolicy = requires { typename std::remove_cvref_t<Tensor>::storage_policy; } &&
                           (!std::same_as<typename std::remove_cvref_t<Tensor>::storage_policy, void>);

template <class Provider>
concept HasStaticBackendSelector = requires { std::remove_cvref_t<Provider>::backend_selector(); };

template <class Tensor> using tensor_storage_policy_t = typename std::remove_cvref_t<Tensor>::storage_policy;

template <class... StoragePolicies> struct FirstBackendBoundStorage
{
    using type = void;
};

template <class StoragePolicy, class... Rest> struct FirstBackendBoundStorage<StoragePolicy, Rest...>
{
    using type = std::conditional_t<enable_backend_neutral_storage<StoragePolicy>,
                                    typename FirstBackendBoundStorage<Rest...>::type, StoragePolicy>;
};

template <class... Tensors>
using first_backend_bound_storage_t = typename FirstBackendBoundStorage<tensor_storage_policy_t<Tensors>...>::type;

template <class StoragePolicy, class... Tensors>
inline constexpr bool tensors_have_compatible_storage =
    ((enable_backend_neutral_storage<tensor_storage_policy_t<Tensors>> ||
      std::same_as<StoragePolicy, tensor_storage_policy_t<Tensors>>) &&
     ...);
} // namespace detail

/// \brief Resolve the immutable backend selector for Tensor operand types.
/// \details Backend-bound operands with storage policies must use one common
///          policy; backend-neutral storage operands are ignored when finding
///          that policy. A global
///          `backend_selector_override<Operation, StoragePolicy>` specialization
///          may replace the storage-provided static default. Tensor adaptors
///          without storage policies must expose a static `backend_selector()`.
template <class FirstTensor, class... RestTensors, class Operation>
[[nodiscard]] constexpr auto select_backend_for(Operation const& operation)
{
  if constexpr (detail::HasStoragePolicy<FirstTensor> && (detail::HasStoragePolicy<RestTensors> && ...))
  {
    using storage_policy = detail::first_backend_bound_storage_t<FirstTensor, RestTensors...>;
    if constexpr (std::is_void_v<storage_policy>)
    {
      using first_storage = detail::tensor_storage_policy_t<FirstTensor>;
      static_assert(detail::HasStaticBackendSelector<first_storage> &&
                        (detail::HasStaticBackendSelector<detail::tensor_storage_policy_t<RestTensors>> && ...),
                    "backend-neutral storage policies must provide a static backend_selector()");
      using selector_type = std::remove_cvref_t<decltype(first_storage::backend_selector())>;
      static_assert(
          (std::same_as<
               selector_type,
               std::remove_cvref_t<decltype(detail::tensor_storage_policy_t<RestTensors>::backend_selector())>> &&
           ...),
          "backend-neutral storage policies must provide compatible default backend selector types");
      return first_storage::backend_selector();
    }
    else
    {
      static_assert(
          detail::tensors_have_compatible_storage<storage_policy, FirstTensor, RestTensors...>,
          "tensor operands with different backend-bound storage policies require an explicit selector or transfer");
      static_assert(detail::HasStaticBackendSelector<storage_policy>,
                    "tensor storage policy must provide a static backend_selector()");

      using override_type = backend_selector_override<std::remove_cvref_t<Operation>, storage_policy>;
      if constexpr (requires { override_type::select(operation); })
      {
        return override_type::select(operation);
      }
      else
      {
        return storage_policy::backend_selector();
      }
    }
  }
  else
  {
    using first_tensor = std::remove_cvref_t<FirstTensor>;
    static_assert(detail::HasStaticBackendSelector<first_tensor> &&
                      (detail::HasStaticBackendSelector<std::remove_cvref_t<RestTensors>> && ...),
                  "tensor operands without storage policies must provide a static backend_selector()");
    using selector_type = std::remove_cvref_t<decltype(first_tensor::backend_selector())>;
    static_assert((std::same_as<selector_type,
                                std::remove_cvref_t<decltype(std::remove_cvref_t<RestTensors>::backend_selector())>> &&
                   ...),
                  "tensor operands must provide compatible default backend selector types");
    return first_tensor::backend_selector();
  }
}

/// \brief Select the default backend list for an operation on Tensor operands.
/// \details Tensor values are accepted for operation-front-end ergonomics but
///          selector resolution depends only on their static types and the
///          operation value.
template <class Operation, class FirstTensor, class... RestTensors>
[[nodiscard]] constexpr auto select_backend(Operation const& operation, FirstTensor const&, RestTensors const&...)
{
  return select_backend_for<FirstTensor, RestTensors...>(operation);
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
