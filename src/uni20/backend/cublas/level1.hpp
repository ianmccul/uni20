#pragma once

/**
 * \file level1.hpp
 * \ingroup backend_cublas
 * \brief Checked cuBLAS dot-product and Euclidean-norm wrappers.
 */

#include <uni20/backend/backend.hpp>
#include <uni20/backend/cublas/cublas_error.hpp>
#include <uni20/backend/cublas/execution.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/core/types.hpp>

#include <cuComplex.h>
#include <cublas_v2.h>

#include <concepts>
#include <type_traits>

namespace uni20::cublas
{

/// \brief Scalar types supported by the cuBLAS level-one wrappers.
template <class Scalar>
concept CublasLevelOneScalar = std::same_as<Scalar, float> || std::same_as<Scalar, double> ||
                               std::same_as<Scalar, uni20::cfloat> || std::same_as<Scalar, uni20::cdouble>;

namespace detail
{

template <class Scalar> struct ProviderComplex;

template <> struct ProviderComplex<uni20::cfloat>
{
    using type = cuComplex;
};

template <> struct ProviderComplex<uni20::cdouble>
{
    using type = cuDoubleComplex;
};

template <class Scalar> using provider_complex_t = typename ProviderComplex<Scalar>::type;

template <class Scalar> consteval void require_complex_provider_abi()
{
  using provider_type = provider_complex_t<Scalar>;
  static_assert(sizeof(Scalar) == sizeof(provider_type));
  static_assert(std::is_trivially_copyable_v<Scalar>);
  static_assert(std::is_trivially_copyable_v<provider_type>);
}

} // namespace detail

/// \brief Compute the conjugate-linear dot product and return a host scalar.
/// \details The provider call uses host pointer mode. The operation stream is
///          synchronized before the host result and any borrowed input storage
///          may leave scope.
template <CublasLevelOneScalar Scalar>
[[nodiscard]] auto dotc(ExecutionLease& execution, int size, Scalar const* lhs, Scalar const* rhs) -> Scalar
{
  CHECK(execution);
  CHECK(size >= 0, size);
  if (size == 0) return Scalar{};
  CHECK(lhs != nullptr && rhs != nullptr);

  cuda::ScopedDevice guard(execution.stream().device());
  cublasStatus_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  if constexpr (std::same_as<Scalar, float>)
  {
    Scalar result{};
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasSdot, size);
    status = cublasSdot(execution.handle().native_handle(), size, lhs, 1, rhs, 1, &result);
    check(status, "cublasSdot", execution.stream().device());
    execution.stream().synchronize();
    return result;
  }
  else if constexpr (std::same_as<Scalar, double>)
  {
    Scalar result{};
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasDdot, size);
    status = cublasDdot(execution.handle().native_handle(), size, lhs, 1, rhs, 1, &result);
    check(status, "cublasDdot", execution.stream().device());
    execution.stream().synchronize();
    return result;
  }
  else if constexpr (std::same_as<Scalar, uni20::cfloat>)
  {
    detail::require_complex_provider_abi<Scalar>();
    cuComplex result{};
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasCdotc, size);
    status = cublasCdotc(execution.handle().native_handle(), size, reinterpret_cast<cuComplex const*>(lhs), 1,
                         reinterpret_cast<cuComplex const*>(rhs), 1, &result);
    check(status, "cublasCdotc", execution.stream().device());
    execution.stream().synchronize();
    return Scalar{result.x, result.y};
  }
  else
  {
    static_assert(std::same_as<Scalar, uni20::cdouble>);
    detail::require_complex_provider_abi<Scalar>();
    cuDoubleComplex result{};
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasZdotc, size);
    status = cublasZdotc(execution.handle().native_handle(), size, reinterpret_cast<cuDoubleComplex const*>(lhs), 1,
                         reinterpret_cast<cuDoubleComplex const*>(rhs), 1, &result);
    check(status, "cublasZdotc", execution.stream().device());
    execution.stream().synchronize();
    return Scalar{result.x, result.y};
  }
}

/// \brief Compute a Euclidean norm and return a host real scalar.
/// \details The operation stream is synchronized before the result returns.
template <CublasLevelOneScalar Scalar>
[[nodiscard]] auto nrm2(ExecutionLease& execution, int size, Scalar const* input) -> uni20::make_real_t<Scalar>
{
  CHECK(execution);
  CHECK(size >= 0, size);
  using result_type = uni20::make_real_t<Scalar>;
  if (size == 0) return result_type{};
  CHECK(input != nullptr);

  cuda::ScopedDevice guard(execution.stream().device());
  result_type result{};
  cublasStatus_t status = CUBLAS_STATUS_INTERNAL_ERROR;
  if constexpr (std::same_as<Scalar, float>)
  {
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasSnrm2, size);
    status = cublasSnrm2(execution.handle().native_handle(), size, input, 1, &result);
    check(status, "cublasSnrm2", execution.stream().device());
  }
  else if constexpr (std::same_as<Scalar, double>)
  {
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasDnrm2, size);
    status = cublasDnrm2(execution.handle().native_handle(), size, input, 1, &result);
    check(status, "cublasDnrm2", execution.stream().device());
  }
  else if constexpr (std::same_as<Scalar, uni20::cfloat>)
  {
    detail::require_complex_provider_abi<Scalar>();
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasScnrm2, size);
    status =
        cublasScnrm2(execution.handle().native_handle(), size, reinterpret_cast<cuComplex const*>(input), 1, &result);
    check(status, "cublasScnrm2", execution.stream().device());
  }
  else
  {
    static_assert(std::same_as<Scalar, uni20::cdouble>);
    detail::require_complex_provider_abi<Scalar>();
    UNI20_EXTERNAL_API_CALL(CUBLAS, cublasDznrm2, size);
    status = cublasDznrm2(execution.handle().native_handle(), size, reinterpret_cast<cuDoubleComplex const*>(input), 1,
                          &result);
    check(status, "cublasDznrm2", execution.stream().device());
  }
  execution.stream().synchronize();
  return result;
}

} // namespace uni20::cublas
