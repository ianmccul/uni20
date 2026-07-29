#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief BLAS backend adapter for operation-tag GEMM dispatch.
 */

#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/layout.hpp>
#include <uni20/tensor/output.hpp>

#include <concepts>
#include <type_traits>

namespace uni20::linalg
{
namespace detail::blas_backend
{

template <class OutputTensor> using output_mdspan_t = uni20::detail::host_write_tensor_mdspan_t<OutputTensor>;

template <class OutputTensor, class Scalar, class LhsTensor, class RhsTensor> consteval bool gemm_types_compatible()
{
  using output_span = output_mdspan_t<OutputTensor>;
  using lhs_span = uni20::detail::host_read_tensor_mdspan_t<LhsTensor>;
  using rhs_span = uni20::detail::host_read_tensor_mdspan_t<RhsTensor>;
  return requires(output_span& output, Scalar alpha, lhs_span& lhs, rhs_span& rhs) {
    { uni20::linalg::blas::try_gemm(output, alpha, lhs, rhs, alpha) } -> std::same_as<KernelAttempt>;
  };
}

template <class OutputTensor>
auto prepared_output_mapping(typename output_mdspan_t<OutputTensor>::extents_type const& extents) ->
    typename output_mdspan_t<OutputTensor>::mapping_type
{
  using output_span = output_mdspan_t<OutputTensor>;
  using storage_policy = uni20::detail::tensor_storage_policy_t<OutputTensor>;
  if constexpr (!std::is_void_v<storage_policy>)
  {
    if constexpr (requires { typename storage_policy::default_mapping_builder; })
    {
      using builder_type = typename storage_policy::default_mapping_builder;
      if constexpr (uni20::layout::mapping_builder_for<builder_type, typename output_span::layout_type,
                                                       typename output_span::extents_type>)
      {
        return builder_type{}(extents);
      }
    }
  }
  return uni20::layout::make_mapping<typename output_span::layout_type>(extents);
}

template <class OutputTensor>
concept PreparedOutputProbeAvailable =
    std::default_initializable<typename output_mdspan_t<OutputTensor>::data_handle_type> &&
    std::default_initializable<typename output_mdspan_t<OutputTensor>::accessor_type> &&
    requires(typename output_mdspan_t<OutputTensor>::extents_type const& extents) {
      {
        prepared_output_mapping<OutputTensor>(extents)
      } -> std::same_as<typename output_mdspan_t<OutputTensor>::mapping_type>;
      output_mdspan_t<OutputTensor>{typename output_mdspan_t<OutputTensor>::data_handle_type{},
                                    prepared_output_mapping<OutputTensor>(extents),
                                    typename output_mdspan_t<OutputTensor>::accessor_type{}};
    };

template <class OutputTensor>
  requires PreparedOutputProbeAvailable<OutputTensor>
auto prepared_output_probe(matrix_product_extents const& shape) -> output_mdspan_t<OutputTensor>
{
  using output_span = output_mdspan_t<OutputTensor>;
  auto const extents = uni20::detail::convert_tensor_extents<typename output_span::extents_type>(shape);
  return output_span{typename output_span::data_handle_type{}, prepared_output_mapping<OutputTensor>(extents),
                     typename output_span::accessor_type{}};
}

template <class OutputTensor, class Scalar, class LhsTensor, class RhsTensor>
KernelAttempt probe_assign_product(matrix_product_extents const& shape, LhsTensor const& lhs, RhsTensor const& rhs)
{
  if constexpr (!PreparedOutputProbeAvailable<OutputTensor>)
  {
    return KernelAttempt::unsupported_layout;
  }
  else
  {
    auto lhs_access = acquire_host_read_access(lhs);
    auto rhs_access = acquire_host_read_access(rhs);
    auto output_span = prepared_output_probe<OutputTensor>(shape);
    auto lhs_span = lhs_access.mdspan();
    auto rhs_span = rhs_access.mdspan();
    return uni20::linalg::blas::probe_gemm<Scalar>(output_span, lhs_span, rhs_span);
  }
}

} // namespace detail::blas_backend

/// \brief Report BLAS eligibility for DeviceTensorView GEMM operands.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
consteval auto kernel_accepts_types(BlasBackend const&, gemm_op const&, OutputTensor&, Scalar const&, LhsTensor&,
                                    RhsTensor&, Scalar const&)
{
  if constexpr (detail::blas_backend::gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Lower DeviceTensorView operands and invoke the BLAS GEMM adapter.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, OutputTensor& output, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto lhs_access = acquire_host_read_access(lhs);
  auto rhs_access = acquire_host_read_access(rhs);
  auto output_span = output_access.mdspan();
  auto lhs_span = lhs_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  return uni20::linalg::blas::try_gemm(output_span, alpha, lhs_span, rhs_span, beta);
}

/// \brief Report BLAS eligibility for replaceable-output tensor matrix products.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
consteval auto kernel_accepts_types(BlasBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsTensor&, RhsTensor&)
{
  if constexpr (detail::blas_backend::gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Probe and lower a replaceable-output tensor matrix product to BLAS GEMM.
/// \details A mismatched output is represented by prospective mdspan metadata
///          during the BLAS runtime probe. The real output is prepared only
///          after every instance, layout, and transform check succeeds.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
KernelAttempt try_kernel(BlasBackend backend, assign_product_op const&, OutputTensor& output, Scalar alpha,
                         LhsTensor const& lhs, RhsTensor const& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  if (uni20::detail::tensor_extents_equal(output.extents(), shape))
    return try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});

  if constexpr (!uni20::ResizableTensorOutput<OutputTensor>)
  {
    return KernelAttempt::unsupported_shape;
  }

  KernelAttempt const probe = detail::blas_backend::probe_assign_product<OutputTensor, Scalar>(shape, lhs, rhs);
  if (!kernel_attempt_succeeded(probe)) return probe;

  uni20::prepare_output(output, shape);
  KernelAttempt const attempt = try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
  CHECK(kernel_attempt_succeeded(attempt));
  return attempt;
}

/// \brief Report BLAS eligibility for a deferred Tensor matrix-product output.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor>
consteval auto kernel_accepts_types(BlasBackend const&, assign_product_op const&, async::shared_storage<OutputTensor>&,
                                    Scalar const&, LhsTensor&, RhsTensor&)
{
  if constexpr (detail::blas_backend::gemm_types_compatible<OutputTensor, Scalar, LhsTensor, RhsTensor>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Probe BLAS GEMM before constructing or resizing a deferred host output.
template <uni20::MutableRankedDeviceTensorView<2> OutputTensor, class Scalar,
          uni20::RankedDeviceTensorView<2> LhsTensor, uni20::RankedDeviceTensorView<2> RhsTensor>
  requires uni20::detail::HostWritableTensor<OutputTensor> && uni20::detail::HostReadableTensor<LhsTensor> &&
           uni20::detail::HostReadableTensor<RhsTensor> &&
           requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape) {
             uni20::prepare_output(storage, shape);
           }
KernelAttempt try_kernel(BlasBackend backend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsTensor const& lhs,
                         RhsTensor const& rhs)
{
  if (output_storage.constructed()) return try_kernel(backend, assign_product_op{}, *output_storage, alpha, lhs, rhs);

  auto const shape = detail::matrix_product_shape(lhs, rhs);
  KernelAttempt const probe = detail::blas_backend::probe_assign_product<OutputTensor, Scalar>(shape, lhs, rhs);
  if (!kernel_attempt_succeeded(probe)) return probe;

  auto& output = uni20::prepare_output(output_storage, shape);
  KernelAttempt const attempt = try_kernel(backend, gemm_op{}, output, alpha, lhs, rhs, Scalar{});
  CHECK(kernel_attempt_succeeded(attempt));
  return attempt;
}

} // namespace uni20::linalg
