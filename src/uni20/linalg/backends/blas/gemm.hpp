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

template <class OutputTensor>
using output_mdspec_t = std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>;

template <class OutputTensor> using output_mdspan_t = uni20::host_write_mdspan_t<output_mdspec_t<OutputTensor>>;

template <class OutputTensor, class LhsMdspan, class RhsMdspan>
concept HostAssignProductAccess = uni20::HostWritableMdspec<output_mdspec_t<OutputTensor>> &&
                                  uni20::HostReadableMdspec<LhsMdspan> && uni20::HostReadableMdspec<RhsMdspan>;

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
consteval bool gemm_mdspan_types_compatible()
{
  using output_span = uni20::host_write_mdspan_t<OutputMdspan>;
  using lhs_span = uni20::host_read_mdspan_t<LhsMdspan>;
  using rhs_span = uni20::host_read_mdspan_t<RhsMdspan>;
  return requires(output_span& output, Scalar alpha, lhs_span& lhs, rhs_span& rhs) {
    { uni20::linalg::blas::try_gemm(output, alpha, lhs, rhs, alpha) } -> std::same_as<KernelAttempt>;
  };
}

template <class OutputTensor, class Scalar, class LhsMdspan, class RhsMdspan>
consteval bool assign_product_types_compatible()
{
  using output_span = std::remove_cvref_t<decltype(uni20::mdspec_of(std::declval<OutputTensor&>()))>;
  return gemm_mdspan_types_compatible<output_span, Scalar, LhsMdspan, RhsMdspan>();
}

template <uni20::MutableRankedMdspecLike<2> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<LhsMdspan> &&
           uni20::HostReadableMdspec<RhsMdspan>
KernelAttempt try_gemm(OutputMdspan& output, Scalar alpha, LhsMdspan& lhs, RhsMdspan& rhs, Scalar beta)
{
  auto output_access = acquire_host_write_access(output);
  auto lhs_access = acquire_host_read_access(lhs);
  auto rhs_access = acquire_host_read_access(rhs);
  auto output_span = output_access.mdspan();
  auto lhs_span = lhs_access.mdspan();
  auto rhs_span = rhs_access.mdspan();
  return uni20::linalg::blas::try_gemm(output_span, alpha, lhs_span, rhs_span, beta);
}

template <class OutputTensor>
auto prepared_output_mapping(typename output_mdspan_t<OutputTensor>::extents_type const& extents) ->
    typename output_mdspan_t<OutputTensor>::mapping_type
{
  using output_span = output_mdspan_t<OutputTensor>;
  using storage_policy = uni20::tensor_storage_policy_t<OutputTensor>;
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
  auto const extents = uni20::convert_tensor_extents<typename output_span::extents_type>(shape);
  return output_span{typename output_span::data_handle_type{}, prepared_output_mapping<OutputTensor>(extents),
                     typename output_span::accessor_type{}};
}

template <class OutputTensor, class Scalar, class LhsMdspan, class RhsMdspan>
KernelAttempt probe_assign_product(matrix_product_extents const& shape, LhsMdspan& lhs, RhsMdspan& rhs)
{
  if constexpr (!PreparedOutputProbeAvailable<OutputTensor>)
  {
    return KernelAttempt::unsupported_layout;
  }
  else
  {
    auto output_span = prepared_output_probe<OutputTensor>(shape);
    return uni20::linalg::blas::probe_gemm<Scalar>(output_span, lhs, rhs);
  }
}

} // namespace detail::blas_backend

/// \brief Report BLAS eligibility for host-accessible mdspec GEMM operands.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<LhsMdspan> &&
           uni20::HostReadableMdspec<RhsMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, gemm_op const&, OutputMdspan&, Scalar const&, LhsMdspan&,
                                    RhsMdspan&, Scalar const&)
{
  if constexpr (detail::blas_backend::gemm_mdspan_types_compatible<OutputMdspan, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Resolve host access and invoke the BLAS GEMM adapter.
template <uni20::MutableRankedMdspecLike<2> OutputMdspan, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires uni20::HostWritableMdspec<OutputMdspan> && uni20::HostReadableMdspec<LhsMdspan> &&
           uni20::HostReadableMdspec<RhsMdspan>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, OutputMdspan& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs, Scalar beta)
{
  return detail::blas_backend::try_gemm(output, alpha, lhs, rhs, beta);
}

/// \brief Report BLAS eligibility for replaceable-output tensor matrix products.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::blas_backend::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, assign_product_op const&, OutputTensor&, Scalar const&,
                                    LhsMdspan&, RhsMdspan&)
{
  if constexpr (detail::blas_backend::assign_product_types_compatible<OutputTensor, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Probe and lower a replaceable-output tensor matrix product to BLAS GEMM.
/// \details A mismatched output is represented by prospective mdspan metadata
///          during the BLAS runtime probe. Input mappings and accessor state are
///          inspected without acquiring host access. The real output is prepared
///          only after every instance, layout, and transform check succeeds.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::blas_backend::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
KernelAttempt try_kernel(BlasBackend, assign_product_op const&, OutputTensor& output, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs)
{
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  if (uni20::tensor_extents_equal(output.extents(), shape))
  {
    auto output_span = uni20::mdspec_of(output);
    return detail::blas_backend::try_gemm(output_span, alpha, lhs, rhs, Scalar{});
  }

  if constexpr (!uni20::ResizableTensorOutput<OutputTensor>)
  {
    return KernelAttempt::unsupported_shape;
  }

  KernelAttempt const probe = detail::blas_backend::probe_assign_product<OutputTensor, Scalar>(shape, lhs, rhs);
  if (!kernel_attempt_succeeded(probe)) return probe;

  uni20::prepare_output(output, shape);
  auto output_span = uni20::mdspec_of(output);
  KernelAttempt const attempt = detail::blas_backend::try_gemm(output_span, alpha, lhs, rhs, Scalar{});
  CHECK(kernel_attempt_succeeded(attempt));
  return attempt;
}

/// \brief Report BLAS eligibility for a deferred Tensor matrix-product output.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::blas_backend::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan>
consteval auto kernel_accepts_types(BlasBackend const&, assign_product_op const&, async::shared_storage<OutputTensor>&,
                                    Scalar const&, LhsMdspan&, RhsMdspan&)
{
  if constexpr (detail::blas_backend::assign_product_types_compatible<OutputTensor, Scalar, LhsMdspan, RhsMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Probe BLAS GEMM before constructing or resizing a deferred host output.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedMdspecLike<2> LhsMdspan,
          uni20::RankedMdspecLike<2> RhsMdspan>
  requires detail::blas_backend::HostAssignProductAccess<OutputTensor, LhsMdspan, RhsMdspan> &&
           requires(async::shared_storage<OutputTensor>& storage, detail::matrix_product_extents const& shape) {
             uni20::prepare_output(storage, shape);
           }
KernelAttempt try_kernel(BlasBackend backend, assign_product_op const&,
                         async::shared_storage<OutputTensor>& output_storage, Scalar alpha, LhsMdspan& lhs,
                         RhsMdspan& rhs)
{
  if (output_storage.constructed()) return try_kernel(backend, assign_product_op{}, *output_storage, alpha, lhs, rhs);

  auto const shape = detail::matrix_product_shape(lhs, rhs);
  KernelAttempt const probe = detail::blas_backend::probe_assign_product<OutputTensor, Scalar>(shape, lhs, rhs);
  if (!kernel_attempt_succeeded(probe)) return probe;

  auto& output = uni20::prepare_output(output_storage, shape);
  auto output_span = uni20::mdspec_of(output);
  KernelAttempt const attempt = detail::blas_backend::try_gemm(output_span, alpha, lhs, rhs, Scalar{});
  CHECK(kernel_attempt_succeeded(attempt));
  return attempt;
}

} // namespace uni20::linalg
