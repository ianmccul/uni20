#pragma once

/**
 * \file conjugate_inplace.hpp
 * \ingroup linalg
 * \brief CUDA reference backend for in-place scalar conjugation.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/core/math.hpp>
#include <uni20/linalg/backends/cuda/elementwise_conjugate.hpp>
#include <uni20/linalg/backends/cuda/mdspec_traits.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cuda_reference
{

template <class Mdspec> using conjugate_scalar_t = std::remove_cv_t<typename std::remove_cvref_t<Mdspec>::element_type>;

template <class Mdspec>
concept SupportedConjugateMdspec = uni20::MutableMdspecLike<Mdspec> && is_raw_mutable_cuda_mdspec<Mdspec> &&
                                   uni20::Scalar<conjugate_scalar_t<Mdspec>> &&
                                   (uni20::has_trivial_conj<conjugate_scalar_t<Mdspec>> ||
                                    ((std::remove_cvref_t<Mdspec>::rank() == 0 || uni20::StridedMdspecLike<Mdspec>) &&
                                     supports_elementwise_conjugate<conjugate_scalar_t<Mdspec>>));

template <class Scalar> struct ConjugatePlan
{
    KernelAttempt attempt = KernelAttempt::unsupported_instance;
    uni20::cuda::CudaBuffer<Scalar>* buffer = nullptr;
    std::size_t element_offset = 0;
    std::size_t required_span_size = 0;
    LoweredElementwiseConjugatePlan elementwise_plan;
    bool has_work = false;
};

template <uni20::MdspecLike Mdspec>
  requires(std::remove_cvref_t<Mdspec>::rank() == 0 || uni20::StridedMdspecLike<Mdspec>)
[[nodiscard]] bool try_make_elementwise_conjugate_plan(Mdspec const& mdspec, LoweredElementwiseConjugatePlan& plan)
{
  auto host_plan = make_multi_iteration_plan(std::tuple{mdspec.mapping()});
  return try_lower_strided_elementwise_plan<elementwise_conjugate_maximum_rank>(host_plan, plan);
}

template <class Mdspec>
  requires SupportedConjugateMdspec<Mdspec>
[[nodiscard]] auto prepare_conjugate_inplace(Mdspec& mdspec)
{
  using scalar_type = conjugate_scalar_t<Mdspec>;
  ConjugatePlan<scalar_type> plan;

  if constexpr (uni20::has_trivial_conj<scalar_type>)
  {
    // Direct backend dispatch remains valid even when a higher-level conj view
    // would normally erase this identity operation.
    plan.attempt = KernelAttempt::success;
    return plan;
  }
  else
  {
    if (!mdspec.mapping().is_unique() || !try_make_elementwise_conjugate_plan(mdspec, plan.elementwise_plan))
    {
      plan.attempt = KernelAttempt::unsupported_layout;
      return plan;
    }

    plan.has_work = plan.elementwise_plan.element_count() != 0;
    if (!plan.has_work)
    {
      plan.attempt = KernelAttempt::success;
      return plan;
    }

    auto const required = mdspec.mapping().required_span_size();
    if (std::cmp_less(required, 0) || std::cmp_greater(required, std::numeric_limits<std::size_t>::max()))
    {
      plan.attempt = KernelAttempt::unsupported_shape;
      return plan;
    }
    plan.required_span_size = static_cast<std::size_t>(required);

    auto descriptor = mdspec.data_descriptor();
    plan.buffer = std::addressof(descriptor.buffer());
    plan.element_offset = descriptor.element_offset();
    CHECK(plan.element_offset <= plan.buffer->size() &&
              plan.required_span_size <= plan.buffer->size() - plan.element_offset,
          plan.element_offset, plan.required_span_size, plan.buffer->size());

    plan.attempt = KernelAttempt::success;
    return plan;
  }
}

template <class Scalar> void execute_conjugate_inplace(ConjugatePlan<Scalar> const& plan)
{
  CHECK(kernel_attempt_succeeded(plan.attempt));
  if (!plan.has_work) return;
  CHECK(plan.buffer != nullptr);

  auto stream = plan.buffer->resources().streams().acquire();
  int const device = plan.buffer->device().ordinal();
  CHECK_EQUAL(stream.device(), device);
  uni20::cuda::ScopedDevice guard(device);
  auto access = plan.buffer->write_synchronized_with(stream);
  auto* output = access.data() + plan.element_offset;

  if constexpr (supports_elementwise_conjugate<Scalar>)
  {
    plan.elementwise_plan.visit([&](auto const& elementwise_plan) {
      enqueue_elementwise_conjugate(output, elementwise_plan, stream.native_handle(), device);
    });
  }
  else
  {
    PANIC("unsupported scalar reached CUDA reference elementwise conjugation");
  }
}

/// \brief Report compile-time eligibility for CUDA mdspec in-place conjugation.
template <uni20::MutableMdspecLike Mdspec> consteval auto conjugate_acceptance()
{
  if constexpr (SupportedConjugateMdspec<Mdspec>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Conjugate a supported CUDA mdspec through one exclusive access.
template <class Mdspec>
  requires SupportedConjugateMdspec<Mdspec>
KernelAttempt conjugate_inplace(Mdspec& mdspec)
{
  auto preparation = prepare_conjugate_inplace(mdspec);
  if (!kernel_attempt_succeeded(preparation.attempt)) return preparation.attempt;
  execute_conjugate_inplace(preparation);
  return KernelAttempt::success;
}

} // namespace detail::cuda_reference

/// \brief Report CUDA in-place conjugation eligibility for a normalized mdspec.
template <uni20::MutableMdspecLike Mdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, conjugate_inplace_op const&, Mdspec&)
{
  constexpr auto acceptance = detail::cuda_reference::conjugate_acceptance<Mdspec>();
  if constexpr (acceptance == KernelTypeAcceptance::maybe)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Perform in-place conjugation on a normalized CUDA mdspec.
template <class Mdspec>
  requires detail::cuda_reference::SupportedConjugateMdspec<Mdspec>
KernelAttempt try_kernel(CudaReferenceBackend, conjugate_inplace_op const&, Mdspec& mdspec)
{
  return detail::cuda_reference::conjugate_inplace(mdspec);
}

} // namespace uni20::linalg
