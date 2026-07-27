#include <uni20/core/types.hpp>
#include <uni20/linalg/blas/gemm.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/tensor.hpp>

#include "gemm_conformance.hpp"

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using extents_1d = stdex::dextents<uni20::index_type, 1>;

template <class Scalar> struct ValueTransformAccessor
{
    using element_type = Scalar;
    using data_handle_type = Scalar*;
    using reference = Scalar;
    using offset_policy = ValueTransformAccessor;

    constexpr data_handle_type offset(data_handle_type ptr, std::size_t offset) const { return ptr + offset; }

    constexpr reference access(data_handle_type ptr, std::size_t offset) const { return Scalar{2} * ptr[offset]; }
};

template <class Scalar>
using value_transform_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left, ValueTransformAccessor<Scalar>>;

template <class Scalar> using left_mdspan = stdex::mdspan<Scalar, extents_2d, stdex::layout_left>;
template <class Scalar> using vector_mdspan = stdex::mdspan<Scalar, extents_1d, stdex::layout_left>;

struct HostGemmPlatform
{
    static constexpr std::size_t expected_backend_candidates = 2;

    template <class Scalar, class Layout> [[nodiscard]] auto make_matrix(uni20::index_type rows, uni20::index_type cols)
    {
      return uni20::DenseMatrix<Scalar, Layout>(rows, cols);
    }

    template <class Scalar>
    [[nodiscard]] auto make_strided_matrix(uni20::index_type rows, uni20::index_type cols,
                                           std::array<uni20::index_type, 2> strides)
    {
      using matrix_type = uni20::StridedTensor<Scalar, 2>;
      return matrix_type(typename matrix_type::extents_type{rows, cols}, strides);
    }

    template <class Tensor>
    void write_physical(Tensor& tensor, std::vector<uni20::tensor_element_t<Tensor>> const& values)
    {
      ASSERT_EQ(values.size(), tensor.storage().size());
      tensor.storage() = values;
    }

    template <class Tensor>
    [[nodiscard]] auto read_physical(Tensor const& tensor) -> std::vector<uni20::tensor_element_t<Tensor>>
    {
      return tensor.storage();
    }
};

template <class Scalar> class NonStridedMatrixView {
  public:
    using element_type = Scalar;
    using value_type = std::remove_cv_t<element_type>;
    using extents_type = extents_2d;
    using layout_type = stdex::layout_left;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using accessor_type = stdex::default_accessor<element_type>;
    using data_handle_type = typename accessor_type::data_handle_type;
    using reference = typename accessor_type::reference;
    using index_type = typename extents_type::index_type;

    NonStridedMatrixView(data_handle_type data, index_type rows, index_type cols) : span_(data, rows, cols) {}

    static constexpr std::size_t rank() noexcept { return 2; }
    static constexpr bool is_always_strided() noexcept { return false; }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return span_.extents(); }
    [[nodiscard]] auto mapping() const noexcept -> mapping_type const& { return span_.mapping(); }
    [[nodiscard]] auto data_handle() const noexcept -> data_handle_type { return span_.data_handle(); }
    [[nodiscard]] auto accessor() const noexcept -> accessor_type const& { return span_.accessor(); }
    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return span_.extent(axis); }

    reference operator[](index_type row, index_type col) const { return span_[row, col]; }

  private:
    left_mdspan<element_type> span_;
};

template <class Matrix> void fill_matrix(Matrix&& matrix, std::initializer_list<double> values)
{
  auto it = values.begin();
  for (uni20::index_type row = 0; row < static_cast<uni20::index_type>(matrix.extent(0)); ++row)
  {
    for (uni20::index_type col = 0; col < static_cast<uni20::index_type>(matrix.extent(1)); ++col)
    {
      matrix[row, col] = *it;
      ++it;
    }
  }
}

using uni20::linalg::backend_list;
using uni20::linalg::BlasBackend;
using uni20::linalg::CpuReferenceBackend;
using uni20::linalg::gemm_op;
using uni20::linalg::kernel_types_maybe;
using uni20::linalg::kernel_types_yes;
using uni20::linalg::KernelAttempt;
using uni20::linalg::KernelTypeAcceptance;

namespace selector_customization_test
{
struct Backend
{
    static constexpr std::string_view name = "selector_customization";
};

inline bool backend_was_called = false;

struct StoragePolicy
{
    using backend_selector_type = backend_list<CpuReferenceBackend>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{CpuReferenceBackend{}};
    }
};

struct TensorAdapter
{
    using storage_policy = StoragePolicy;

    left_mdspan<double> span;

    [[nodiscard]] auto mdspan() const noexcept { return span; }
    [[nodiscard]] static constexpr auto backend_selector() noexcept { return StoragePolicy::backend_selector(); }
    [[nodiscard]] auto extents() const noexcept { return span.extents(); }
    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return span.extent(axis); }
};

template <class... Args> consteval auto kernel_accepts_types(Backend const&, gemm_op const&, Args&...)
{
  return kernel_types_yes;
}

template <class OutputMdspan, class Scalar, class LhsMdspan, class RhsMdspan>
KernelAttempt try_kernel(Backend backend, gemm_op const&, OutputMdspan&& output, Scalar alpha, LhsMdspan&& lhs,
                         RhsMdspan&& rhs, Scalar beta)
{
  (void)backend;
  backend_was_called = true;
  auto output_span = uni20::detail::tensor_device_mdspan(output);
  auto lhs_span = uni20::detail::tensor_device_mdspan(lhs);
  auto rhs_span = uni20::detail::tensor_device_mdspan(rhs);
  return try_kernel(CpuReferenceBackend{}, gemm_op{}, output_span, alpha, lhs_span, rhs_span, beta);
}
} // namespace selector_customization_test

struct TryKernelOnlyBackend
{
    static constexpr std::string_view name = "try_kernel_only";
};

template <class... Args> KernelAttempt try_kernel(TryKernelOnlyBackend, gemm_op const&, Args&&...)
{
  return KernelAttempt::success;
}

struct DecliningBackend
{
    static constexpr std::string_view name = "declining";
};

struct test_dispatch_op
{
    static constexpr std::string_view name = "test_dispatch";
};

[[maybe_unused]] consteval auto kernel_accepts_types(DecliningBackend const&, test_dispatch_op const&, int&)
{
  return kernel_types_maybe;
}

KernelAttempt try_kernel(DecliningBackend, test_dispatch_op, int&) { return KernelAttempt::unsupported_instance; }

struct MoveObservable
{
    explicit MoveObservable(int initial_value) : value(initial_value) {}

    MoveObservable(MoveObservable const&) = delete;
    MoveObservable& operator=(MoveObservable const&) = delete;

    MoveObservable(MoveObservable&& other) noexcept : value(std::exchange(other.value, -1)) {}

    MoveObservable& operator=(MoveObservable&& other) noexcept
    {
      value = std::exchange(other.value, -1);
      return *this;
    }

    int value;
};

struct PreservingDecliningBackend
{
    static constexpr std::string_view name = "preserving_decline";
};

struct ObservingBackend
{
    static constexpr std::string_view name = "observing";
    int* observed_value = nullptr;
};

[[maybe_unused]] consteval auto kernel_accepts_types(PreservingDecliningBackend const&, test_dispatch_op const&,
                                                     MoveObservable&)
{
  return kernel_types_maybe;
}

[[maybe_unused]] consteval auto kernel_accepts_types(ObservingBackend const&, test_dispatch_op const&, MoveObservable&)
{
  return kernel_types_yes;
}

template <class Argument>
  requires std::same_as<std::remove_cvref_t<Argument>, MoveObservable>
KernelAttempt try_kernel(PreservingDecliningBackend, test_dispatch_op, Argument&& argument)
{
  CHECK_EQUAL(argument.value, 42);
  return KernelAttempt::unsupported_instance;
}

template <class Argument>
  requires std::same_as<std::remove_cvref_t<Argument>, MoveObservable>
KernelAttempt try_kernel(ObservingBackend backend, test_dispatch_op, Argument&& argument)
{
  *backend.observed_value = argument.value;
  return KernelAttempt::success;
}

struct IncorrectTotalBackend
{
    static constexpr std::string_view name = "incorrect_total";
};

[[maybe_unused]] consteval auto kernel_accepts_types(IncorrectTotalBackend const&, test_dispatch_op const&, int&)
{
  return kernel_types_yes;
}

KernelAttempt try_kernel(IncorrectTotalBackend, test_dispatch_op, int&) { return KernelAttempt::unsupported_instance; }

template <class Backends, class Op, class... Args>
concept HasTryDispatchKernel = requires(Backends const& backends, Op op, Args&&... args) {
  { uni20::linalg::try_dispatch_kernel(backends, op, std::forward<Args>(args)...) } -> std::same_as<bool>;
};

template <class Backends, class Op, class... Args>
concept HasDispatchKernel = requires(Backends const& backends, Op op, Args&&... args) {
  { uni20::linalg::dispatch_kernel(backends, op, std::forward<Args>(args)...) } -> std::same_as<void>;
};

template <class Backends, class Op, class... Args>
concept HasDynamicDispatchKernel = requires(Backends const& backends, Op op, Args&&... args) {
  { uni20::linalg::dynamic_dispatch_kernel(backends, op, std::forward<Args>(args)...) } -> std::same_as<void>;
};

using declining_backends = backend_list<DecliningBackend>;
using unavailable_backends = backend_list<TryKernelOnlyBackend>;

static_assert(HasTryDispatchKernel<declining_backends, test_dispatch_op, int&>);
static_assert(HasDispatchKernel<declining_backends, test_dispatch_op, int&>);
static_assert(!HasTryDispatchKernel<unavailable_backends, test_dispatch_op, int&>);
static_assert(!HasDispatchKernel<unavailable_backends, test_dispatch_op, int&>);
static_assert(HasDynamicDispatchKernel<unavailable_backends, test_dispatch_op, int&>);

static_assert(uni20::RankedSpanLike<NonStridedMatrixView<double>, 2>);
static_assert(uni20::MutableRankedSpanLike<NonStridedMatrixView<double>, 2>);
static_assert(!uni20::StridedMdspan<NonStridedMatrixView<double>>);

static_assert(requires(BlasBackend backend, gemm_op op, left_mdspan<double>& output, double scalar,
                       value_transform_mdspan<double>& lhs, left_mdspan<double>& rhs) {
  { try_kernel(backend, op, output, scalar, lhs, rhs, scalar) } -> std::same_as<KernelAttempt>;
});
} // namespace

template <>
struct uni20::linalg::backend_selector_override<uni20::linalg::gemm_op, selector_customization_test::StoragePolicy>
{
    static auto select(uni20::linalg::gemm_op const&)
    {
      return uni20::linalg::backend_list{selector_customization_test::Backend{}};
    }
};

TEST(LinalgGemmDispatchTest, MissingOrNonViableTypeGateIsHardNo)
{
  std::vector<double> storage(1);
  vector_mdspan<double> vector(storage.data(), 1);

  auto candidates =
      uni20::linalg::kernel_type_candidates(backend_list{TryKernelOnlyBackend{}, BlasBackend{}, CpuReferenceBackend{}},
                                            gemm_op{}, vector, 1.0, vector, vector, 0.0);
  static_assert(std::same_as<decltype(candidates), backend_list<>>);

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{TryKernelOnlyBackend{}}, gemm_op{}, vector, 1.0, vector,
                                                 vector, 0.0),
            KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{CpuReferenceBackend{}}, gemm_op{}, vector, 1.0, vector,
                                                 vector, 0.0),
            KernelTypeAcceptance::no);
  EXPECT_EQ(
      uni20::linalg::probe_dispatch_kernel(backend_list{BlasBackend{}}, gemm_op{}, vector, 1.0, vector, vector, 0.0),
      KernelTypeAcceptance::no);
}

TEST(LinalgGemmDispatchTest, TypeCandidatesExposeEveryCompatibleGemmBackend)
{
  std::vector<double> a_storage(4);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4);
  left_mdspan<double> a(a_storage.data(), 2, 2);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);
  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  auto selector =
      backend_list{TryKernelOnlyBackend{}, uni20::linalg::LapackBackend{}, BlasBackend{}, CpuReferenceBackend{}};
  auto candidates = uni20::linalg::kernel_type_candidates(selector, gemm_op{}, c, 1.0, a, b, 0.0);
  static_assert(std::same_as<decltype(candidates), backend_list<BlasBackend, CpuReferenceBackend>>);

  int tested_backends = 0;
  auto test_backend = [&](auto const& backend) {
    fill_matrix(c, {0.0, 0.0, 0.0, 0.0});
    EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(backend, gemm_op{}, c, 1.0, a, b, 0.0));
    EXPECT_DOUBLE_EQ((c[0, 0]), 19.0);
    EXPECT_DOUBLE_EQ((c[0, 1]), 22.0);
    EXPECT_DOUBLE_EQ((c[1, 0]), 43.0);
    EXPECT_DOUBLE_EQ((c[1, 1]), 50.0);
    ++tested_backends;
  };
  std::apply([&](auto const&... backend) { (test_backend(backend), ...); }, candidates.entries);
  EXPECT_EQ(tested_backends, 2);
}

TEST(GemmBackendConformanceTest, ScalarsAndCanonicalLayouts)
{
  HostGemmPlatform platform;
  uni20::test::gemm_conformance::check_all_scalar_and_layout_cases(platform);
}

TEST(GemmBackendConformanceTest, PaddedLeadingDimensions)
{
  HostGemmPlatform platform;
  uni20::test::gemm_conformance::check_all_padded_layout_cases(platform);
}

TEST(GemmBackendConformanceTest, ConjugatingInputs)
{
  HostGemmPlatform platform;
  uni20::test::gemm_conformance::check_all_conjugating_input_cases(platform);
}

TEST(LinalgGemmDispatchTest, TryAndCheckedDispatchDistinguishRuntimeDecline)
{
  int argument = 0;
  auto backends = backend_list{TryKernelOnlyBackend{}, DecliningBackend{}};

  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(backends, test_dispatch_op{}, argument));

  bool const previous_errors_abort = trace::get_formatting_options().errors_abort();
  trace::get_formatting_options().set_errors_abort(false);

  std::optional<uni20::linalg::KernelDispatchError> captured_error;
  try
  {
    uni20::linalg::dispatch_kernel(backends, test_dispatch_op{}, argument);
    ADD_FAILURE() << "dispatch_kernel should have raised KernelDispatchError";
  }
  catch (uni20::linalg::KernelDispatchError const& error)
  {
    captured_error = error;
  }
  catch (std::exception const& error)
  {
    ADD_FAILURE() << "dispatch_kernel raised the wrong exception type: " << error.what();
  }

  EXPECT_THROW(
      uni20::linalg::dynamic_dispatch_kernel(backend_list{TryKernelOnlyBackend{}}, test_dispatch_op{}, argument),
      uni20::linalg::KernelDispatchError);
  trace::get_formatting_options().set_errors_abort(previous_errors_abort);

  ASSERT_TRUE(captured_error.has_value());
  EXPECT_EQ(captured_error->operation(), "test_dispatch");
  EXPECT_EQ(captured_error->failure(), uni20::linalg::KernelDispatchFailure::all_candidates_declined);
  ASSERT_EQ(captured_error->backend_attempts().size(), 2);
  EXPECT_EQ(captured_error->backend_attempts()[0].backend, "try_kernel_only");
  EXPECT_EQ(captured_error->backend_attempts()[0].type_acceptance, KernelTypeAcceptance::no);
  EXPECT_FALSE(captured_error->backend_attempts()[0].runtime_result.has_value());
  EXPECT_EQ(captured_error->backend_attempts()[1].backend, "declining");
  EXPECT_EQ(captured_error->backend_attempts()[1].type_acceptance, KernelTypeAcceptance::maybe);
  EXPECT_EQ(captured_error->backend_attempts()[1].runtime_result, KernelAttempt::unsupported_instance);
  ASSERT_TRUE(captured_error->source_location().has_value());
  EXPECT_GT(captured_error->source_location()->line(), 0);
#if UNI20_HAS_STACKTRACE
  EXPECT_TRUE(captured_error->stacktrace().has_value());
#endif

  auto const report = diagnostic_report(*captured_error);
  EXPECT_TRUE(report.title().empty());
  ASSERT_EQ(report.statuses().size(), 1);
  EXPECT_EQ(report.statuses()[0].first, uni20::presentation::semantic_glyph::failure);
  EXPECT_EQ(report.statuses()[0].second, "Kernel dispatch failed for 'test_dispatch'");
  ASSERT_EQ(report.tables().size(), 1);
  EXPECT_EQ(report.tables()[0].title(), "No available backend accepted this runtime instance");

  auto const diagnostic = trace::format_diagnostic(*captured_error);
  auto const rendered_diagnostic =
      uni20::presentation::render_plain(diagnostic, trace::get_formatting_options().presentation_policy());
  EXPECT_NE(rendered_diagnostic.find("unsupported runtime instance"), std::string::npos);
  EXPECT_NE(rendered_diagnostic.find("Source location:"), std::string::npos);
#if UNI20_HAS_STACKTRACE
  EXPECT_NE(rendered_diagnostic.find("Stacktrace:"), std::string::npos);
#endif
}

TEST(LinalgGemmDispatchDeathTest, CheckedDispatchRendersStructuredBackendReport)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  int argument = 0;
  auto backends = backend_list{TryKernelOnlyBackend{}, DecliningBackend{}};

  trace::get_formatting_options().set_errors_abort(true);
  EXPECT_DEATH(uni20::linalg::dispatch_kernel(backends, test_dispatch_op{}, argument),
               "Kernel dispatch failed for 'test_dispatch'");
}

TEST(LinalgGemmDispatchTest, DecliningBackendPreservesStableArgumentForFallback)
{
  int accepting_backend_observed_value = 0;
  MoveObservable argument(42);
  auto backends =
      backend_list{PreservingDecliningBackend{}, ObservingBackend{.observed_value = &accepting_backend_observed_value}};

  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(backends, test_dispatch_op{}, std::move(argument)));
  EXPECT_EQ(accepting_backend_observed_value, 42);
  EXPECT_EQ(argument.value, 42);
}

TEST(LinalgGemmDispatchDeathTest, BackendWithTotalTypeAcceptanceMustNotDecline)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  int argument = 0;

  EXPECT_DEATH(uni20::linalg::try_dispatch_kernel(IncorrectTotalBackend{}, test_dispatch_op{}, argument),
               "kernel_attempt_succeeded");
}

TEST(LinalgGemmDispatchTest, ForcedBlasBackendRunsRepresentableMdspans)
{
  std::vector<double> a_storage(6);
  std::vector<double> b_storage(6);
  std::vector<double> c_storage(4);

  left_mdspan<double> a(a_storage.data(), 2, 3);
  left_mdspan<double> b(b_storage.data(), 3, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{BlasBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::maybe);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{TryKernelOnlyBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{BlasBackend{}, CpuReferenceBackend{}}, gemm_op{}, c, 1.0,
                                                 a, b, 0.0),
            KernelTypeAcceptance::yes);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(BlasBackend{}, gemm_op{}, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 154.0);
}

TEST(LinalgGemmDispatchTest, ForcedBlasBackendDeclinesUnsupportedStrideBeforeSideEffects)
{
  std::vector<double> a_storage(8);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4, -7.0);

  stdex::layout_stride::mapping<extents_2d> bad_mapping(extents_2d{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), bad_mapping);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(BlasBackend{}, gemm_op{}, c, 1.0, a, b, 0.0));

  bool const previous_errors_abort = trace::get_formatting_options().errors_abort();
  trace::get_formatting_options().set_errors_abort(false);
  std::optional<uni20::linalg::KernelDispatchError> captured_error;
  try
  {
    uni20::linalg::dispatch_kernel(BlasBackend{}, gemm_op{}, c, 1.0, a, b, 0.0);
    ADD_FAILURE() << "direct BLAS dispatch should have declined the unsupported layout";
  }
  catch (uni20::linalg::KernelDispatchError const& error)
  {
    captured_error = error;
  }
  trace::get_formatting_options().set_errors_abort(previous_errors_abort);

  ASSERT_TRUE(captured_error.has_value());
  ASSERT_EQ(captured_error->backend_attempts().size(), 1);
  EXPECT_EQ(captured_error->backend_attempts()[0].runtime_result, KernelAttempt::unsupported_layout);

  EXPECT_DOUBLE_EQ((c[0, 0]), -7.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), -7.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), -7.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), -7.0);
}

TEST(LinalgGemmDispatchTest, BackendListFallsThroughWhenBlasDeclinesStride)
{
  std::vector<double> a_storage(8);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4, -7.0);

  stdex::layout_stride::mapping<extents_2d> bad_mapping(extents_2d{2, 2}, std::array<uni20::index_type, 2>{2, 5});
  stdex::mdspan<double, extents_2d, stdex::layout_stride> a(a_storage.data(), bad_mapping);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  auto selector = backend_list{BlasBackend{}, CpuReferenceBackend{}};
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(selector, gemm_op{}, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 50.0);
}

TEST(LinalgGemmDispatchTest, BackendListFallsThroughForAccessorOnlyReadableInput)
{
  std::vector<double> a_storage(4);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4);

  left_mdspan<double> a_raw(a_storage.data(), 2, 2);
  value_transform_mdspan<double> a(a_storage.data(), 2, 2);
  left_mdspan<double> b(b_storage.data(), 2, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{BlasBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{CpuReferenceBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::yes);

  fill_matrix(a_raw, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {1.0, 0.0, 0.0, 1.0});

  auto selector = backend_list{BlasBackend{}, CpuReferenceBackend{}};
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(selector, gemm_op{}, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 2.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 4.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 6.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 8.0);
}

TEST(LinalgGemmDispatchTest, CpuReferenceDoesNotReadOutputWhenBetaIsZero)
{
  std::vector<double> a_storage(1, 3.0);
  std::vector<double> b_storage(1, 4.0);
  std::vector<double> c_storage(1, std::numeric_limits<double>::quiet_NaN());

  left_mdspan<double> a(a_storage.data(), 1, 1);
  left_mdspan<double> b(b_storage.data(), 1, 1);
  left_mdspan<double> c(c_storage.data(), 1, 1);

  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(CpuReferenceBackend{}, gemm_op{}, c, 1.0, a, b, 0.0));
  EXPECT_DOUBLE_EQ((c[0, 0]), 12.0);
}

TEST(LinalgGemmDispatchTest, CpuReferenceAppliesBetaForZeroInnerExtent)
{
  std::vector<double> empty_storage;
  std::vector<double> c_storage(4);

  left_mdspan<double> a(empty_storage.data(), 2, 0);
  left_mdspan<double> b(empty_storage.data(), 0, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);

  fill_matrix(c, {1.0, 2.0, 3.0, 4.0});
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(CpuReferenceBackend{}, gemm_op{}, c,
                                                 std::numeric_limits<double>::infinity(), a, b, 3.0));
  EXPECT_EQ(c_storage, (std::vector<double>{3.0, 9.0, 6.0, 12.0}));

  fill_matrix(c, {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()});
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(CpuReferenceBackend{}, gemm_op{}, c,
                                                 std::numeric_limits<double>::infinity(), a, b, 0.0));
  EXPECT_EQ(c_storage, (std::vector<double>{0.0, 0.0, 0.0, 0.0}));
}

TEST(LinalgGemmDispatchTest, BlasDeclineFallsThroughForZeroInnerExtent)
{
  std::vector<double> empty_storage;
  std::vector<double> c_storage(4);

  left_mdspan<double> a(empty_storage.data(), 2, 0);
  left_mdspan<double> b(empty_storage.data(), 0, 2);
  left_mdspan<double> c(c_storage.data(), 2, 2);
  fill_matrix(c, {1.0, 2.0, 3.0, 4.0});

  auto selector = backend_list{BlasBackend{}, CpuReferenceBackend{}};
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(selector, gemm_op{}, c, 1.0, a, b, 2.0));
  EXPECT_EQ(c_storage, (std::vector<double>{2.0, 6.0, 4.0, 8.0}));
}

TEST(LinalgGemmDispatchTest, CpuReferenceAcceptsNonStridedAddressableViews)
{
  std::vector<double> a_storage(4);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4);

  NonStridedMatrixView<double> a(a_storage.data(), 2, 2);
  NonStridedMatrixView<double> b(b_storage.data(), 2, 2);
  NonStridedMatrixView<double> c(c_storage.data(), 2, 2);

  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{BlasBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::no);
  EXPECT_EQ(uni20::linalg::probe_dispatch_kernel(backend_list{CpuReferenceBackend{}}, gemm_op{}, c, 1.0, a, b, 0.0),
            KernelTypeAcceptance::yes);

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  auto selector = backend_list{BlasBackend{}, CpuReferenceBackend{}};
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(selector, gemm_op{}, c, 1.0, a, b, 0.0));

  EXPECT_DOUBLE_EQ((c[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 50.0);
}

TEST(LinalgGemmDispatchTest, TensorOperandsUseStorageDefaultSelector)
{
  using tensor_type = uni20::Tensor<double, 2>;
  using tensor_extents = typename tensor_type::extents_type;

  tensor_type a(tensor_extents{2, 3});
  tensor_type b(tensor_extents{3, 2});
  tensor_type c(tensor_extents{2, 2});

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  fill_matrix(b, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

  uni20::linalg::gemm(c, 1.0, a, b, 0.0);

  EXPECT_DOUBLE_EQ((c[0, 0]), 58.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 64.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 139.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 154.0);
}

TEST(LinalgGemmDispatchTest, TensorOperandsAcceptExplicitSelectorOverride)
{
  using tensor_type = uni20::Tensor<double, 2>;
  using tensor_extents = typename tensor_type::extents_type;

  tensor_type a(tensor_extents{2, 2});
  tensor_type b(tensor_extents{2, 2});
  tensor_type c(tensor_extents{2, 2});

  fill_matrix(a, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b, {5.0, 6.0, 7.0, 8.0});

  uni20::linalg::gemm(CpuReferenceBackend{}, c, 1.0, a, b, 0.0);

  EXPECT_DOUBLE_EQ((c[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((c[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((c[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((c[1, 1]), 50.0);
}

TEST(LinalgGemmDispatchTest, TensorOperandsUseGlobalStoragePolicyOverride)
{
  using selector_customization_test::Backend;
  using selector_customization_test::TensorAdapter;

  std::vector<double> a_storage(4);
  std::vector<double> b_storage(4);
  std::vector<double> c_storage(4);
  TensorAdapter a{left_mdspan<double>(a_storage.data(), 2, 2)};
  TensorAdapter b{left_mdspan<double>(b_storage.data(), 2, 2)};
  TensorAdapter c{left_mdspan<double>(c_storage.data(), 2, 2)};
  selector_customization_test::backend_was_called = false;

  fill_matrix(a.span, {1.0, 2.0, 3.0, 4.0});
  fill_matrix(b.span, {5.0, 6.0, 7.0, 8.0});

  uni20::linalg::gemm(c, 1.0, a, b, 0.0);

  EXPECT_TRUE(selector_customization_test::backend_was_called);
  EXPECT_DOUBLE_EQ((c.span[0, 0]), 19.0);
  EXPECT_DOUBLE_EQ((c.span[0, 1]), 22.0);
  EXPECT_DOUBLE_EQ((c.span[1, 0]), 43.0);
  EXPECT_DOUBLE_EQ((c.span[1, 1]), 50.0);
}
