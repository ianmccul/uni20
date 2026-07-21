#pragma once

#include <uni20/core/scalar_traits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/conjugate.hpp>

#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::test::gemm_conformance
{

template <class Scalar> [[nodiscard]] constexpr Scalar scalar(double real, double imag = 0.0)
{
  if constexpr (uni20::is_complex_v<Scalar>)
  {
    using real_type = typename Scalar::value_type;
    return Scalar{static_cast<real_type>(real), static_cast<real_type>(imag)};
  }
  else
  {
    CHECK_EQUAL(imag, 0.0);
    return static_cast<Scalar>(real);
  }
}

template <class Tensor>
[[nodiscard]] auto pack_logical(Tensor const& tensor, std::span<uni20::tensor_element_t<Tensor> const> logical)
    -> std::vector<uni20::tensor_element_t<Tensor>>
{
  using scalar_type = uni20::tensor_element_t<Tensor>;
  CHECK_EQUAL(logical.size(), static_cast<std::size_t>(tensor.extent(0) * tensor.extent(1)));
  std::vector<scalar_type> physical(tensor.size());
  for (uni20::index_type row = 0; row < tensor.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < tensor.extent(1); ++col)
    {
      physical[tensor.mapping()(row, col)] = logical[static_cast<std::size_t>(row * tensor.extent(1) + col)];
    }
  }
  return physical;
}

template <class Tensor>
[[nodiscard]] auto unpack_logical(Tensor const& tensor, std::span<uni20::tensor_element_t<Tensor> const> physical)
    -> std::vector<uni20::tensor_element_t<Tensor>>
{
  using scalar_type = uni20::tensor_element_t<Tensor>;
  CHECK_EQUAL(physical.size(), static_cast<std::size_t>(tensor.size()));
  std::vector<scalar_type> logical(static_cast<std::size_t>(tensor.extent(0) * tensor.extent(1)));
  for (uni20::index_type row = 0; row < tensor.extent(0); ++row)
  {
    for (uni20::index_type col = 0; col < tensor.extent(1); ++col)
    {
      logical[static_cast<std::size_t>(row * tensor.extent(1) + col)] = physical[tensor.mapping()(row, col)];
    }
  }
  return logical;
}

template <class Platform, class Tensor>
void write_logical(Platform& platform, Tensor& tensor, std::vector<uni20::tensor_element_t<Tensor>> const& logical)
{
  platform.write_physical(tensor, pack_logical(tensor, std::span<uni20::tensor_element_t<Tensor> const>{logical}));
}

template <class Platform, class Tensor>
[[nodiscard]] auto read_logical(Platform& platform,
                                Tensor const& tensor) -> std::vector<uni20::tensor_element_t<Tensor>>
{
  auto physical = platform.read_physical(tensor);
  return unpack_logical(tensor, std::span<uni20::tensor_element_t<Tensor> const>{physical});
}

template <class Scalar>
[[nodiscard]] auto reference_gemm(std::span<Scalar const> output, std::span<Scalar const> lhs,
                                  std::span<Scalar const> rhs, std::size_t rows, std::size_t inner, std::size_t cols,
                                  Scalar alpha, Scalar beta) -> std::vector<Scalar>
{
  CHECK_EQUAL(output.size(), rows * cols);
  CHECK_EQUAL(lhs.size(), rows * inner);
  CHECK_EQUAL(rhs.size(), inner * cols);
  std::vector<Scalar> result(rows * cols);
  for (std::size_t row = 0; row < rows; ++row)
  {
    for (std::size_t col = 0; col < cols; ++col)
    {
      Scalar product{};
      for (std::size_t index = 0; index < inner; ++index)
      {
        product += lhs[row * inner + index] * rhs[index * cols + col];
      }
      result[row * cols + col] = alpha * product + beta * output[row * cols + col];
    }
  }
  return result;
}

template <class Platform, class OutputTensor, class Scalar, class LhsTensor, class RhsTensor>
void expect_each_candidate(Platform& platform, OutputTensor& output, Scalar alpha, LhsTensor const& lhs,
                           RhsTensor const& rhs, Scalar beta, std::vector<Scalar> const& initial_output,
                           std::vector<Scalar> const& expected)
{
  auto selector = uni20::linalg::select_backend(uni20::linalg::gemm_op{}, output, lhs, rhs);
  auto output_span = output.mdspan();
  auto lhs_span = lhs.mdspan();
  auto rhs_span = rhs.mdspan();
  auto candidates = [&] {
    if constexpr (requires { platform.kernel_type_candidates(selector, output_span, alpha, lhs_span, rhs_span, beta); })
    {
      return platform.kernel_type_candidates(selector, output_span, alpha, lhs_span, rhs_span, beta);
    }
    else
    {
      return uni20::linalg::kernel_type_candidates(selector, uni20::linalg::gemm_op{}, output_span, alpha, lhs_span,
                                                   rhs_span, beta);
    }
  }();
  constexpr auto candidate_count = std::tuple_size_v<decltype(candidates.entries)>;
  static_assert(candidate_count == Platform::expected_backend_candidates);

  std::size_t tested = 0;
  auto test_backend = [&](auto const& backend) {
    write_logical(platform, output, initial_output);
    if constexpr (requires { platform.gemm(backend, output, alpha, lhs, rhs, beta); })
    {
      platform.gemm(backend, output, alpha, lhs, rhs, beta);
    }
    else
    {
      uni20::linalg::gemm(backend, output, alpha, lhs, rhs, beta);
    }
    EXPECT_EQ(read_logical(platform, output), expected) << "backend: " << backend.name;
    ++tested;
  };
  std::apply([&](auto const&... backend) { (test_backend(backend), ...); }, candidates.entries);
  EXPECT_EQ(tested, candidate_count);
}

template <class Scalar, class OutputLayout, class LhsLayout, class RhsLayout, class Platform>
void check_layout_case(Platform& platform)
{
  auto lhs = platform.template make_matrix<Scalar, LhsLayout>(2, 3);
  auto rhs = platform.template make_matrix<Scalar, RhsLayout>(3, 2);
  auto output = platform.template make_matrix<Scalar, OutputLayout>(2, 2);

  std::vector<Scalar> const lhs_values{scalar<Scalar>(1), scalar<Scalar>(2), scalar<Scalar>(3),
                                       scalar<Scalar>(4), scalar<Scalar>(5), scalar<Scalar>(6)};
  std::vector<Scalar> const rhs_values{scalar<Scalar>(7),  scalar<Scalar>(8),  scalar<Scalar>(9),
                                       scalar<Scalar>(10), scalar<Scalar>(11), scalar<Scalar>(12)};
  std::vector<Scalar> const output_values{scalar<Scalar>(1), scalar<Scalar>(2), scalar<Scalar>(3), scalar<Scalar>(4)};
  Scalar const alpha = scalar<Scalar>(2);
  Scalar const beta = scalar<Scalar>(3);
  auto const expected = reference_gemm<Scalar>(output_values, lhs_values, rhs_values, 2, 3, 2, alpha, beta);

  write_logical(platform, lhs, lhs_values);
  write_logical(platform, rhs, rhs_values);
  expect_each_candidate(platform, output, alpha, lhs, rhs, beta, output_values, expected);
}

template <class Scalar, class Platform> void check_canonical_layouts(Platform& platform)
{
  check_layout_case<Scalar, uni20::ColumnMajor, uni20::ColumnMajor, uni20::ColumnMajor>(platform);
  check_layout_case<Scalar, uni20::RowMajor, uni20::RowMajor, uni20::RowMajor>(platform);
  check_layout_case<Scalar, uni20::ColumnMajor, uni20::RowMajor, uni20::ColumnMajor>(platform);
  check_layout_case<Scalar, uni20::RowMajor, uni20::ColumnMajor, uni20::RowMajor>(platform);
}

template <class Platform> void check_all_scalar_and_layout_cases(Platform& platform)
{
  check_canonical_layouts<float>(platform);
  check_canonical_layouts<double>(platform);
  check_canonical_layouts<uni20::complex<float>>(platform);
  check_canonical_layouts<uni20::complex<double>>(platform);
}

template <class Scalar, class Platform> void check_padded_layout_case(Platform& platform)
{
  auto lhs = platform.template make_strided_matrix<Scalar>(2, 3, {1, 4});
  auto rhs = platform.template make_strided_matrix<Scalar>(3, 2, {4, 1});
  auto output = platform.template make_strided_matrix<Scalar>(2, 2, {1, 5});

  std::vector<Scalar> const lhs_values{scalar<Scalar>(1), scalar<Scalar>(2), scalar<Scalar>(3),
                                       scalar<Scalar>(4), scalar<Scalar>(5), scalar<Scalar>(6)};
  std::vector<Scalar> const rhs_values{scalar<Scalar>(7),  scalar<Scalar>(8),  scalar<Scalar>(9),
                                       scalar<Scalar>(10), scalar<Scalar>(11), scalar<Scalar>(12)};
  std::vector<Scalar> const output_values{scalar<Scalar>(1), scalar<Scalar>(2), scalar<Scalar>(3), scalar<Scalar>(4)};
  Scalar const alpha = scalar<Scalar>(2);
  Scalar const beta = scalar<Scalar>(3);
  auto const expected = reference_gemm<Scalar>(output_values, lhs_values, rhs_values, 2, 3, 2, alpha, beta);

  write_logical(platform, lhs, lhs_values);
  write_logical(platform, rhs, rhs_values);
  expect_each_candidate(platform, output, alpha, lhs, rhs, beta, output_values, expected);
}

template <class Platform> void check_all_padded_layout_cases(Platform& platform)
{
  check_padded_layout_case<float>(platform);
  check_padded_layout_case<double>(platform);
  check_padded_layout_case<uni20::complex<float>>(platform);
  check_padded_layout_case<uni20::complex<double>>(platform);
}

template <class Scalar, class Platform> void check_conjugating_input_case(Platform& platform)
{
  auto lhs = platform.template make_matrix<Scalar, uni20::RowMajor>(1, 2);
  auto rhs = platform.template make_matrix<Scalar, uni20::ColumnMajor>(2, 1);
  auto output = platform.template make_matrix<Scalar, uni20::ColumnMajor>(1, 1);

  std::vector<Scalar> const lhs_values{scalar<Scalar>(1, 2), scalar<Scalar>(3, -1)};
  std::vector<Scalar> const rhs_values{scalar<Scalar>(2, -1), scalar<Scalar>(-1, 4)};
  std::vector<Scalar> const output_values{Scalar{}};
  std::vector<Scalar> const expected{scalar<Scalar>(-7, 6)};
  write_logical(platform, lhs, lhs_values);
  write_logical(platform, rhs, rhs_values);

  auto conjugated_lhs = uni20::conj(lhs);
  expect_each_candidate(platform, output, Scalar{1}, conjugated_lhs, rhs, Scalar{}, output_values, expected);
}

template <class Platform> void check_all_conjugating_input_cases(Platform& platform)
{
  check_conjugating_input_case<uni20::complex<float>>(platform);
  check_conjugating_input_case<uni20::complex<double>>(platform);
}

} // namespace uni20::test::gemm_conformance
