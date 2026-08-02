#include <uni20/core/compiler_attributes.hpp>
#include <uni20/mdspan/conjugate_accessor.hpp>
#include <uni20/mdspan/generated_accessor.hpp>
#include <uni20/mdspan/generated_layout.hpp>
#include <uni20/mdspan/transform_view.hpp>
#include <uni20/mdspan/zip_layout.hpp>
#include <uni20/storage/cuda_accessor.hpp>

#include <cuda/std/complex>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

namespace
{

using extents_type = stdex::dextents<uni20::index_type, 2>;
using vector_extents_type = stdex::dextents<uni20::index_type, 1>;
using left_mapping = stdex::layout_left::mapping<extents_type>;
using right_mapping = stdex::layout_right::mapping<extents_type>;
using stride_mapping = stdex::layout_stride::mapping<extents_type>;
using generated_mapping = uni20::GeneratedLayout::mapping<extents_type>;
using strided_zip_mapping = uni20::StridedZipLayout<2>::mapping<extents_type>;
using general_zip_mapping = uni20::GeneralZipLayout<uni20::GeneratedLayout, stdex::layout_left>::mapping<extents_type>;

struct DeviceScale
{
    double factor;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr double operator()(double value) const noexcept { return factor * value; }
};

struct DeviceAdd
{
    [[nodiscard]] UNI20_HOST_DEVICE constexpr double operator()(double lhs, double rhs) const noexcept
    {
      return lhs + rhs;
    }
};

struct DeviceComplexScale
{
    float factor;

    [[nodiscard]] UNI20_HOST_DEVICE constexpr cuda::std::complex<float>
    operator()(cuda::std::complex<float> value) const noexcept
    {
      return factor * value;
    }
};

struct DeviceGeneratedValue
{
    template <class Indices>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr double operator()(Indices const& indices) const noexcept
    {
      return static_cast<double>(indices[0] * 10 + indices[1]);
    }
};

using cuda_input_accessor = uni20::cuda::CudaPointerAccessor<double const>;
using cuda_output_accessor = uni20::cuda::CudaPointerAccessor<double>;
using cuda_input_span = stdex::mdspan<double const, vector_extents_type, stdex::layout_left, cuda_input_accessor>;
using unary_transform_span = decltype(uni20::transform_view(DeviceScale{}, std::declval<cuda_input_span const&>()));
using binary_transform_span = decltype(uni20::transform_view(DeviceAdd{}, std::declval<cuda_input_span const&>(),
                                                             std::declval<cuda_input_span const&>()));
using complex_cuda_input_span = stdex::mdspan<uni20::cfloat const, vector_extents_type, stdex::layout_left,
                                              uni20::cuda::CudaPointerAccessor<uni20::cfloat const>>;
using complex_transform_span =
    decltype(uni20::transform_view(DeviceComplexScale{}, std::declval<complex_cuda_input_span const&>()));
using conjugated_complex_transform_span = decltype(uni20::conj(std::declval<complex_transform_span const&>()));
using generated_accessor = uni20::generated_accessor<double, extents_type, DeviceGeneratedValue>;
using const_cuda_accessor = uni20::const_accessor_adaptor<cuda_output_accessor>;
using complex_input_accessor = uni20::cuda::CudaPointerAccessor<uni20::cdouble const>;
using conjugated_input_accessor = uni20::conjugated_accessor<complex_input_accessor>;

template <class Mapping> __device__ auto evaluate_mapping(Mapping const& mapping)
{
  return mapping(uni20::index_type{1}, uni20::index_type{2});
}

[[maybe_unused]] __global__ void canonical_mapping_probe(left_mapping left, right_mapping right, stride_mapping stride)
{
  auto left_offset = evaluate_mapping(left);
  auto right_offset = evaluate_mapping(right);
  auto stride_offset = evaluate_mapping(stride);
  if (left_offset == right_offset && right_offset == stride_offset) return;
}

[[maybe_unused]] __global__ void generated_mapping_accessor_probe(generated_mapping mapping,
                                                                  generated_accessor accessor)
{
  auto offset = evaluate_mapping(mapping);
  auto value = accessor.access(uni20::generated_data_handle{}, static_cast<std::size_t>(offset));
  if (value == 0.0) return;
}

[[maybe_unused]] __global__ void zip_mapping_probe(strided_zip_mapping strided, general_zip_mapping general)
{
  auto strided_offsets = evaluate_mapping(strided);
  auto general_offsets = evaluate_mapping(general);
  if (std::get<0>(strided_offsets) == std::get<0>(general_offsets)) return;
}

[[maybe_unused]] __global__ void accessor_adaptor_probe(const_cuda_accessor adapted,
                                                        conjugated_input_accessor conjugated, double* output,
                                                        uni20::cdouble const* complex_input)
{
  auto const value = adapted.access(output, 0);
  auto const complex_value = conjugated.access(complex_input, 0);
  if (value == complex_value.real()) return;
}

[[maybe_unused]] __global__ void
transform_accessor_probe(typename unary_transform_span::accessor_type unary,
                         typename binary_transform_span::accessor_type binary,
                         typename conjugated_complex_transform_span::accessor_type complex_transform, double const* lhs,
                         double const* rhs, uni20::cfloat const* complex_input)
{
  auto unary_value = unary.access(lhs, 0);
  typename binary_transform_span::data_handle_type handles{lhs, rhs};
  typename binary_transform_span::accessor_type::offset_type offsets{0, 0};
  auto binary_value = binary.access(handles, offsets);
  auto complex_value = complex_transform.access(complex_input, 0);
  if (unary_value == binary_value && complex_value.real() == 0.0F) return;
}

} // namespace

TEST(CudaDescriptorCompileTest, ExecutionDescriptorCompositionsCompileForDevice) { SUCCEED(); }
