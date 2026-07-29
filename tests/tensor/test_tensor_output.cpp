#include <uni20/common/trace.hpp>
#include <uni20/linalg/backend_selector.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace
{
using extents_2d = stdex::dextents<uni20::index_type, 2>;
using mutable_mdspan = stdex::mdspan<double, extents_2d, stdex::layout_left>;
using const_mdspan = stdex::mdspan<double const, extents_2d, stdex::layout_left>;

struct FixedTensorView
{
    double* data = nullptr;
    extents_2d shape;

    [[nodiscard]] auto backend_selector() const noexcept { return uni20::linalg::CpuReferenceBackend{}; }
    [[nodiscard]] auto extents() const noexcept -> extents_2d const& { return shape; }
    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return shape.extent(axis); }
    [[nodiscard]] auto mdspan() noexcept { return mutable_mdspan(data, shape); }
    [[nodiscard]] auto mdspan() const noexcept { return const_mdspan(data, shape); }
};

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

static_assert(uni20::TensorView<FixedTensorView>);
static_assert(uni20::MutableTensorView<FixedTensorView>);
static_assert(!uni20::ResizableTensorOutput<FixedTensorView>);
static_assert(uni20::ResizableTensorOutput<uni20::DenseMatrix<double>>);
static_assert(std::same_as<uni20::tensor_element_t<uni20::DenseMatrix<double>>, double>);
} // namespace

TEST(TensorOutputTest, PrepareOutputRetainsMatchingOwningTensor)
{
  uni20::DenseMatrix<double> matrix(2, 3);
  matrix[0, 0] = 7.0;
  auto* const original_handle = matrix.mutable_handle();

  uni20::prepare_output(matrix, extents_2d{2, 3});

  EXPECT_EQ(matrix.mutable_handle(), original_handle);
  EXPECT_EQ((matrix[0, 0]), 7.0);
}

TEST(TensorOutputTest, PrepareOutputRebuildsOwningTensorWithDefaultMapping)
{
  uni20::DenseMatrix<double> matrix(1, 1);

  uni20::prepare_output(matrix, extents_2d{2, 3});

  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 3);
  EXPECT_EQ(matrix.size(), 6);
  EXPECT_EQ(matrix.mapping().stride(0), 1);
  EXPECT_EQ(matrix.mapping().stride(1), 2);
}

TEST(TensorOutputTest, RequireOutputNeverResizesOwningTensor)
{
  uni20::DenseMatrix<double> matrix(1, 1);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::require_output(matrix, extents_2d{2, 3}), std::runtime_error);
  EXPECT_EQ(matrix.rows(), 1);
  EXPECT_EQ(matrix.cols(), 1);
}

TEST(TensorOutputTest, FixedTensorViewValidatesWithoutRebinding)
{
  double storage[6] = {};
  FixedTensorView view{.data = storage, .shape = extents_2d{2, 3}};

  uni20::prepare_output(view, extents_2d{2, 3});
  EXPECT_EQ(view.data, storage);

  ErrorModeGuard const error_mode;
  EXPECT_THROW(uni20::prepare_output(view, extents_2d{3, 2}), std::runtime_error);
  EXPECT_EQ(view.extent(0), 2);
  EXPECT_EQ(view.extent(1), 3);
}
