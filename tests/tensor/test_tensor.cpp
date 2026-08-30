#include <uni20/common/trace.hpp>
#include <uni20/tensor/layout.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace uni20;

namespace
{

using index_t = index_type;
using extents_2d = stdex::dextents<index_t, 2>;
using tensor_type = Tensor<int, 2, HostStorage>;
using strided_tensor_type = StridedTensor<int, 2, HostStorage>;
using scalar_tensor_type = ScalarTensor<double>;

static_assert(!HostStorage::storage_t<double>::initializes_elements);
static_assert(!HostStorage::storage_t<uni20::complex<double>>::initializes_elements);
static_assert(!std::equality_comparable<HostBuffer<int>>);

struct ThrowingAssignmentValue
{
    static inline int live_count = 0;
    static inline bool throw_on_assignment = false;

    int value = 0;

    ThrowingAssignmentValue() { ++live_count; }
    explicit ThrowingAssignmentValue(int value) : value(value) { ++live_count; }
    ThrowingAssignmentValue(ThrowingAssignmentValue const& other) : value(other.value) { ++live_count; }

    auto operator=(ThrowingAssignmentValue const& other) -> ThrowingAssignmentValue&
    {
      if (throw_on_assignment) throw std::runtime_error("requested assignment failure");
      value = other.value;
      return *this;
    }

    ~ThrowingAssignmentValue() { --live_count; }
};

struct ImmediateAndDescriptorStorage
{
    template <class ElementType> using storage_t = std::vector<ElementType>;
    using backend_selector_type = HostStorage::backend_selector_type;

    template <class ElementType>
    [[nodiscard]] static auto make_handle(storage_t<ElementType>& storage) noexcept -> ElementType*
    {
      return storage.data();
    }

    template <class ElementType>
    [[nodiscard]] static auto make_handle(storage_t<ElementType> const& storage) noexcept -> ElementType const*
    {
      return storage.data();
    }

    template <class ElementType>
    [[nodiscard]] static auto make_data_descriptor(storage_t<ElementType>& storage) noexcept -> ElementType*
    {
      return storage.data();
    }

    template <class ElementType>
    [[nodiscard]] static auto make_data_descriptor(storage_t<ElementType> const& storage) noexcept -> ElementType const*
    {
      return storage.data();
    }

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return HostStorage::backend_selector();
    }
};

struct ReadOnlyImmediateStorage
{
    template <class ElementType> using storage_t = std::vector<ElementType>;
    using backend_selector_type = HostStorage::backend_selector_type;

    template <class ElementType>
    [[nodiscard]] static auto make_handle(storage_t<ElementType> const& storage) noexcept -> ElementType const*
    {
      return storage.data();
    }

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return HostStorage::backend_selector();
    }
};

template <typename ElementType> struct PassthroughAccessor
{
    using offset_policy = PassthroughAccessor;
    using element_type = ElementType;
    using reference = element_type&;
    using data_handle_type = element_type*;

    [[nodiscard]] reference access(data_handle_type handle, std::size_t offset) const noexcept
    {
      return handle[offset];
    }

    [[nodiscard]] data_handle_type offset(data_handle_type handle, std::size_t value) const noexcept
    {
      return handle + value;
    }
};

struct PassthroughAccessorFactory
{
    template <typename ElementType> using accessor_t = PassthroughAccessor<ElementType>;

    template <typename ElementType, typename Storage>
    [[nodiscard]] constexpr auto make_accessor(Storage const&) const noexcept -> accessor_t<ElementType>
    {
      return accessor_t<ElementType>{};
    }
};

class StorageFreeTensorView {
  public:
    using extents_type = extents_2d;
    using mutable_mdspan_type = stdex::mdspan<int, extents_type>;
    using const_mdspan_type = stdex::mdspan<int const, extents_type>;
    using backend_selector_type = linalg::backend_list<linalg::CpuReferenceBackend>;

    StorageFreeTensorView(int* data, extents_type extents) : data_(data), extents_(extents) {}

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto mdspan() noexcept -> mutable_mdspan_type { return mutable_mdspan_type{data_, extents_}; }

    [[nodiscard]] auto mdspan() const noexcept -> const_mdspan_type { return const_mdspan_type{data_, extents_}; }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return extents_; }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return extents_.extent(axis); }

  private:
    int* data_;
    extents_type extents_;
};

struct MismatchedMutableRankTensorView
{
    using const_mdspan_type = stdex::mdspan<int const, extents_2d>;
    using mutable_extents_type = stdex::dextents<index_t, 1>;
    using mutable_mdspan_type = stdex::mdspan<int, mutable_extents_type>;
    using backend_selector_type = linalg::backend_list<linalg::CpuReferenceBackend>;

    [[nodiscard]] static auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto mdspan() noexcept -> mutable_mdspan_type
    {
      return mutable_mdspan_type{data_.data(), mutable_extents_type{4}};
    }

    [[nodiscard]] auto mdspan() const noexcept -> const_mdspan_type
    {
      return const_mdspan_type{data_.data(), extents_};
    }

    [[nodiscard]] auto extents() const noexcept -> extents_2d const& { return extents_; }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept -> index_t { return extents_.extent(axis); }

    std::array<int, 4> data_{};
    extents_2d extents_{2, 2};
};

struct MetadataDescriptor
{};

class DeferredStridedTensorView {
  public:
    using mdspan_type = uni20::mdspec<int const, extents_2d, stdex::layout_stride, stdex::default_accessor<int const>,
                                      MetadataDescriptor>;
    using backend_selector_type = linalg::backend_list<linalg::CpuReferenceBackend>;

    DeferredStridedTensorView(extents_2d extents, std::array<index_t, 2> strides)
        : span_(MetadataDescriptor{}, typename mdspan_type::mapping_type{extents, strides},
                typename mdspan_type::accessor_type{})
    {}

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return backend_selector_type{linalg::CpuReferenceBackend{}};
    }

    [[nodiscard]] auto mdspec() const noexcept -> mdspan_type const& { return span_; }

    [[nodiscard]] auto extents() const noexcept -> extents_2d const& { return span_.extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept -> index_t { return span_.extent(axis); }

  private:
    mdspan_type span_;
};

struct AccessStateCounters
{
    int release_calls = 0;
    int completed_releases = 0;
};

class InstrumentedTensorAccessState {
  public:
    explicit InstrumentedTensorAccessState(AccessStateCounters& counters) noexcept : counters_(&counters) {}

    InstrumentedTensorAccessState(InstrumentedTensorAccessState const&) = delete;
    InstrumentedTensorAccessState& operator=(InstrumentedTensorAccessState const&) = delete;

    InstrumentedTensorAccessState(InstrumentedTensorAccessState&& other) noexcept
        : counters_(std::exchange(other.counters_, nullptr)), active_(std::exchange(other.active_, false))
    {}

    InstrumentedTensorAccessState& operator=(InstrumentedTensorAccessState&& other) noexcept
    {
      if (this != &other)
      {
        this->release();
        counters_ = std::exchange(other.counters_, nullptr);
        active_ = std::exchange(other.active_, false);
      }
      return *this;
    }

    ~InstrumentedTensorAccessState() { this->release(); }

    void release() noexcept
    {
      if (counters_ == nullptr) return;
      ++counters_->release_calls;
      if (!active_) return;
      active_ = false;
      ++counters_->completed_releases;
    }

  private:
    AccessStateCounters* counters_ = nullptr;
    bool active_ = true;
};

class ThrowingMoveTensorAccessState {
  public:
    ThrowingMoveTensorAccessState() = default;
    ThrowingMoveTensorAccessState(ThrowingMoveTensorAccessState const&) = delete;
    ThrowingMoveTensorAccessState& operator=(ThrowingMoveTensorAccessState const&) = delete;
    ThrowingMoveTensorAccessState(ThrowingMoveTensorAccessState&&) noexcept(false) {}

    void release() noexcept {}
};

struct ThrowingMoveAccessor
{
    using offset_policy = ThrowingMoveAccessor;
    using element_type = int const;
    using reference = element_type&;
    using data_handle_type = element_type*;

    ThrowingMoveAccessor() = default;
    ThrowingMoveAccessor(ThrowingMoveAccessor const&) noexcept = default;
    ThrowingMoveAccessor(ThrowingMoveAccessor&&) noexcept(false) {}

    [[nodiscard]] reference access(data_handle_type handle, std::size_t offset) const noexcept
    {
      return handle[offset];
    }

    [[nodiscard]] data_handle_type offset(data_handle_type handle, std::size_t value) const noexcept
    {
      return handle + value;
    }
};

using throwing_move_mdspan = stdex::mdspan<int const, extents_2d, stdex::layout_left, ThrowingMoveAccessor>;

template <class Mdspan, class AccessState>
concept CanFormReadMdspanLease = requires { typename read_mdspan_lease<Mdspan, AccessState>; };

using access_test_selector = linalg::backend_list<linalg::CpuReferenceBackend>;
using access_test_mutable_mdspan = stdex::mdspan<int, extents_2d>;
using access_test_const_mdspan = stdex::mdspan<int const, extents_2d>;
using instrumented_read_mdspan_lease = read_mdspan_lease<access_test_const_mdspan, InstrumentedTensorAccessState>;
using instrumented_write_mdspan_lease = write_mdspan_lease<access_test_mutable_mdspan, InstrumentedTensorAccessState>;
using instrumented_read_tensor_lease =
    read_tensor_lease<access_test_const_mdspan, InstrumentedTensorAccessState, access_test_selector, HostStorage>;
using instrumented_write_tensor_lease =
    write_tensor_lease<access_test_mutable_mdspan, access_test_const_mdspan, InstrumentedTensorAccessState,
                       access_test_selector, HostStorage>;

using immediate_and_descriptor_tensor = Tensor<int, 2, ImmediateAndDescriptorStorage>;
using read_only_tensor = Tensor<int, 2, ReadOnlyImmediateStorage>;
using semantic_accessor_tensor = Tensor<int, 2, HostStorage, ColumnMajor, PassthroughAccessorFactory>;

using read_lease_type = decltype(acquire_host_read_access_sync(std::declval<tensor_type const&>()));
using write_lease_type = decltype(acquire_host_write_access_sync(std::declval<tensor_type&>()));
using read_access_type = decltype(acquire_host_read_access_async(std::declval<tensor_type const&>()));
using write_access_type = decltype(acquire_host_write_access_async(std::declval<tensor_type&>()));
using storage_free_read_lease = decltype(acquire_host_read_access_sync(std::declval<StorageFreeTensorView const&>()));
using storage_free_write_lease = decltype(acquire_host_write_access_sync(std::declval<StorageFreeTensorView&>()));

template <class T>
concept HasStorageObserver = requires(T& value) { value.storage(); };

template <class T>
concept HasMutableData = requires(T& value) {
  { value.data() } -> std::same_as<typename T::element_type*>;
};

template <class T>
concept HasConstData = requires(T const& value) {
  { value.data() } -> std::same_as<typename T::element_type const*>;
};

template <class T>
concept CanBorrowReadFromRvalue = requires(T&& value) { acquire_host_read_access_sync(std::move(value)); };

static_assert(std::same_as<tensor_type, BasicTensor<int, extents_2d, HostStorage, ColumnMajor>>);
static_assert(std::same_as<tensor_type, ColumnMajorTensor<int, 2>>);
static_assert(std::same_as<RowMajorTensor<int, 2>, Tensor<int, 2, HostStorage, RowMajor>>);
static_assert(std::same_as<strided_tensor_type, Tensor<int, 2, HostStorage, stdex::layout_stride>>);
static_assert(!std::constructible_from<tensor_type, extents_2d const&, std::array<index_t, 2> const&>);
static_assert(std::constructible_from<strided_tensor_type, extents_2d const&, std::array<index_t, 2> const&>);
static_assert(HasMutableData<tensor_type>);
static_assert(HasConstData<tensor_type>);
static_assert(HasMutableData<RowMajorTensor<int, 2>>);
static_assert(HasConstData<RowMajorTensor<int, 2>>);
static_assert(!HasMutableData<strided_tensor_type>);
static_assert(!HasConstData<strided_tensor_type>);
static_assert(!HasMutableData<read_only_tensor>);
static_assert(HasConstData<read_only_tensor>);
static_assert(!HasMutableData<semantic_accessor_tensor>);
static_assert(!HasConstData<semantic_accessor_tensor>);

static_assert(ImmediateTensorView<tensor_type>);
static_assert(TensorView<tensor_type>);
static_assert(HostAccessibleMdspan<immediate_tensor_mdspan_t<tensor_type>>);
static_assert(!CudaAccessibleMdspan<immediate_tensor_mdspan_t<tensor_type>>);
static_assert(MutableTensorView<tensor_type>);
static_assert(OwningTensor<tensor_type>);
static_assert(OwningTensor<tensor_type const>);
static_assert(MutableImmediateTensorView<tensor_type>);
static_assert(RankedImmediateTensorView<tensor_type, 2>);
static_assert(MutableRankedImmediateTensorView<tensor_type, 2>);
static_assert(StridedImmediateTensorView<tensor_type>);
static_assert(MutableStridedImmediateTensorView<tensor_type>);
static_assert(RankedStridedImmediateTensorView<tensor_type, 2>);
static_assert(MutableRankedStridedImmediateTensorView<tensor_type, 2>);
static_assert(!RankedImmediateTensorView<tensor_type, 1>);
static_assert(!MutableRankedImmediateTensorView<tensor_type, 1>);
static_assert(!MutableImmediateTensorView<tensor_type const>);
static_assert(!MdspanLike<tensor_type>);
static_assert(!MdspecLike<tensor_type>);
static_assert(!StridedMdspanLike<tensor_type>);
static_assert(std::same_as<tensor_storage_policy_t<tensor_type>, HostStorage>);
static_assert(std::same_as<mutable_tensor_mdspec_t<tensor_type>, mutable_immediate_tensor_mdspan_t<tensor_type>>);
static_assert(HostReadableMdspec<access_test_const_mdspan>);
static_assert(HostWritableMdspec<access_test_mutable_mdspan>);
static_assert(ReadMdspanLease<instrumented_read_mdspan_lease>);
static_assert(WriteMdspanLease<instrumented_write_mdspan_lease>);
static_assert(!ImmediateTensorView<instrumented_read_mdspan_lease>);
static_assert(!ImmediateTensorView<instrumented_write_mdspan_lease>);
static_assert(ScalarImmediateTensorView<scalar_tensor_type>);
static_assert(MutableScalarImmediateTensorView<scalar_tensor_type>);
static_assert(ScalarTensorView<scalar_tensor_type>);
static_assert(MutableScalarTensorView<scalar_tensor_type>);
static_assert(OwningTensor<scalar_tensor_type>);
static_assert(!ScalarImmediateTensorView<tensor_type>);
static_assert(MdspanLike<decltype(std::declval<tensor_type const&>().mdspec())>);
static_assert(ReadTensorLease<read_lease_type>);
static_assert(WriteTensorLease<write_lease_type>);
static_assert(HostReadTensorLease<read_lease_type>);
static_assert(HostWriteTensorLease<write_lease_type>);
static_assert(!CudaReadTensorLease<read_lease_type>);
static_assert(!CudaWriteTensorLease<write_lease_type>);
static_assert(sizeof(read_lease_type) == sizeof(void*));
static_assert(sizeof(write_lease_type) == sizeof(void*));
static_assert(std::is_trivially_destructible_v<read_lease_type>);
static_assert(std::is_trivially_destructible_v<write_lease_type>);
static_assert(
    std::same_as<decltype(std::declval<read_lease_type const&>().storage()), tensor_type::storage_type const&>);
static_assert(!HasStorageObserver<write_lease_type>);
static_assert(std::same_as<decltype(std::declval<read_access_type&>().await_resume()), read_lease_type>);
static_assert(std::same_as<decltype(std::declval<write_access_type&>().await_resume()), write_lease_type>);
static_assert(ImmediateTensorView<StorageFreeTensorView>);
static_assert(TensorView<StorageFreeTensorView>);
static_assert(MutableImmediateTensorView<StorageFreeTensorView>);
static_assert(MutableTensorView<StorageFreeTensorView>);
static_assert(ImmediateTensorView<MismatchedMutableRankTensorView>);
static_assert(TensorView<MismatchedMutableRankTensorView>);
static_assert(MutableImmediateTensorView<MismatchedMutableRankTensorView>);
static_assert(MutableTensorView<MismatchedMutableRankTensorView>);
static_assert(RankedImmediateTensorView<MismatchedMutableRankTensorView, 2>);
static_assert(RankedTensorView<MismatchedMutableRankTensorView, 2>);
static_assert(!MutableRankedImmediateTensorView<MismatchedMutableRankTensorView, 2>);
static_assert(!MutableRankedTensorView<MismatchedMutableRankTensorView, 2>);
static_assert(std::same_as<tensor_mdspec_t<StorageFreeTensorView>, StorageFreeTensorView::const_mdspan_type>);
static_assert(std::same_as<decltype(mdspec_of(std::declval<StorageFreeTensorView const&>())),
                           StorageFreeTensorView::const_mdspan_type>);
static_assert(std::same_as<decltype(mdspec_of(std::declval<DeferredStridedTensorView const&>())),
                           decltype(std::declval<DeferredStridedTensorView const&>().mdspec())>);
static_assert(std::same_as<std::remove_cvref_t<decltype(mdspec_of(std::declval<DeferredStridedTensorView const&>()))>,
                           DeferredStridedTensorView::mdspan_type>);
static_assert(immediate_and_descriptor_tensor::immediately_readable);
static_assert(immediate_and_descriptor_tensor::immediately_writable);
static_assert(immediate_and_descriptor_tensor::deferred_readable);
static_assert(immediate_and_descriptor_tensor::deferred_writable);
static_assert(std::same_as<decltype(mdspec_of(std::declval<immediate_and_descriptor_tensor const&>())),
                           immediate_and_descriptor_tensor::const_mdspan_type>);
static_assert(std::same_as<decltype(std::declval<immediate_and_descriptor_tensor&>().mdspec()),
                           immediate_and_descriptor_tensor::mdspan_type>);
static_assert(std::same_as<decltype(std::declval<immediate_and_descriptor_tensor const&>().mdspec()),
                           immediate_and_descriptor_tensor::const_mdspan_type>);
static_assert(read_only_tensor::immediately_readable);
static_assert(!read_only_tensor::immediately_writable);
static_assert(!read_only_tensor::deferred_readable);
static_assert(!read_only_tensor::deferred_writable);
static_assert(ImmediateTensorView<read_only_tensor>);
static_assert(TensorView<read_only_tensor>);
static_assert(!MutableImmediateTensorView<read_only_tensor>);
static_assert(!MutableTensorView<read_only_tensor>);
static_assert(std::same_as<decltype(std::declval<read_only_tensor&>().mdspec()), read_only_tensor::const_mdspan_type>);
static_assert(ReadTensorLease<storage_free_read_lease>);
static_assert(WriteTensorLease<storage_free_write_lease>);
static_assert(TensorAccessState<InstrumentedTensorAccessState>);
static_assert(!TensorAccessState<ThrowingMoveTensorAccessState>);
static_assert(MdspanLike<throwing_move_mdspan>);
static_assert(!std::is_nothrow_move_constructible_v<throwing_move_mdspan>);
static_assert(!CanFormReadMdspanLease<throwing_move_mdspan, InstrumentedTensorAccessState>);
static_assert(ReadTensorLease<instrumented_read_tensor_lease>);
static_assert(WriteTensorLease<instrumented_write_tensor_lease>);
static_assert(sizeof(storage_free_read_lease) == sizeof(void*));
static_assert(sizeof(storage_free_write_lease) == sizeof(void*));
static_assert(!HasStorageObserver<storage_free_read_lease>);
static_assert(!HasStorageObserver<storage_free_write_lease>);
static_assert(!CanBorrowReadFromRvalue<StorageFreeTensorView>);
static_assert(TensorView<DeferredStridedTensorView>);
static_assert(StridedTensorView<DeferredStridedTensorView>);
static_assert(!ImmediateTensorView<DeferredStridedTensorView>);

using row_major_matrix = DenseMatrix<int, RowMajor>;
using strided_matrix = typename row_major_matrix::template rebind_layout_type<stdex::layout_stride>;
static_assert(std::same_as<strided_matrix, StridedTensor<int, 2>>);

template <typename Span>
constexpr bool can_assign_element_v =
    std::is_assignable_v<typename std::remove_reference_t<Span>::reference,
                         std::remove_const_t<typename std::remove_reference_t<Span>::value_type>>;

TEST(TensorTest, DefaultMappingUsesColumnMajorHostStorage)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  EXPECT_EQ(tensor.extents().extent(0), exts.extent(0));
  EXPECT_EQ(tensor.extents().extent(1), exts.extent(1));
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.mapping().required_span_size(), 6);
  EXPECT_EQ(tensor.storage().size(), 6u);

  for (index_t i = 0; i < static_cast<index_t>(exts.extent(0)); ++i)
  {
    for (index_t j = 0; j < static_cast<index_t>(exts.extent(1)); ++j)
    {
      tensor[i, j] = static_cast<int>(i * static_cast<index_t>(exts.extent(1)) + j);
    }
  }

  auto const& storage = tensor.storage();
  std::vector<int> expected{0, 3, 1, 4, 2, 5};
  ASSERT_EQ(storage.size(), expected.size());
  for (std::size_t idx = 0; idx < expected.size(); ++idx)
  {
    EXPECT_EQ(storage[idx], expected[idx]);
  }

  EXPECT_EQ((tensor[1, 2]), expected.back());
  EXPECT_EQ(tensor.mapping().stride(0), 1);
  EXPECT_EQ(tensor.mapping().stride(1), 2);
}

TEST(HostBufferTest, CopyMoveAndResizePreserveInitializedPrefix)
{
  HostBuffer<int> source(4, 7);
  source[1] = 11;

  HostBuffer<int> copy = source;
  EXPECT_EQ(copy.size(), 4u);
  EXPECT_EQ(copy[0], 7);
  EXPECT_EQ(copy[1], 11);

  copy.resize(2);
  EXPECT_EQ(copy.size(), 2u);
  EXPECT_EQ(copy[0], 7);
  EXPECT_EQ(copy[1], 11);

  HostBuffer<int> moved = std::move(copy);
  EXPECT_TRUE(copy.empty());
  EXPECT_EQ(moved.size(), 2u);
  EXPECT_EQ(moved[0], 7);
  EXPECT_EQ(moved[1], 11);
}

TEST(HostBufferTest, PreservesAllocationAlignmentForSmallBuffers)
{
  uni20::HostBuffer<double> buffer(1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % uni20::HostStorage::allocation_alignment, 0);
}

TEST(HostBufferTest, HonorsOverAlignedElementTypes)
{
  struct alignas(128) over_aligned_value
  {
      double value;
  };

  uni20::HostBuffer<over_aligned_value> buffer(1);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % alignof(over_aligned_value), 0);
}

TEST(HostBufferTest, RejectsUnrepresentableByteSize)
{
  auto const size = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
  EXPECT_THROW(static_cast<void>(uni20::HostBuffer<double>{size}), std::bad_array_new_length);
}

TEST(HostBufferTest, ConstructsAndDestroysNontrivialElements)
{
  HostBuffer<std::string> buffer(2);
  EXPECT_EQ(buffer[0], "");
  EXPECT_EQ(buffer[1], "");

  buffer[0] = "host";
  buffer.resize(3);
  EXPECT_EQ(buffer[0], "host");
  EXPECT_EQ(buffer[2], "");
}

TEST(HostBufferTest, ReleasesNontrivialElementsWhenInitializationThrows)
{
  EXPECT_EQ(ThrowingAssignmentValue::live_count, 0);
  {
    ThrowingAssignmentValue value{7};
    ThrowingAssignmentValue::throw_on_assignment = true;
    EXPECT_THROW(static_cast<void>(HostBuffer<ThrowingAssignmentValue>{3, value}), std::runtime_error);
    EXPECT_EQ(ThrowingAssignmentValue::live_count, 1);

    ThrowingAssignmentValue::throw_on_assignment = false;
    HostBuffer<ThrowingAssignmentValue> source(3, value);
    EXPECT_EQ(ThrowingAssignmentValue::live_count, 4);
    ThrowingAssignmentValue::throw_on_assignment = true;
    EXPECT_THROW(static_cast<void>(HostBuffer<ThrowingAssignmentValue>{source}), std::runtime_error);
    EXPECT_EQ(ThrowingAssignmentValue::live_count, 4);
  }
  ThrowingAssignmentValue::throw_on_assignment = false;
  EXPECT_EQ(ThrowingAssignmentValue::live_count, 0);
}

TEST(TensorTest, StridesUseNormalizedImmediateAndDeferredMetadata)
{
  StridedTensor<int, 2> immediate(extents_2d{2, 3}, std::array<index_t, 2>{1, 5});
  DeferredStridedTensorView deferred(extents_2d{2, 3}, std::array<index_t, 2>{7, 2});

  EXPECT_EQ(strides(immediate), (std::array<index_t, 2>{1, 5}));
  EXPECT_EQ(strides(deferred), (std::array<index_t, 2>{7, 2}));
}

TEST(TensorTest, CanonicalDefaultAccessorStorageExposesData)
{
  tensor_type column_major(2, 3);
  RowMajorTensor<int, 2> row_major(2, 3);

  EXPECT_EQ(column_major.data(), column_major.mdspan().data_handle());
  EXPECT_EQ(std::as_const(column_major).data(), std::as_const(column_major).mdspan().data_handle());
  EXPECT_EQ(row_major.data(), row_major.mdspan().data_handle());
  EXPECT_EQ(std::as_const(row_major).data(), std::as_const(row_major).mdspan().data_handle());
}

TEST(TensorTest, ImmediateHandlePrecedesAvailableDescriptor)
{
  immediate_and_descriptor_tensor tensor(2, 3);
  tensor[1, 2] = 42;

  auto mutable_span = tensor.mdspec();
  auto const_span = std::as_const(tensor).mdspec();

  static_assert(MdspanLike<decltype(mutable_span)>);
  static_assert(MdspanLike<decltype(const_span)>);
  EXPECT_EQ(mutable_span.data_handle(), tensor.storage().data());
  EXPECT_EQ(const_span.data_handle(), tensor.storage().data());
  EXPECT_EQ((const_span[1, 2]), 42);
}

TEST(TensorTest, ReadOnlyImmediateStorageExposesConstMdspec)
{
  read_only_tensor tensor(2, 3);
  tensor.storage()[5] = 42;

  auto span = tensor.mdspec();

  static_assert(std::is_const_v<typename decltype(span)::element_type>);
  EXPECT_EQ((span[1, 2]), 42);
}

TEST(TensorTest, ImmediateLeasesDoNotRequireStorageObserver)
{
  std::array<int, 6> values{};
  StorageFreeTensorView view(values.data(), extents_2d{2, 3});

  {
    auto lease = acquire_host_write_access_sync(view);
    lease.mdspan()[1, 2] = 42;
  }

  auto const& const_view = view;
  auto lease = acquire_host_read_access_sync(const_view);
  EXPECT_EQ((lease.mdspan()[1, 2]), 42);
}

TEST(TensorTest, DynamicExtentsConstructorAcceptsOneExtentPerAxis)
{
  tensor_type tensor(2, 3);

  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
}

TEST(TensorTest, ScalarTensorAllocatesItsSoleElement)
{
  scalar_tensor_type scalar;

  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.size(), 1);
  EXPECT_EQ(scalar.storage().size(), 1u);

  scalar[] = 3.5;
  scalar_tensor_type const& const_scalar = scalar;
  EXPECT_DOUBLE_EQ(const_scalar[], 3.5);
}

TEST(BasicTensorTest, SupportsStaticMdspanExtents)
{
  using fixed_extents = stdex::extents<index_t, 2, 3>;
  BasicTensor<int, fixed_extents> tensor(fixed_extents{});

  static_assert(decltype(tensor)::rank() == 2);
  static_assert(decltype(tensor)::rank_dynamic() == 0);
  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
}

TEST(TensorTest, DenseMatrixDefaultsToColumnMajor)
{
  DenseMatrix<int> matrix(2, 3);

  static_assert(std::same_as<typename DenseMatrix<int>::layout_type, ColumnMajor>);
  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 3);
  EXPECT_EQ(matrix.mapping().stride(0), 1);
  EXPECT_EQ(matrix.mapping().stride(1), 2);
}

TEST(TensorTest, DenseMatrixSupportsRowMajorOwnership)
{
  DenseMatrix<int, RowMajor> matrix(2, 3);

  EXPECT_EQ(matrix.mapping().stride(0), 3);
  EXPECT_EQ(matrix.mapping().stride(1), 1);
}

TEST(TensorTest, CustomStridesAllocateFullSpan)
{
  extents_2d exts{2, 2};
  std::array<index_t, 2> strides{3, 1};
  strided_tensor_type tensor(exts, strides);

  EXPECT_EQ(tensor.mapping().stride(0), strides[0]);
  EXPECT_EQ(tensor.mapping().stride(1), strides[1]);
  EXPECT_EQ(tensor.mapping().required_span_size(), 5);
  EXPECT_EQ(tensor.storage().size(), 5u);

  tensor[0, 0] = 10;
  tensor[0, 1] = 11;
  tensor[1, 0] = 12;
  tensor[1, 1] = 13;

  auto const& storage = tensor.storage();
  EXPECT_EQ(storage[0], 10);
  EXPECT_EQ(storage[1], 11);
  EXPECT_EQ(storage[3], 12);
  EXPECT_EQ(storage[4], 13);
  EXPECT_EQ((tensor[1, 1]), 13);
}

TEST(TensorTest, AdoptedStorageMayRetainPaddingAndUnusedTail)
{
  extents_2d exts{2, 2};
  strided_tensor_type::mapping_type mapping(exts, std::array<index_t, 2>{1, 3});
  strided_tensor_type::storage_type storage(8, -1);
  int* original_storage = storage.data();
  storage[0] = 10;
  storage[1] = 11;
  storage[3] = 12;
  storage[4] = 13;

  auto tensor = strided_tensor_type::adopt_storage(mapping, std::move(storage));

  EXPECT_EQ(tensor.mutable_handle(), original_storage);
  EXPECT_EQ(tensor.storage().size(), 8u);
  EXPECT_EQ(tensor.size(), 5);
  EXPECT_EQ((tensor[0, 0]), 10);
  EXPECT_EQ((tensor[1, 0]), 11);
  EXPECT_EQ((tensor[0, 1]), 12);
  EXPECT_EQ((tensor[1, 1]), 13);

  auto released = std::move(tensor).release_storage();
  EXPECT_EQ(released.data(), original_storage);
  EXPECT_EQ(released.size(), 8u);
}

TEST(TensorTest, MappingBuilderSupportsLayoutLeft)
{
  extents_2d exts{2, 3};
  strided_tensor_type tensor(exts, layout::LayoutLeft());

  EXPECT_EQ(tensor.mapping().stride(0), 1);
  EXPECT_EQ(tensor.mapping().stride(1), 2);
  EXPECT_EQ(tensor.storage().size(), 6u);

  for (index_t j = 0; j < static_cast<index_t>(exts.extent(1)); ++j)
  {
    for (index_t i = 0; i < static_cast<index_t>(exts.extent(0)); ++i)
    {
      tensor[i, j] = static_cast<int>(j * 10 + i);
    }
  }

  auto const& storage = tensor.storage();
  std::vector<int> expected{0, 1, 10, 11, 20, 21};
  ASSERT_EQ(storage.size(), expected.size());
  for (std::size_t idx = 0; idx < expected.size(); ++idx)
  {
    EXPECT_EQ(storage[idx], expected[idx]);
  }

  EXPECT_EQ((tensor[1, 2]), 21);
}

TEST(TensorTest, CopyConstructionOwnsIndependentStorage)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 0] = 1;
  source[1, 2] = 6;

  tensor_type copy(source);

  EXPECT_EQ(copy.handle(), static_cast<int const*>(copy.storage().data()));
  EXPECT_NE(copy.handle(), source.handle());
  EXPECT_EQ(copy.extents().extent(0), 2);
  EXPECT_EQ(copy.extents().extent(1), 3);
  EXPECT_EQ((copy[0, 0]), 1);
  EXPECT_EQ((copy[1, 2]), 6);

  copy[0, 0] = 99;
  EXPECT_EQ((source[0, 0]), 1);
  EXPECT_EQ((copy[0, 0]), 99);
}

TEST(TensorTest, CopyAssignmentOwnsIndependentStorageAndMapping)
{
  strided_tensor_type source(extents_2d{2, 3}, std::array<index_t, 2>{4, 1});
  source[0, 0] = 3;
  source[1, 2] = 9;

  strided_tensor_type target(extents_2d{1, 1});
  target = source;

  EXPECT_EQ(target.handle(), static_cast<int const*>(target.storage().data()));
  EXPECT_NE(target.handle(), source.handle());
  EXPECT_EQ(target.mapping().stride(0), 4);
  EXPECT_EQ(target.mapping().stride(1), 1);
  EXPECT_EQ(target.storage().size(), source.storage().size());
  EXPECT_EQ((target[0, 0]), 3);
  EXPECT_EQ((target[1, 2]), 9);

  target[1, 2] = 42;
  EXPECT_EQ((source[1, 2]), 9);
  EXPECT_EQ((target[1, 2]), 42);
}

TEST(TensorTest, MoveConstructionRetainsOwnedMapping)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 1] = 7;
  source[1, 2] = 8;

  tensor_type moved(std::move(source));

  EXPECT_EQ(moved.handle(), static_cast<int const*>(moved.storage().data()));
  EXPECT_EQ(source.handle(), static_cast<int const*>(source.storage().data()));
  EXPECT_EQ((moved[0, 1]), 7);
  EXPECT_EQ((moved[1, 2]), 8);
}

TEST(TensorTest, MoveAssignmentRetainsOwnedMapping)
{
  tensor_type source(extents_2d{2, 3});
  source[0, 1] = 11;
  source[1, 2] = 12;

  tensor_type target(extents_2d{1, 1});
  target = std::move(source);

  EXPECT_EQ(target.handle(), static_cast<int const*>(target.storage().data()));
  EXPECT_EQ(source.handle(), static_cast<int const*>(source.storage().data()));
  EXPECT_EQ(target.extents().extent(0), 2);
  EXPECT_EQ(target.extents().extent(1), 3);
  EXPECT_EQ((target[0, 1]), 11);
  EXPECT_EQ((target[1, 2]), 12);
}

TEST(TensorTest, MdspanFromConstTensorIsReadOnly)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  auto mutable_span = tensor.mdspan();
  static_assert(std::is_same_v<typename decltype(mutable_span)::reference, int&>);
  mutable_span[0, 0] = 5;
  mutable_span[1, 2] = 17;

  tensor_type const& const_tensor = tensor;
  using const_span_type = decltype(const_tensor.mdspan());
  static_assert(std::is_same_v<typename const_span_type::reference, int const&>);
  static_assert(!can_assign_element_v<const_span_type const&>);

  auto span_from_mdspan = tensor.mdspan();
  static_assert(std::is_same_v<typename decltype(span_from_mdspan)::reference, int&>);
  static_assert(can_assign_element_v<decltype(span_from_mdspan) const&>);

  EXPECT_EQ((span_from_mdspan[0, 0]), 5);
  EXPECT_EQ((span_from_mdspan[1, 2]), 17);
}

TEST(TensorTest, ResolvedMdspansShareOwnedStorage)
{
  extents_2d exts{2, 3};
  tensor_type tensor(exts);

  auto span = tensor.mdspan();
  span[0, 0] = 9;
  span[1, 2] = 42;

  EXPECT_EQ(tensor.storage()[0], 9);
  EXPECT_EQ(tensor.storage()[5], 42);

  tensor_type const& const_tensor = tensor;
  auto const_span = const_tensor.mdspan();
  static_assert(!can_assign_element_v<decltype(const_span) const&>);

  EXPECT_EQ(span.data_handle(), tensor.mutable_handle());
  EXPECT_EQ(const_span.data_handle(), tensor.handle());
  EXPECT_EQ((const_span[0, 0]), 9);
  EXPECT_EQ((const_span[1, 2]), 42);
  EXPECT_EQ(tensor.backend_selector(), HostStorage::backend_selector());
}

TEST(TensorTest, ImmediateTensorAccessUsesNoOpTensorViewLeases)
{
  tensor_type tensor(extents_2d{2, 3});

  {
    auto acquisition = acquire_host_write_access_async(tensor);
    EXPECT_TRUE(acquisition.await_ready());
    auto lease = acquisition.await_resume();
    static_assert(MutableImmediateTensorView<decltype(lease)>);
    static_assert(ImmediateTensorView<decltype(std::as_const(lease))>);
    EXPECT_EQ(lease.backend_selector(), tensor.backend_selector());
    lease.mdspan()[1, 2] = 42;
  }

  auto lease = acquire_host_read_access_sync(std::as_const(tensor));
  static_assert(ImmediateTensorView<decltype(lease)>);
  static_assert(!MutableImmediateTensorView<decltype(lease)>);
  EXPECT_EQ(&lease.storage(), &tensor.storage());
  EXPECT_EQ((lease.mdspan()[1, 2]), 42);
}

TEST(TensorTest, ImmediateMdspanAccessUsesPolicyFreeLeases)
{
  std::array<int, 4> data{};
  access_test_mutable_mdspan mutable_span{data.data(), extents_2d{2, 2}};

  {
    auto lease = acquire_host_write_access_sync(mutable_span);
    static_assert(HostWriteMdspanLease<decltype(lease)>);
    lease.mdspan()[1, 0] = 42;
  }

  access_test_const_mdspan const_span{data.data(), extents_2d{2, 2}};
  auto lease = acquire_host_read_access_sync(const_span);
  static_assert(HostReadMdspanLease<decltype(lease)>);
  EXPECT_EQ((lease.mdspan()[1, 0]), 42);
}

TEST(TensorTest, GenericMdspanLeasesReleaseAndMoveAccessState)
{
  AccessStateCounters counters;
  std::array<int, 4> data{};

  {
    instrumented_write_mdspan_lease source{InstrumentedTensorAccessState{counters},
                                           access_test_mutable_mdspan{data.data(), extents_2d{2, 2}}};
    instrumented_write_mdspan_lease lease{std::move(source)};
    source.release();
    EXPECT_EQ(counters.completed_releases, 0);

    lease.release();
    lease.release();
    EXPECT_EQ(counters.completed_releases, 1);
  }

  EXPECT_EQ(counters.completed_releases, 1);
}

TEST(TensorTest, GenericReadTensorLeaseReleaseIsIdempotent)
{
  AccessStateCounters counters;
  std::array<int, 4> data{};

  {
    instrumented_read_tensor_lease lease{InstrumentedTensorAccessState{counters},
                                         access_test_const_mdspan{data.data(), extents_2d{2, 2}},
                                         access_test_selector{linalg::CpuReferenceBackend{}}};

    lease.release();
    EXPECT_EQ(counters.release_calls, 1);
    EXPECT_EQ(counters.completed_releases, 1);

    lease.release();
    EXPECT_EQ(counters.release_calls, 1);
    EXPECT_EQ(counters.completed_releases, 1);
  }

  EXPECT_EQ(counters.release_calls, 2);
  EXPECT_EQ(counters.completed_releases, 1);
}

TEST(TensorTest, GenericWriteTensorLeaseMovesTransferAccessState)
{
  AccessStateCounters counters;
  std::array<int, 4> source_data{};
  std::array<int, 4> destination_data{};

  {
    instrumented_write_tensor_lease source{InstrumentedTensorAccessState{counters},
                                           access_test_mutable_mdspan{source_data.data(), extents_2d{2, 2}},
                                           access_test_const_mdspan{source_data.data(), extents_2d{2, 2}},
                                           access_test_selector{linalg::CpuReferenceBackend{}}};

    instrumented_write_tensor_lease moved{std::move(source)};
    source.release();
    EXPECT_EQ(counters.completed_releases, 0);

    instrumented_write_tensor_lease destination{InstrumentedTensorAccessState{counters},
                                                access_test_mutable_mdspan{destination_data.data(), extents_2d{2, 2}},
                                                access_test_const_mdspan{destination_data.data(), extents_2d{2, 2}},
                                                access_test_selector{linalg::CpuReferenceBackend{}}};

    destination = std::move(moved);
    EXPECT_EQ(counters.completed_releases, 1);

    moved.release();
    EXPECT_EQ(counters.completed_releases, 1);

    EXPECT_EQ(destination.mdspan().data_handle(), source_data.data());
    destination.mdspan()[0, 1] = 42;
    EXPECT_EQ((destination.mdspan()[0, 1]), 42);

    destination.release();
    EXPECT_EQ(counters.completed_releases, 2);

    destination.release();
    EXPECT_EQ(counters.completed_releases, 2);
  }

  EXPECT_EQ(counters.completed_releases, 2);
}

TEST(TensorTest, TraceFormattingUsesPresentationTensorArt)
{
  tensor_type tensor(extents_2d{2, 2});
  tensor[0, 0] = 1;
  tensor[0, 1] = 20;
  tensor[1, 0] = 300;
  tensor[1, 1] = 4;

  auto opts = trace::get_formatting_options("tensor-format-test");
  opts.set_color_output(trace::FormattingOptions::ColorOptions::no);

  EXPECT_EQ(trace::formatValue(tensor, opts), "shape=(2, 2)\n"
                                              "\xE2\x8E\xA1   1 20 \xE2\x8E\xA4\n"
                                              "\xE2\x8E\xA3 300  4 \xE2\x8E\xA6");
}

} // namespace
