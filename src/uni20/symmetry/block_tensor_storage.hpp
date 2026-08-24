/**
 * \file block_tensor_storage.hpp
 * \ingroup symmetry
 * \brief Defines the first sparse BlockTensor storage policies.
 */

#pragma once

#include <uni20/async/async.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/mdspec.hpp>
#include <uni20/storage/vectorstorage.hpp>
#include <uni20/symmetry/block_key.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Execute block-level work serially on the calling thread.
struct SerialBlockExecution
{};

/// \brief Execute independent block-level work through the active scheduler's synchronous batch interface.
struct SchedulerBatchBlockExecution
{};

/// \brief Storage policy for a BlockTensor container.
/// \tparam Storage Candidate policy type.
template <class Storage>
concept BlockTensorStorage = requires {
  typename Storage::leaf_storage_policy;
  typename Storage::backend_selector_type;
  typename Storage::block_execution_policy;
  { Storage::stores_all_legal_blocks } -> std::convertible_to<bool>;
  { Storage::is_distributed } -> std::convertible_to<bool>;
  { Storage::backend_selector() } -> std::same_as<typename Storage::backend_selector_type>;
};

/// \brief BlockTensor storage whose stored key set may omit legal blocks.
/// \tparam Storage Candidate policy type.
template <class Storage>
concept SparseBlockStorage = BlockTensorStorage<Storage> && !Storage::stores_all_legal_blocks;

/// \brief BlockTensor storage which contains every symmetry-legal block.
/// \tparam Storage Candidate policy type.
template <class Storage>
concept CompleteBlockStorage = BlockTensorStorage<Storage> && Storage::stores_all_legal_blocks;

/// \brief Storage policy instantiated for one block scalar and tensor order.
/// \tparam Storage Candidate block storage policy.
/// \tparam T Numerical block element type.
/// \tparam KeyCoordinateCount Number of stored block-selection coordinates.
/// \tparam DenseBlockOrder Number of numerical axes in each dense block.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept BlockTensorStorageFor =
    BlockTensorStorage<Storage> &&
    requires(typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>& storage,
             typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder> const& const_storage,
             std::array<std::size_t, DenseBlockOrder> const& extents) {
      typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::mutable_block_type;
      typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::const_block_type;
      { const_storage.size() } -> std::same_as<std::size_t>;
      { const_storage.keys() } -> std::same_as<std::span<BlockKey<KeyCoordinateCount> const>>;
      {
        storage.block(std::size_t{}, extents)
      }
      -> std::same_as<typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::mutable_block_type>;
      {
        const_storage.block(std::size_t{}, extents)
      } -> std::same_as<typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::const_block_type>;
    } &&
    MutableRankedTensorView<
        typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::mutable_block_type,
        DenseBlockOrder> &&
    RankedTensorView<typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::const_block_type,
                     DenseBlockOrder> &&
    std::same_as<tensor_element_t<
                     typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::mutable_block_type>,
                 T> &&
    std::same_as<typename tensor_mdspec_t<typename Storage::template storage_t<
                     T, KeyCoordinateCount, DenseBlockOrder>::const_block_type>::element_type,
                 T const>;

/// \brief Local BlockTensor storage whose block TensorViews are immediately accessible.
/// \tparam Storage Candidate block storage policy.
/// \tparam T Numerical block element type.
/// \tparam KeyCoordinateCount Number of stored block-selection coordinates.
/// \tparam DenseBlockOrder Number of numerical axes in each dense block.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept ImmediateLocalBlockStorageFor =
    BlockTensorStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> && !Storage::is_distributed &&
    MutableRankedImmediateTensorView<
        typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::mutable_block_type,
        DenseBlockOrder> &&
    RankedImmediateTensorView<
        typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::const_block_type,
        DenseBlockOrder>;

/// \brief Local BlockTensor storage with one independently scheduled async value per block.
/// \tparam Storage Candidate block storage policy.
/// \tparam T Numerical block element type.
/// \tparam KeyCoordinateCount Number of stored block-selection coordinates.
/// \tparam DenseBlockOrder Number of numerical axes in each dense block.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept AsyncLocalBlockStorageFor =
    BlockTensorStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> && !Storage::is_distributed &&
    requires(typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>& storage,
             typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder> const& const_storage) {
      typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::block_value_type;
      typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::async_block_type;
      {
        storage.async_block(std::size_t{})
      }
      -> std::same_as<typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::async_block_type&>;
      {
        const_storage.async_block(std::size_t{})
      } -> std::same_as<
          typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::async_block_type const&>;
    } &&
    std::same_as<
        typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::async_block_type,
        async::Async<typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::block_value_type>> &&
    MutableRankedTensorView<
        typename Storage::template storage_t<T, KeyCoordinateCount, DenseBlockOrder>::block_value_type,
        DenseBlockOrder>;

/// \brief Distributed BlockTensor storage refinement.
/// \details A concrete distributed policy will add placement and ownership
///          operations while retaining the common block-metadata contract.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept DistributedBlockStorageFor =
    BlockTensorStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> && Storage::is_distributed;

namespace detail
{

template <std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder> struct SparseBlockSpec
{
    BlockKey<KeyCoordinateCount> key;
    std::array<std::size_t, DenseBlockOrder> extents;
};

template <class Tensor, std::size_t Order, std::size_t... I>
auto make_block_tensor(std::array<std::size_t, Order> const& extents, std::index_sequence<I...>) -> Tensor
{
  return Tensor(static_cast<index_type>(extents[I])...);
}

template <class Tensor, std::size_t Order>
auto make_block_tensor(std::array<std::size_t, Order> const& extents) -> Tensor
{
  return make_block_tensor<Tensor>(extents, std::make_index_sequence<Order>{});
}

template <class Extents, std::size_t Order, std::size_t... I>
auto make_extents(std::array<std::size_t, Order> const& extents, std::index_sequence<I...>) -> Extents
{
  return Extents(static_cast<typename Extents::index_type>(extents[I])...);
}

template <class Extents, std::size_t Order> auto make_extents(std::array<std::size_t, Order> const& extents) -> Extents
{
  return make_extents<Extents>(extents, std::make_index_sequence<Order>{});
}

inline auto checked_block_size(std::span<std::size_t const> extents) -> std::size_t
{
  std::size_t size = 1;
  for (std::size_t const extent : extents)
  {
    if (extent != 0 && size > std::numeric_limits<std::size_t>::max() / extent)
    {
      throw std::length_error("BlockTensor block size overflows size_t");
    }
    size *= extent;
  }
  return size;
}

template <class Storage, class T> auto make_storage(std::size_t size) -> typename Storage::template storage_t<T>
{
  using storage_type = typename Storage::template storage_t<T>;
  if constexpr (std::default_initializable<storage_type> && requires(storage_type& storage) { storage.resize(size); })
  {
    storage_type storage;
    storage.resize(size);
    return storage;
  }
  else
  {
    static_assert(std::constructible_from<storage_type, std::size_t>,
                  "packed BlockTensor leaf storage must be resizable or constructible from a size");
    return storage_type(size);
  }
}

template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder, class LeafStorage>
class SeparateSparseBlockStorageData {
  public:
    using key_type = BlockKey<KeyCoordinateCount>;
    using block_type = ColumnMajorTensor<T, DenseBlockOrder, LeafStorage>;
    using mutable_mdspan_type = typename block_type::mdspan_type;
    using const_mdspan_type = typename block_type::const_mdspan_type;
    using mutable_block_type = MdspecTensorView<mutable_mdspan_type, const_mdspan_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspan_type, const_mdspan_type, LeafStorage>;

    static_assert(block_type::immediately_readable && block_type::immediately_writable,
                  "separate sparse block storage currently requires immediate leaf access");

    explicit SeparateSparseBlockStorageData(
        std::vector<SparseBlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
    {
      keys_.reserve(specs.size());
      blocks_.reserve(specs.size());
      for (auto const& spec : specs)
      {
        keys_.push_back(spec.key);
        blocks_.push_back(make_block_tensor<block_type>(spec.extents));
      }
    }

    auto size() const noexcept -> std::size_t { return keys_.size(); }
    auto keys() const noexcept -> std::span<key_type const> { return keys_; }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const&) -> mutable_block_type
    {
      return mutable_block_type{blocks_[ordinal].mdspan(), std::as_const(blocks_[ordinal]).mdspan()};
    }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const&) const -> const_block_type
    {
      auto span = blocks_[ordinal].mdspan();
      return const_block_type{span, span};
    }

  private:
    std::vector<key_type> keys_;
    std::vector<block_type> blocks_;
};

template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder, class LeafStorage>
class PackedSparseBlockStorageData {
  public:
    using key_type = BlockKey<KeyCoordinateCount>;
    using buffer_type = typename LeafStorage::template storage_t<T>;
    using extents_type = stdex::dextents<index_type, DenseBlockOrder>;
    using mapping_type = typename ColumnMajor::template mapping<extents_type>;
    using accessor_type = stdex::default_accessor<T>;
    using const_accessor_type = stdex::default_accessor<T const>;
    using mutable_mdspan_type = stdex::mdspan<T, extents_type, ColumnMajor, accessor_type>;
    using const_mdspan_type = stdex::mdspan<T const, extents_type, ColumnMajor, const_accessor_type>;
    using mutable_block_type = MdspecTensorView<mutable_mdspan_type, const_mdspan_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspan_type, const_mdspan_type, LeafStorage>;

    static_assert(std::same_as<detail::tensor_accessor_factory_t<LeafStorage>, DefaultAccessorFactory>,
                  "packed sparse block storage currently requires default-accessor leaf storage");
    static_assert(
        requires(buffer_type& buffer, buffer_type const& const_buffer) {
          { LeafStorage::make_handle(buffer) } -> std::same_as<T*>;
          { LeafStorage::make_handle(const_buffer) } -> std::same_as<T const*>;
        }, "packed sparse block storage currently requires immediate contiguous host leaf storage");

    explicit PackedSparseBlockStorageData(
        std::vector<SparseBlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
        : PackedSparseBlockStorageData(make_layout(specs))
    {}

    auto size() const noexcept -> std::size_t { return keys_.size(); }
    auto keys() const noexcept -> std::span<key_type const> { return keys_; }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) -> mutable_block_type
    {
      auto const block_extents = make_extents<extents_type>(extents);
      auto const mutable_handle = accessor_type{}.offset(LeafStorage::make_handle(buffer_), offsets_[ordinal]);
      auto const const_handle =
          const_accessor_type{}.offset(LeafStorage::make_handle(std::as_const(buffer_)), offsets_[ordinal]);
      return mutable_block_type{mutable_mdspan_type{mutable_handle, mapping_type(block_extents)},
                                const_mdspan_type{const_handle, mapping_type(block_extents)}};
    }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) const -> const_block_type
    {
      auto const block_extents = make_extents<extents_type>(extents);
      auto const handle = const_accessor_type{}.offset(LeafStorage::make_handle(buffer_), offsets_[ordinal]);
      auto span = const_mdspan_type{handle, mapping_type(block_extents)};
      return const_block_type{span, span};
    }

    auto buffer() noexcept -> buffer_type& { return buffer_; }
    auto buffer() const noexcept -> buffer_type const& { return buffer_; }
    auto offsets() const noexcept -> std::span<std::size_t const> { return offsets_; }

  private:
    struct Layout
    {
        std::vector<key_type> keys;
        std::vector<std::size_t> offsets;
    };

    static auto make_layout(std::vector<SparseBlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs) -> Layout
    {
      Layout layout;
      layout.keys.reserve(specs.size());
      layout.offsets.reserve(specs.size() + 1);
      layout.offsets.push_back(0);

      for (auto const& spec : specs)
      {
        layout.keys.push_back(spec.key);
        std::size_t const block_size = checked_block_size(spec.extents);
        if (layout.offsets.back() > std::numeric_limits<std::size_t>::max() - block_size)
        {
          throw std::length_error("packed BlockTensor size overflows size_t");
        }
        layout.offsets.push_back(layout.offsets.back() + block_size);
      }
      return layout;
    }

    explicit PackedSparseBlockStorageData(Layout layout)
        : keys_(std::move(layout.keys)), offsets_(std::move(layout.offsets)),
          buffer_(make_storage<LeafStorage, T>(offsets_.back()))
    {}

    std::vector<key_type> keys_;
    std::vector<std::size_t> offsets_;
    buffer_type buffer_;
};

/// \brief Borrowed descriptor retaining the identity of one async dense block.
/// \details The descriptor itself does not extend the owning BlockTensor's
///          lifetime. A read or write buffer obtained from the referenced
///          async value does retain storage and epoch ownership for submitted work.
template <class AsyncBlock> class AsyncBlockDataDescriptor {
  public:
    using async_block_type = AsyncBlock;

    explicit AsyncBlockDataDescriptor(async_block_type& block) noexcept : block_(&block) {}

    /// \brief Return the async block whose epoch governs this descriptor.
    auto async_block() const noexcept -> async_block_type& { return *block_; }

  private:
    async_block_type* block_;
};

template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder, class LeafStorage>
class AsyncSeparateSparseBlockStorageData {
  public:
    using key_type = BlockKey<KeyCoordinateCount>;
    using block_value_type = ColumnMajorTensor<T, DenseBlockOrder, LeafStorage>;
    using async_block_type = async::Async<block_value_type>;
    using extents_type = stdex::dextents<index_type, DenseBlockOrder>;
    using mapping_type = typename ColumnMajor::template mapping<extents_type>;
    using accessor_type = stdex::default_accessor<T>;
    using const_accessor_type = stdex::default_accessor<T const>;
    using mutable_descriptor_type = AsyncBlockDataDescriptor<async_block_type>;
    using const_descriptor_type = AsyncBlockDataDescriptor<async_block_type const>;
    using mutable_mdspec_type = mdspec<T, extents_type, ColumnMajor, accessor_type, mutable_descriptor_type>;
    using const_mdspec_type = mdspec<T const, extents_type, ColumnMajor, const_accessor_type, const_descriptor_type>;
    using mutable_block_type = MdspecTensorView<mutable_mdspec_type, const_mdspec_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspec_type, const_mdspec_type, LeafStorage>;

    static_assert(block_value_type::immediately_readable && block_value_type::immediately_writable,
                  "async separate sparse block storage currently requires immediate leaf access");
    static_assert(std::same_as<detail::tensor_accessor_factory_t<LeafStorage>, DefaultAccessorFactory>,
                  "async separate sparse block storage currently requires default-accessor leaf storage");

    explicit AsyncSeparateSparseBlockStorageData(
        std::vector<SparseBlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
    {
      keys_.reserve(specs.size());
      blocks_.reserve(specs.size());
      for (auto const& spec : specs)
      {
        keys_.push_back(spec.key);
        blocks_.emplace_back(make_block_tensor<block_value_type>(spec.extents));
      }
    }

    auto size() const noexcept -> std::size_t { return keys_.size(); }
    auto keys() const noexcept -> std::span<key_type const> { return keys_; }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) -> mutable_block_type
    {
      auto mapping = mapping_type(make_extents<extents_type>(extents));
      return mutable_block_type{
          mutable_mdspec_type{mutable_descriptor_type{blocks_[ordinal]}, mapping, accessor_type{}},
          const_mdspec_type{const_descriptor_type{std::as_const(blocks_[ordinal])}, mapping, const_accessor_type{}}};
    }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) const -> const_block_type
    {
      auto span = const_mdspec_type{const_descriptor_type{blocks_[ordinal]},
                                    mapping_type(make_extents<extents_type>(extents)), const_accessor_type{}};
      return const_block_type{span, span};
    }

    auto async_block(std::size_t ordinal) noexcept -> async_block_type& { return blocks_[ordinal]; }
    auto async_block(std::size_t ordinal) const noexcept -> async_block_type const& { return blocks_[ordinal]; }

  private:
    std::vector<key_type> keys_;
    std::vector<async_block_type> blocks_;
};

} // namespace detail

/// \brief Sparse storage with one independently owning dense Tensor per stored block.
/// \tparam LeafStorage Tensor storage policy used by each dense block.
template <class LeafStorage = VectorStorage> struct SeparateSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::SeparateSparseBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Sparse storage whose separate dense blocks may be processed in a scheduler batch.
/// \details Algorithms must place all writes to one logical output block in the
///          same batch item. The batch call is synchronous, so the BlockTensor
///          remains an ordinary immediate value after the operation returns.
/// \tparam LeafStorage Tensor storage policy used by each dense block.
template <class LeafStorage = VectorStorage> struct ParallelSeparateSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SchedulerBatchBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::SeparateSparseBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Sparse storage packing every stored dense block into one leaf buffer.
/// \tparam LeafStorage Immediate host Tensor storage policy used by the packed buffer.
template <class LeafStorage = VectorStorage> struct PackedSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t = detail::PackedSparseBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Sparse storage with one independently scheduled dense Tensor per block.
/// \details `block()` returns a TensorView over borrowed mdspec metadata.
///          `async_block()` exposes the retained async value used to create read
///          and write capabilities.
/// \tparam LeafStorage Immediate host Tensor storage policy used by each block.
template <class LeafStorage = VectorStorage> struct AsyncSeparateSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::AsyncSeparateSparseBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

static_assert(SparseBlockStorage<SeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<ParallelSeparateSparseBlockStorage<>>);
static_assert(SparseBlockStorage<PackedSparseBlockStorage<>>);
static_assert(SparseBlockStorage<AsyncSeparateSparseBlockStorage<>>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(AsyncLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(AsyncLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 4, 0>);

} // namespace uni20
