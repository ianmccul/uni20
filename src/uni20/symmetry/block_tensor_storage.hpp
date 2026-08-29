/**
 * \file block_tensor_storage.hpp
 * \ingroup symmetry
 * \brief Defines local sparse and packed-complete BlockTensor storage policies.
 */

#pragma once

#include <uni20/async/async.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/diagonal_accessor.hpp>
#include <uni20/mdspan/generated_layout.hpp>
#include <uni20/mdspan/mdspec.hpp>
#include <uni20/storage/host_storage.hpp>
#include <uni20/symmetry/block_key.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>
#include <uni20/tensor/tensor.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <numeric>
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

/// \brief BlockTensor storage whose stored numerical blocks are generalized diagonals.
/// \details Logical block presence remains the ordinary explicit BlockTensor
///          key set. This refinement describes only the compressed numerical
///          representation within each stored block.
/// \tparam Storage Candidate policy type.
template <class Storage>
concept DiagonalBlockStorage = BlockTensorStorage<Storage> && requires {
  { Storage::stores_diagonal_blocks } -> std::convertible_to<bool>;
} && Storage::stores_diagonal_blocks;

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

/// \brief BlockTensor storage whose blocks reside within one local process.
/// \details This refinement is independent of immediate versus descriptor-backed
///          access and of inline versus independently scheduled block execution.
/// \tparam Storage Candidate block storage policy.
/// \tparam T Numerical block element type.
/// \tparam KeyCoordinateCount Number of stored block-selection coordinates.
/// \tparam DenseBlockOrder Number of numerical axes in each dense block.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept LocalBlockStorageFor =
    BlockTensorStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> && !Storage::is_distributed;

/// \brief Local BlockTensor storage whose block TensorViews are immediately accessible.
/// \tparam Storage Candidate block storage policy.
/// \tparam T Numerical block element type.
/// \tparam KeyCoordinateCount Number of stored block-selection coordinates.
/// \tparam DenseBlockOrder Number of numerical axes in each dense block.
template <class Storage, typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
concept ImmediateLocalBlockStorageFor =
    LocalBlockStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> &&
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
    LocalBlockStorageFor<Storage, T, KeyCoordinateCount, DenseBlockOrder> &&
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

template <std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder> struct BlockSpec
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

template <std::size_t Rank>
constexpr auto diagonal_extent(std::array<std::size_t, Rank> const& extents) noexcept -> std::size_t
{
  if constexpr (Rank == 0)
  {
    return 1;
  }
  else
  {
    return *std::ranges::min_element(extents);
  }
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

template <class Storage, class T>
auto make_storage_like(typename Storage::template storage_t<T> const& prototype, std::size_t size) ->
    typename Storage::template storage_t<T>
{
  if constexpr (requires {
                  {
                    Storage::template make_storage_like<T>(prototype, size)
                  } -> std::same_as<typename Storage::template storage_t<T>>;
                })
  {
    return Storage::template make_storage_like<T>(prototype, size);
  }
  else
  {
    return make_storage<Storage, T>(size);
  }
}

template <class Storage, class T, class = void> struct PackedStorageType
{
    using type = typename Storage::template storage_t<T>;
};

template <class Storage, class T>
struct PackedStorageType<Storage, T, std::void_t<typename Storage::template packed_storage_t<T>>>
{
    using type = typename Storage::template packed_storage_t<T>;
};

template <class Storage, class T> using packed_storage_t = typename PackedStorageType<Storage, T>::type;

template <class Storage, class T>
auto make_packed_storage(std::size_t size, std::span<std::size_t const> offsets) -> packed_storage_t<Storage, T>
{
  if constexpr (requires {
                  {
                    Storage::template make_packed_storage<T>(size, offsets)
                  } -> std::same_as<packed_storage_t<Storage, T>>;
                })
  {
    return Storage::template make_packed_storage<T>(size, offsets);
  }
  else
  {
    return make_storage<Storage, T>(size);
  }
}

template <class Storage, class T>
auto make_packed_storage_like(packed_storage_t<Storage, T> const& prototype, std::size_t size,
                              std::span<std::size_t const> offsets) -> packed_storage_t<Storage, T>
{
  if constexpr (requires {
                  {
                    Storage::template make_packed_storage_like<T>(prototype, size, offsets)
                  } -> std::same_as<packed_storage_t<Storage, T>>;
                })
  {
    return Storage::template make_packed_storage_like<T>(prototype, size, offsets);
  }
  else
  {
    return make_storage_like<Storage, T>(prototype, size);
  }
}

template <class Storage, class Buffer>
decltype(auto) make_packed_data_descriptor(Buffer& buffer, std::size_t ordinal, std::size_t offset)
{
  if constexpr (requires { Storage::make_packed_data_descriptor(buffer, ordinal); })
    return Storage::make_packed_data_descriptor(buffer, ordinal);
  else
    return Storage::make_data_descriptor(buffer).offset_by(offset);
}

template <class Storage, class T, class Buffer>
void initialize_packed_padding(Buffer& buffer, std::span<std::size_t const> offsets,
                               std::span<std::size_t const> block_ends)
{
  if constexpr (requires { Storage::initialize_packed_padding(buffer, offsets, block_ends); })
  {
    Storage::initialize_packed_padding(buffer, offsets, block_ends);
  }
  else
  {
    static_assert(
        requires { Storage::make_handle(buffer); },
        "aligned packed storage must provide immediate access or initialize_packed_padding");
    auto* data = Storage::make_handle(buffer);
    for (std::size_t ordinal = 0; ordinal < block_ends.size(); ++ordinal)
      std::fill(data + block_ends[ordinal], data + offsets[ordinal + 1], T{});
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

    explicit SeparateSparseBlockStorageData(std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
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

template <class LeafStorage, typename T>
concept ImmediatePackedLeafStorage =
    std::default_initializable<tensor_accessor_factory_t<LeafStorage>> &&
    requires(typename LeafStorage::template storage_t<T>& buffer,
             typename LeafStorage::template storage_t<T> const& const_buffer,
             tensor_accessor_factory_t<LeafStorage> const& accessor_factory) {
      {
        LeafStorage::make_handle(buffer)
      }
      -> std::convertible_to<typename tensor_accessor_factory_t<LeafStorage>::template accessor_t<T>::data_handle_type>;
      {
        LeafStorage::make_handle(const_buffer)
      } -> std::convertible_to<
            typename tensor_accessor_factory_t<LeafStorage>::template accessor_t<T const>::data_handle_type>;
      {
        accessor_factory.template make_accessor<T>(buffer)
      } -> std::same_as<typename tensor_accessor_factory_t<LeafStorage>::template accessor_t<T>>;
      {
        accessor_factory.template make_accessor<T const>(const_buffer)
      } -> std::same_as<typename tensor_accessor_factory_t<LeafStorage>::template accessor_t<T const>>;
    };

template <typename T, std::size_t DenseBlockOrder, class LeafStorage,
          bool Immediate = ImmediatePackedLeafStorage<LeafStorage, T>>
struct PackedBlockViewTraits;

template <typename T, std::size_t DenseBlockOrder, class LeafStorage>
struct PackedBlockViewTraits<T, DenseBlockOrder, LeafStorage, true>
{
    using buffer_type = packed_storage_t<LeafStorage, T>;
    using extents_type = stdex::dextents<index_type, DenseBlockOrder>;
    using mapping_type = typename ColumnMajor::template mapping<extents_type>;
    using accessor_factory_type = tensor_accessor_factory_t<LeafStorage>;
    using accessor_type = typename accessor_factory_type::template accessor_t<T>;
    using const_accessor_type = typename accessor_factory_type::template accessor_t<T const>;
    using mutable_mdspec_type = stdex::mdspan<T, extents_type, ColumnMajor, accessor_type>;
    using const_mdspec_type = stdex::mdspan<T const, extents_type, ColumnMajor, const_accessor_type>;
    using mutable_block_type = MdspecTensorView<mutable_mdspec_type, const_mdspec_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspec_type, const_mdspec_type, LeafStorage>;

    static auto mutable_block(buffer_type& buffer, std::size_t offset,
                              extents_type const& extents) -> mutable_block_type
    {
      accessor_factory_type accessor_factory;
      auto accessor = accessor_factory.template make_accessor<T>(buffer);
      auto const_accessor = accessor_factory.template make_accessor<T const>(std::as_const(buffer));
      auto mutable_handle = accessor.offset(LeafStorage::make_handle(buffer), offset);
      auto const_handle = const_accessor.offset(LeafStorage::make_handle(std::as_const(buffer)), offset);
      return mutable_block_type{mutable_mdspec_type{mutable_handle, mapping_type(extents), accessor},
                                const_mdspec_type{const_handle, mapping_type(extents), const_accessor}};
    }

    static auto const_block(buffer_type const& buffer, std::size_t offset,
                            extents_type const& extents) -> const_block_type
    {
      accessor_factory_type accessor_factory;
      auto accessor = accessor_factory.template make_accessor<T const>(buffer);
      auto handle = accessor.offset(LeafStorage::make_handle(buffer), offset);
      auto span = const_mdspec_type{handle, mapping_type(extents), accessor};
      return const_block_type{span, span};
    }
};

template <typename T, std::size_t DenseBlockOrder, class LeafStorage>
struct PackedBlockViewTraits<T, DenseBlockOrder, LeafStorage, false>
{
    using buffer_type = packed_storage_t<LeafStorage, T>;
    using extents_type = stdex::dextents<index_type, DenseBlockOrder>;
    using mapping_type = typename ColumnMajor::template mapping<extents_type>;
    using accessor_factory_type = tensor_accessor_factory_t<LeafStorage>;
    using accessor_type = tensor_device_accessor_t<accessor_factory_type, T>;
    using const_accessor_type = tensor_device_accessor_t<accessor_factory_type, T const>;
    using descriptor_type =
        decltype(make_packed_data_descriptor<LeafStorage>(std::declval<buffer_type&>(), std::size_t{}, std::size_t{}));
    using const_descriptor_type = decltype(make_packed_data_descriptor<LeafStorage>(std::declval<buffer_type const&>(),
                                                                                    std::size_t{}, std::size_t{}));
    using mutable_mdspec_type = mdspec<T, extents_type, ColumnMajor, accessor_type, descriptor_type>;
    using const_mdspec_type = mdspec<T const, extents_type, ColumnMajor, const_accessor_type, const_descriptor_type>;
    using mutable_block_type = MdspecTensorView<mutable_mdspec_type, const_mdspec_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspec_type, const_mdspec_type, LeafStorage>;

    static_assert(std::default_initializable<accessor_factory_type>,
                  "packed descriptor-backed storage requires a default-constructible accessor factory");
    template <class ElementType>
    static auto
    device_accessor(buffer_type const& buffer) -> tensor_device_accessor_t<accessor_factory_type, ElementType>
    {
      accessor_factory_type accessor_factory;
      if constexpr (requires { accessor_factory.template make_device_accessor<ElementType>(buffer); })
        return accessor_factory.template make_device_accessor<ElementType>(buffer);
      else
        return accessor_factory.template make_accessor<ElementType>(buffer);
    }

    static auto mutable_block(buffer_type& buffer, std::size_t offset, extents_type const& extents,
                              std::size_t ordinal) -> mutable_block_type
    {
      auto descriptor = make_packed_data_descriptor<LeafStorage>(buffer, ordinal, offset);
      auto const_descriptor = make_packed_data_descriptor<LeafStorage>(std::as_const(buffer), ordinal, offset);
      return mutable_block_type{
          mutable_mdspec_type{std::move(descriptor), mapping_type(extents), device_accessor<T>(buffer)},
          const_mdspec_type{std::move(const_descriptor), mapping_type(extents), device_accessor<T const>(buffer)}};
    }

    static auto const_block(buffer_type const& buffer, std::size_t offset, extents_type const& extents,
                            std::size_t ordinal) -> const_block_type
    {
      auto descriptor = make_packed_data_descriptor<LeafStorage>(buffer, ordinal, offset);
      auto span = const_mdspec_type{std::move(descriptor), mapping_type(extents), device_accessor<T const>(buffer)};
      return const_block_type{span, span};
    }
};

template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder, class LeafStorage,
          std::size_t BlockAlignment>
class PackedBlockStorageData {
  public:
    using key_type = BlockKey<KeyCoordinateCount>;
    using buffer_type = packed_storage_t<LeafStorage, T>;
    using view_traits = PackedBlockViewTraits<T, DenseBlockOrder, LeafStorage>;
    using extents_type = typename view_traits::extents_type;
    using mutable_block_type = typename view_traits::mutable_block_type;
    using const_block_type = typename view_traits::const_block_type;

    static_assert(BlockAlignment > 0, "packed block alignment must be positive");
    static_assert(
        [] {
          if constexpr (BlockAlignment == 1)
            return true;
          else if constexpr (requires { LeafStorage::allocation_alignment; })
            return LeafStorage::allocation_alignment % BlockAlignment == 0;
          else
            return false;
        }(),
        "nontrivial packed block alignment must divide the leaf allocation alignment");
    explicit PackedBlockStorageData(std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
        : PackedBlockStorageData(make_layout(specs))
    {}

    auto size() const noexcept -> std::size_t { return keys_.size(); }
    auto keys() const noexcept -> std::span<key_type const> { return keys_; }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) -> mutable_block_type
    {
      if constexpr (ImmediatePackedLeafStorage<LeafStorage, T>)
        return view_traits::mutable_block(buffer_, offsets_[ordinal], make_extents<extents_type>(extents));
      else
        return view_traits::mutable_block(buffer_, offsets_[ordinal], make_extents<extents_type>(extents), ordinal);
    }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) const -> const_block_type
    {
      if constexpr (ImmediatePackedLeafStorage<LeafStorage, T>)
        return view_traits::const_block(buffer_, offsets_[ordinal], make_extents<extents_type>(extents));
      else
        return view_traits::const_block(buffer_, offsets_[ordinal], make_extents<extents_type>(extents), ordinal);
    }

    auto buffer() noexcept -> buffer_type& { return buffer_; }
    auto buffer() const noexcept -> buffer_type const& { return buffer_; }
    auto offsets() const noexcept -> std::span<std::size_t const> { return offsets_; }

    /// \brief Return the exclusive payload end of each stored block.
    /// \details The difference between one payload end and the next block
    ///          offset is zero-initialized alignment padding.
    auto block_ends() const noexcept -> std::span<std::size_t const> { return block_ends_; }

    /// \brief Return whether alignment inserted elements between block payloads.
    [[nodiscard]] bool has_padding() const noexcept
    {
      for (std::size_t ordinal = 0; ordinal < block_ends_.size(); ++ordinal)
        if (block_ends_[ordinal] != offsets_[ordinal + 1]) return true;
      return false;
    }

    /// \brief Allocate storage with the same packed block layout.
    /// \details Numerical values are unspecified until an operation writes them.
    /// \return Independent storage retaining the canonical keys and offsets.
    [[nodiscard]] auto allocate_like() const -> PackedBlockStorageData
    {
      return PackedBlockStorageData(Layout{.keys = keys_, .offsets = offsets_, .block_ends = block_ends_}, buffer_);
    }

  private:
    struct Layout
    {
        std::vector<key_type> keys;
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> block_ends;
    };

    static constexpr std::size_t alignment_elements = BlockAlignment / std::gcd(BlockAlignment, sizeof(T));

    static auto align_offset(std::size_t offset) -> std::size_t
    {
      std::size_t const remainder = offset % alignment_elements;
      if (remainder == 0) return offset;
      std::size_t const padding = alignment_elements - remainder;
      if (offset > std::numeric_limits<std::size_t>::max() - padding)
        throw std::length_error("packed BlockTensor alignment overflows size_t");
      return offset + padding;
    }

    static auto make_layout(std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs) -> Layout
    {
      Layout layout;
      layout.keys.reserve(specs.size());
      layout.offsets.reserve(specs.size() + 1);
      layout.block_ends.reserve(specs.size());

      std::size_t end_offset = 0;

      for (auto const& spec : specs)
      {
        std::size_t const block_offset = align_offset(end_offset);
        layout.keys.push_back(spec.key);
        layout.offsets.push_back(block_offset);
        std::size_t const block_size = checked_block_size(spec.extents);
        if (block_offset > std::numeric_limits<std::size_t>::max() - block_size)
        {
          throw std::length_error("packed BlockTensor size overflows size_t");
        }
        end_offset = block_offset + block_size;
        layout.block_ends.push_back(end_offset);
      }
      layout.offsets.push_back(end_offset);
      return layout;
    }

    explicit PackedBlockStorageData(Layout layout)
        : keys_(std::move(layout.keys)), offsets_(std::move(layout.offsets)), block_ends_(std::move(layout.block_ends)),
          buffer_(make_packed_storage<LeafStorage, T>(offsets_.back(), offsets_))
    {
      if constexpr (BlockAlignment > 1) initialize_packed_padding<LeafStorage, T>(buffer_, offsets_, block_ends_);
    }

    PackedBlockStorageData(Layout layout, buffer_type const& prototype)
        : keys_(std::move(layout.keys)), offsets_(std::move(layout.offsets)), block_ends_(std::move(layout.block_ends)),
          buffer_(make_packed_storage_like<LeafStorage, T>(prototype, offsets_.back(), offsets_))
    {
      if constexpr (BlockAlignment > 1) initialize_packed_padding<LeafStorage, T>(buffer_, offsets_, block_ends_);
    }

    std::vector<key_type> keys_;
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> block_ends_;
    buffer_type buffer_;
};

template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder, class LeafStorage>
class PackedDiagonalBlockStorageData {
  public:
    using key_type = BlockKey<KeyCoordinateCount>;
    using buffer_type = typename LeafStorage::template storage_t<T>;
    using extents_type = stdex::dextents<index_type, DenseBlockOrder>;
    using mapping_type = typename GeneratedLayout::template mapping<extents_type>;
    using accessor_type = diagonal_accessor<T, extents_type>;
    using const_accessor_type = diagonal_accessor<T const, extents_type>;
    using mutable_mdspan_type = stdex::mdspan<T, extents_type, GeneratedLayout, accessor_type>;
    using const_mdspan_type = stdex::mdspan<T const, extents_type, GeneratedLayout, const_accessor_type>;
    using component_extents_type = stdex::dextents<index_type, 1>;
    using component_mapping_type = stdex::layout_stride::mapping<component_extents_type>;
    using mutable_components_type = stdex::mdspan<T, component_extents_type, stdex::layout_stride>;
    using const_components_type = stdex::mdspan<T const, component_extents_type, stdex::layout_stride>;
    using mutable_block_type = MdspecTensorView<mutable_mdspan_type, const_mdspan_type, LeafStorage>;
    using const_block_type = MdspecTensorView<const_mdspan_type, const_mdspan_type, LeafStorage>;

    static_assert(DiagonalMdspecLike<mutable_mdspan_type>);
    static_assert(MutableDiagonalMdspecLike<mutable_mdspan_type>);
    static_assert(DiagonalMdspecLike<const_mdspan_type>);
    static_assert(!MutableDiagonalMdspecLike<const_mdspan_type>);

    static_assert(std::same_as<detail::tensor_accessor_factory_t<LeafStorage>, DefaultAccessorFactory>,
                  "packed diagonal block storage currently requires default-accessor leaf storage");
    static_assert(
        requires(buffer_type& buffer, buffer_type const& const_buffer) {
          { LeafStorage::make_handle(buffer) } -> std::same_as<T*>;
          { LeafStorage::make_handle(const_buffer) } -> std::same_as<T const*>;
        }, "packed diagonal block storage currently requires immediate contiguous host leaf storage");

    explicit PackedDiagonalBlockStorageData(std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
        : PackedDiagonalBlockStorageData(make_layout(specs))
    {}

    auto size() const noexcept -> std::size_t { return keys_.size(); }
    auto keys() const noexcept -> std::span<key_type const> { return keys_; }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) -> mutable_block_type
    {
      auto const block_extents = make_extents<extents_type>(extents);
      auto* mutable_data = LeafStorage::make_handle(buffer_);
      auto const* const_data = LeafStorage::make_handle(std::as_const(buffer_));
      if (offsets_[ordinal] != 0)
      {
        mutable_data += offsets_[ordinal];
        const_data += offsets_[ordinal];
      }
      auto const component_extents =
          component_extents_type{static_cast<index_type>(offsets_[ordinal + 1] - offsets_[ordinal])};
      auto const component_mapping =
          component_mapping_type{component_extents, std::array<index_type, 1>{index_type{1}}};
      auto mutable_components = mutable_components_type{mutable_data, component_mapping};
      auto read_components = const_components_type{const_data, component_mapping};
      return mutable_block_type{make_diagonal_mdspan(block_extents, mutable_components),
                                make_diagonal_mdspan(block_extents, read_components)};
    }

    auto block(std::size_t ordinal, std::array<std::size_t, DenseBlockOrder> const& extents) const -> const_block_type
    {
      auto const block_extents = make_extents<extents_type>(extents);
      auto const* data = LeafStorage::make_handle(buffer_);
      if (offsets_[ordinal] != 0) data += offsets_[ordinal];
      auto const component_extents =
          component_extents_type{static_cast<index_type>(offsets_[ordinal + 1] - offsets_[ordinal])};
      auto const component_mapping =
          component_mapping_type{component_extents, std::array<index_type, 1>{index_type{1}}};
      auto components = const_components_type{data, component_mapping};
      auto span = make_diagonal_mdspan(block_extents, components);
      return const_block_type{span, span};
    }

    /// \brief Return the compressed writable values for one stored block.
    auto diagonal_values(std::size_t ordinal) -> std::span<T>
    {
      auto* data = LeafStorage::make_handle(buffer_);
      if (offsets_[ordinal] != 0) data += offsets_[ordinal];
      return {data, offsets_[ordinal + 1] - offsets_[ordinal]};
    }

    /// \brief Return the compressed read-only values for one stored block.
    auto diagonal_values(std::size_t ordinal) const -> std::span<T const>
    {
      auto const* data = LeafStorage::make_handle(buffer_);
      if (offsets_[ordinal] != 0) data += offsets_[ordinal];
      return {data, offsets_[ordinal + 1] - offsets_[ordinal]};
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

    static auto make_layout(std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs) -> Layout
    {
      Layout layout;
      layout.keys.reserve(specs.size());
      layout.offsets.reserve(specs.size() + 1);
      layout.offsets.push_back(0);
      for (auto const& spec : specs)
      {
        layout.keys.push_back(spec.key);
        std::size_t const block_size = diagonal_extent(spec.extents);
        if (layout.offsets.back() > std::numeric_limits<std::size_t>::max() - block_size)
        {
          throw std::length_error("packed diagonal BlockTensor size overflows size_t");
        }
        layout.offsets.push_back(layout.offsets.back() + block_size);
      }
      return layout;
    }

    explicit PackedDiagonalBlockStorageData(Layout layout)
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
        std::vector<BlockSpec<KeyCoordinateCount, DenseBlockOrder>> const& specs)
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
template <class LeafStorage = HostStorage> struct SeparateSparseBlockStorage
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
template <class LeafStorage = HostStorage> struct ParallelSeparateSparseBlockStorage
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
/// \details A `BlockAlignment` greater than one pads block starts to that byte
///          alignment while preserving a single allocation.
/// \tparam LeafStorage Tensor storage policy used by the packed buffer.
/// \tparam BlockAlignment Required byte alignment of every block start.
template <class LeafStorage = HostStorage, std::size_t BlockAlignment = 1> struct PackedSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::PackedBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy, BlockAlignment>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Packed sparse storage whose disjoint dense blocks may be processed in a scheduler batch.
/// \details The packed buffer remains fixed while numerical operations execute.
///          Algorithms must place all writes to one logical output block in
///          the same batch item, so concurrently written block ranges are
///          disjoint. The batch call is synchronous. A `BlockAlignment` greater
///          than one pads block starts to that byte alignment.
/// \tparam LeafStorage Tensor storage policy used by the packed buffer.
/// \tparam BlockAlignment Required byte alignment of every block start.
template <class LeafStorage = HostStorage, std::size_t BlockAlignment = 1> struct ParallelPackedSparseBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SchedulerBatchBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::PackedBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy, BlockAlignment>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Complete storage packing every symmetry-legal dense block into one leaf buffer.
/// \details The owning BlockTensor determines the canonical legal-key set from
///          its boundaries. Fixed packed offsets make block placement stable
///          for the lifetime of the tensor. A `BlockAlignment` greater than one
///          pads block starts to that byte alignment.
/// \tparam LeafStorage Tensor storage policy used by the packed buffer.
/// \tparam BlockAlignment Required byte alignment of every block start.
template <class LeafStorage = HostStorage, std::size_t BlockAlignment = 1> struct PackedCompleteBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = true;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::PackedBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy, BlockAlignment>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Complete packed storage whose disjoint blocks may be processed in a scheduler batch.
/// \details The packed buffer and canonical legal-key layout remain fixed.
///          Algorithms must place all writes to one logical output block in
///          the same synchronous batch item. A `BlockAlignment` greater than
///          one pads block starts to that byte alignment.
/// \tparam LeafStorage Tensor storage policy used by the packed buffer.
/// \tparam BlockAlignment Required byte alignment of every block start.
template <class LeafStorage = HostStorage, std::size_t BlockAlignment = 1> struct ParallelPackedCompleteBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SchedulerBatchBlockExecution;
    static constexpr bool stores_all_legal_blocks = true;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::PackedBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy, BlockAlignment>;

    [[nodiscard]] static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return leaf_storage_policy::backend_selector();
    }
};

/// \brief Sparse BlockTensor storage packing each generalized diagonal block into one leaf buffer.
/// \details Every explicitly stored logical block contains only the entries
///          whose dense indices are all equal. Logical block presence remains
///          controlled by the owning BlockTensor's stored-key set.
/// \tparam LeafStorage Immediate contiguous host storage for compressed values.
template <class LeafStorage = HostStorage> struct PackedDiagonalBlockStorage
{
    using leaf_storage_policy = LeafStorage;
    using backend_selector_type = typename leaf_storage_policy::backend_selector_type;
    using block_execution_policy = SerialBlockExecution;
    static constexpr bool stores_all_legal_blocks = false;
    static constexpr bool stores_diagonal_blocks = true;
    static constexpr bool is_distributed = false;

    template <typename T, std::size_t KeyCoordinateCount, std::size_t DenseBlockOrder>
    using storage_t =
        detail::PackedDiagonalBlockStorageData<T, KeyCoordinateCount, DenseBlockOrder, leaf_storage_policy>;

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
template <class LeafStorage = HostStorage> struct AsyncSeparateSparseBlockStorage
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
static_assert(SparseBlockStorage<ParallelPackedSparseBlockStorage<>>);
static_assert(CompleteBlockStorage<PackedCompleteBlockStorage<>>);
static_assert(CompleteBlockStorage<ParallelPackedCompleteBlockStorage<>>);
static_assert(SparseBlockStorage<PackedDiagonalBlockStorage<>>);
static_assert(DiagonalBlockStorage<PackedDiagonalBlockStorage<>>);
static_assert(SparseBlockStorage<AsyncSeparateSparseBlockStorage<>>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<SeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<ParallelPackedSparseBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<ParallelPackedCompleteBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(BlockTensorStorageFor<PackedDiagonalBlockStorage<>, double, 2, 2>);
static_assert(BlockTensorStorageFor<PackedDiagonalBlockStorage<>, double, 0, 3>);
static_assert(LocalBlockStorageFor<PackedCompleteBlockStorage<>, double, 4, 0>);
static_assert(LocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<SeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<ParallelSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(ImmediateLocalBlockStorageFor<PackedSparseBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<ParallelPackedSparseBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<PackedCompleteBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<ParallelPackedCompleteBlockStorage<>, double, 4, 0>);
static_assert(ImmediateLocalBlockStorageFor<PackedDiagonalBlockStorage<>, double, 2, 2>);
static_assert(AsyncLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 2, 2>);
static_assert(AsyncLocalBlockStorageFor<AsyncSeparateSparseBlockStorage<>, double, 4, 0>);

} // namespace uni20
