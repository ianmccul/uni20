/**
 * \file block_tensor_mapped_view.hpp
 * \ingroup symmetry
 * \brief Defines shared zero-copy BlockTensor axis-mapping machinery.
 */

#pragma once

#include <uni20/mdspan/mdspec.hpp>
#include <uni20/symmetry/block_tensor.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::detail
{

template <std::size_t Offset, class Tuple, std::size_t... I>
auto tuple_slice_impl(Tuple const& tuple, std::index_sequence<I...>)
{
  return std::tuple{std::get<Offset + I>(tuple)...};
}

template <std::size_t Offset, std::size_t Count, class Tuple> auto tuple_slice(Tuple const& tuple)
{
  return tuple_slice_impl<Offset>(tuple, std::make_index_sequence<Count>{});
}

template <class Tuple> auto domain_from_tuple(Tuple&& tuple)
{
  return std::apply([](auto&&... spaces) { return Domain{std::forward<decltype(spaces)>(spaces)...}; },
                    std::forward<Tuple>(tuple));
}

template <class Tuple> auto codomain_from_tuple(Tuple&& tuple)
{
  return std::apply([](auto&&... spaces) { return Codomain{std::forward<decltype(spaces)>(spaces)...}; },
                    std::forward<Tuple>(tuple));
}

template <class Tensor, std::size_t Axis,
          bool DomainAxis = Axis<std::remove_cvref_t<Tensor>::domain_type::size()> struct TensorAxisSpace;

template <class Tensor, std::size_t Axis> struct TensorAxisSpace<Tensor, Axis, true>
{
    using tensor_type = std::remove_cvref_t<Tensor>;
    using type = typename tensor_type::domain_type::template space_type<Axis>;
};

template <class Tensor, std::size_t Axis> struct TensorAxisSpace<Tensor, Axis, false>
{
    using tensor_type = std::remove_cvref_t<Tensor>;
    static constexpr std::size_t codomain_axis = Axis - tensor_type::domain_type::size();
    using type = typename tensor_type::codomain_type::template space_type<codomain_axis>;
};

template <class Tensor, std::size_t Axis> using tensor_axis_space_t = typename TensorAxisSpace<Tensor, Axis>::type;

template <std::size_t Size>
consteval auto is_valid_axis_permutation(std::array<std::size_t, Size> const& permutation) -> bool
{
  std::array<bool, Size> seen{};
  for (std::size_t const axis : permutation)
  {
    if (axis >= Size || seen[axis]) return false;
    seen[axis] = true;
  }
  return true;
}

template <std::size_t Size> consteval auto make_identity_axis_permutation()
{
  std::array<std::size_t, Size> permutation{};
  for (std::size_t axis = 0; axis < Size; ++axis)
    permutation[axis] = axis;
  return permutation;
}

template <class Tensor, std::size_t... Axis> consteval auto key_axis_flags_impl(std::index_sequence<Axis...>)
{
  return std::array<bool, sizeof...(Axis)>{
      BlockTensorSpaceTraits<tensor_axis_space_t<Tensor, Axis>>::has_block_coordinate...};
}

template <class Tensor> consteval auto key_axis_flags()
{
  return key_axis_flags_impl<Tensor>(std::make_index_sequence<std::remove_cvref_t<Tensor>::order()>{});
}

template <class Tensor, std::size_t... Axis> consteval auto dense_axis_flags_impl(std::index_sequence<Axis...>)
{
  return std::array<bool, sizeof...(Axis)>{
      BlockTensorSpaceTraits<tensor_axis_space_t<Tensor, Axis>>::has_dense_axis...};
}

template <class Tensor> consteval auto dense_axis_flags()
{
  return dense_axis_flags_impl<Tensor>(std::make_index_sequence<std::remove_cvref_t<Tensor>::order()>{});
}

template <class Tensor, auto FactorPermutation, bool DenseAxes> consteval auto make_filtered_axis_permutation()
{
  using tensor_type = std::remove_cvref_t<Tensor>;
  constexpr std::size_t order = tensor_type::order();
  constexpr std::size_t filtered_count =
      DenseAxes ? tensor_type::dense_block_order() : tensor_type::key_coordinate_count();
  static_assert(std::tuple_size_v<decltype(FactorPermutation)> == order);
  static_assert(is_valid_axis_permutation(FactorPermutation));

  constexpr auto flags = [] {
    if constexpr (DenseAxes)
      return dense_axis_flags<tensor_type>();
    else
      return key_axis_flags<tensor_type>();
  }();

  std::array<std::size_t, order> source_filtered_axis{};
  std::size_t source_count = 0;
  for (std::size_t axis = 0; axis < order; ++axis)
  {
    if (flags[axis]) source_filtered_axis[axis] = source_count++;
  }

  std::array<std::size_t, filtered_count> permutation{};
  std::size_t output = 0;
  for (std::size_t factor = 0; factor < order; ++factor)
  {
    std::size_t const source_factor = FactorPermutation[factor];
    if (flags[source_factor]) permutation[output++] = source_filtered_axis[source_factor];
  }
  if (output != filtered_count) throw "invalid filtered BlockTensor axis permutation";
  return permutation;
}

template <std::size_t CoordinateCount>
auto permute_key(BlockKey<CoordinateCount> const& source, std::array<std::size_t, CoordinateCount> const& permutation)
    -> BlockKey<CoordinateCount>
{
  std::array<std::size_t, CoordinateCount> coordinates{};
  for (std::size_t axis = 0; axis < CoordinateCount; ++axis)
  {
    coordinates[axis] = source.coordinate(permutation[axis]);
  }
  return BlockKey<CoordinateCount>{coordinates};
}

template <class Block, bool Immediate = MdspanLike<std::remove_cvref_t<Block>>> struct StridedBlock;

template <class Block> struct StridedBlock<Block, true>
{
    using source_type = std::remove_cvref_t<Block>;
    using extents_type = stdex::dextents<typename source_type::index_type, source_type::rank()>;
    using type = stdex::mdspan<typename source_type::element_type, extents_type, stdex::layout_stride,
                               typename source_type::accessor_type>;
};

template <class Block> struct StridedBlock<Block, false>
{
    using source_type = std::remove_cvref_t<Block>;
    static_assert(MdspecLike<source_type>);
    using extents_type = stdex::dextents<typename source_type::index_type, source_type::rank()>;
    using type = mdspec<typename source_type::element_type, extents_type, stdex::layout_stride,
                        typename source_type::accessor_type, typename source_type::data_descriptor_type>;
};

template <class Block> using strided_block_t = typename StridedBlock<Block>::type;

template <MdspecLike Block, std::size_t DenseBlockOrder>
  requires(DenseBlockOrder == 0 || StridedMdspecLike<Block>)
auto permute_block_mdspec(Block block, std::array<std::size_t, DenseBlockOrder> const& permutation)
    -> strided_block_t<Block>
{
  using result_type = strided_block_t<Block>;
  using extents_type = typename result_type::extents_type;
  using mapping_type = typename result_type::mapping_type;
  using index_type = typename extents_type::index_type;

  std::array<std::size_t, DenseBlockOrder> extents{};
  std::array<index_type, DenseBlockOrder> strides{};
  if constexpr (DenseBlockOrder > 0)
  {
    for (std::size_t axis = 0; axis < DenseBlockOrder; ++axis)
    {
      extents[axis] = static_cast<std::size_t>(block.extent(permutation[axis]));
      strides[axis] = static_cast<index_type>(block.stride(permutation[axis]));
    }
  }
  auto const result_extents = make_extents<extents_type>(extents);
  if constexpr (MdspanLike<Block>)
  {
    return result_type(block.data_handle(), mapping_type(result_extents, strides), block.accessor());
  }
  else
  {
    return result_type(std::move(block.data_descriptor()), mapping_type(result_extents, strides), block.accessor());
  }
}

template <TensorView Block, std::size_t DenseBlockOrder>
  requires(DenseBlockOrder == tensor_mdspec_t<Block>::rank() && (DenseBlockOrder == 0 || StridedTensorView<Block>))
auto permute_block(Block block, std::array<std::size_t, DenseBlockOrder> const& permutation)
{
  auto mutable_mdspec = permute_block_mdspec(mdspec_of(block), permutation);
  auto const_mdspec = permute_block_mdspec(mdspec_of(std::as_const(block)), permutation);
  using mutable_mdspec_type = decltype(mutable_mdspec);
  using const_mdspec_type = decltype(const_mdspec);
  using storage_policy = tensor_storage_policy_t<Block>;
  return MdspecTensorView<mutable_mdspec_type, const_mdspec_type, storage_policy>{std::move(mutable_mdspec),
                                                                                  std::move(const_mdspec)};
}

/// \brief Shared implementation of a zero-copy BlockTensor metadata and axis view.
/// \details The transformed boundary and permutations must describe a bijection
///          of the source factors. Replacing or structurally modifying the
///          source invalidates this view and views transitively built from it.
template <class SourceTensor, class DomainType, class CodomainType, auto KeyPermutation, auto DensePermutation>
class BlockTensorMappedView {
  public:
    using source_type = SourceTensor;
    using source_reference_type = SourceTensor&;
    using source_tensor_type = std::remove_const_t<SourceTensor>;
    using element_type = typename source_tensor_type::element_type;
    using value_type = typename source_tensor_type::value_type;
    using storage_policy = typename source_tensor_type::storage_policy;
    using backend_selector_type = typename source_tensor_type::backend_selector_type;
    using key_type = typename source_tensor_type::key_type;
    using domain_type = DomainType;
    using codomain_type = CodomainType;
    using source_mutable_access_block_type =
        decltype(std::declval<SourceTensor&>().block(std::declval<key_type const&>()));
    using source_const_access_block_type =
        decltype(std::declval<SourceTensor const&>().block(std::declval<key_type const&>()));
    using mutable_block_type = decltype(permute_block(std::declval<source_mutable_access_block_type>(),
                                                      std::declval<decltype(DensePermutation) const&>()));
    using const_block_type = decltype(permute_block(std::declval<source_const_access_block_type>(),
                                                    std::declval<decltype(DensePermutation) const&>()));

    static constexpr std::size_t static_order = source_tensor_type::order();
    static constexpr std::size_t static_key_coordinate_count = source_tensor_type::key_coordinate_count();
    static constexpr std::size_t static_dense_block_order = source_tensor_type::dense_block_order();

  private:
    static constexpr bool source_blocks_writable =
        !std::is_const_v<typename std::remove_cvref_t<source_mutable_access_block_type>::element_type>;

    static_assert(IsDomain<domain_type>::value);
    static_assert(IsCodomain<codomain_type>::value);
    static_assert(domain_type::size() + codomain_type::size() == static_order);
    static_assert(BoundaryBlockShape<domain_type>::key_coordinate_count +
                      BoundaryBlockShape<codomain_type>::key_coordinate_count ==
                  static_key_coordinate_count);
    static_assert(BoundaryBlockShape<domain_type>::dense_block_order +
                      BoundaryBlockShape<codomain_type>::dense_block_order ==
                  static_dense_block_order);
    static_assert(std::tuple_size_v<decltype(KeyPermutation)> == static_key_coordinate_count);
    static_assert(std::tuple_size_v<decltype(DensePermutation)> == static_dense_block_order);
    static_assert(is_valid_axis_permutation(KeyPermutation));
    static_assert(is_valid_axis_permutation(DensePermutation));

  public:
    /// \brief Construct transformed metadata over an existing tensor value.
    /// \param source Tensor whose payload storage remains authoritative.
    /// \param domain Transformed ordered domain.
    /// \param codomain Transformed ordered codomain.
    explicit BlockTensorMappedView(SourceTensor& source, domain_type domain, codomain_type codomain)
        : source_(&source), domain_(std::move(domain)), codomain_(std::move(codomain))
    {
      this->build_key_index();
    }

    /// \brief Copy a view while retaining the same source binding.
    BlockTensorMappedView(BlockTensorMappedView const&) = default;

    /// \brief Move a view while retaining its source binding.
    BlockTensorMappedView(BlockTensorMappedView&&) = default;

    /// \brief Prevent replacing metadata observed by a dependent view.
    auto operator=(BlockTensorMappedView const&) -> BlockTensorMappedView& = delete;

    /// \brief Prevent replacing metadata observed by a dependent view.
    auto operator=(BlockTensorMappedView&&) -> BlockTensorMappedView& = delete;

    /// \brief Return the explicit symmetry context.
    auto symmetry() const -> Symmetry { return source_->symmetry(); }

    /// \brief Return the transformed ordered domain.
    auto domain() const -> domain_type const& { return domain_; }

    /// \brief Return the transformed ordered codomain.
    auto codomain() const -> codomain_type const& { return codomain_; }

    /// \brief Return the unchanged logical tensor order.
    static constexpr auto order() noexcept -> std::size_t { return static_order; }

    /// \brief Return the unchanged key-coordinate count.
    static constexpr auto key_coordinate_count() noexcept -> std::size_t { return static_key_coordinate_count; }

    /// \brief Return the unchanged dense-block order.
    static constexpr auto dense_block_order() noexcept -> std::size_t { return static_dense_block_order; }

    /// \brief Return the number of transformed stored-key records.
    auto stored_block_count() const noexcept -> std::size_t { return keys_.size(); }

    /// \brief Return the number of symmetry-legal blocks.
    auto legal_block_count() const -> std::size_t { return source_->legal_block_count(); }

    /// \brief Return whether every transformed legal block is stored.
    auto has_all_legal_blocks() const -> bool { return this->stored_block_count() == this->legal_block_count(); }

    /// \brief Return whether a transformed key satisfies charge conservation.
    auto is_legal(key_type const& key) const -> bool
    {
      return is_legal_block_key(this->symmetry(), domain_, codomain_, key);
    }

    /// \brief Return whether a transformed key has a source block binding.
    auto contains(key_type const& key) const -> bool { return this->ordinal(key).has_value(); }

    /// \brief Find a writable transformed block view.
    auto find_block(key_type const& key) -> std::optional<mutable_block_type>
      requires source_blocks_writable
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return permute_block(source_->block(source_keys_[*found]), DensePermutation);
    }

    /// \brief Find a read-only transformed block view.
    auto find_block(key_type const& key) const -> std::optional<const_block_type>
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return permute_block(std::as_const(*source_).block(source_keys_[*found]), DensePermutation);
    }

    /// \brief Return a writable transformed block view.
    /// \throws std::out_of_range If the transformed key is not stored.
    auto block(key_type const& key) -> mutable_block_type
      requires source_blocks_writable
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("mapped BlockTensor block key is not stored");
      return *found;
    }

    /// \brief Return a read-only transformed block view.
    /// \throws std::out_of_range If the transformed key is not stored.
    auto block(key_type const& key) const -> const_block_type
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("mapped BlockTensor block key is not stored");
      return *found;
    }

    /// \brief Return a writable transformed block by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Transformed block descriptor.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    auto block_by_ordinal(std::size_t ordinal) -> mutable_block_type
      requires source_blocks_writable
    {
      if (ordinal >= keys_.size()) throw std::out_of_range("mapped BlockTensor block ordinal is out of range");
      return permute_block(source_->block(source_keys_[ordinal]), DensePermutation);
    }

    /// \brief Return a read-only transformed block by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Transformed block descriptor.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    auto block_by_ordinal(std::size_t ordinal) const -> const_block_type
    {
      if (ordinal >= keys_.size()) throw std::out_of_range("mapped BlockTensor block ordinal is out of range");
      return permute_block(std::as_const(*source_).block(source_keys_[ordinal]), DensePermutation);
    }

    /// \brief Return transformed keys in canonical lexicographic order.
    auto stored_keys() const noexcept -> std::span<key_type const> { return keys_; }

    /// \brief Return the source tensor retaining numerical storage ownership.
    auto source() noexcept -> SourceTensor& { return *source_; }

    /// \brief Return the source tensor retaining numerical storage ownership.
    auto source() const noexcept -> SourceTensor const& { return *source_; }

    /// \brief Return the source storage's backend selector.
    static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return source_tensor_type::backend_selector();
    }

  private:
    void build_key_index()
    {
      std::vector<std::pair<key_type, key_type>> bindings;
      bindings.reserve(source_->stored_block_count());
      for (key_type const& source_key : source_->stored_keys())
      {
        bindings.emplace_back(permute_key(source_key, KeyPermutation), source_key);
      }
      std::ranges::sort(bindings, {}, &std::pair<key_type, key_type>::first);

      keys_.reserve(bindings.size());
      source_keys_.reserve(bindings.size());
      for (auto const& [key, source_key] : bindings)
      {
        keys_.push_back(key);
        source_keys_.push_back(source_key);
      }
    }

    auto ordinal(key_type const& key) const -> std::optional<std::size_t>
    {
      auto const found = std::ranges::lower_bound(keys_, key);
      if (found == keys_.end() || *found != key) return std::nullopt;
      return static_cast<std::size_t>(found - keys_.begin());
    }

    SourceTensor* source_;
    domain_type domain_;
    codomain_type codomain_;
    std::vector<key_type> keys_;
    std::vector<key_type> source_keys_;
};

} // namespace uni20::detail
