/**
 * \file block_tensor_repartition.hpp
 * \ingroup symmetry
 * \brief Defines zero-copy bosonic edge repartition views.
 */

#pragma once

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

namespace uni20
{

/// \brief Morphism side from which `repartition` bends one edge leg.
enum class MorphismSide
{
  Domain,
  Codomain
};

/// \brief Planar boundary end at which `repartition` bends one leg.
enum class BoundaryEnd
{
  Left,
  Right
};

namespace detail
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

template <MorphismSide Side, BoundaryEnd End, class Tensor> auto make_repartitioned_domain(Tensor const& tensor)
{
  constexpr std::size_t domain_size = Tensor::domain_type::size();
  constexpr std::size_t codomain_size = Tensor::codomain_type::size();
  auto const domain_spaces = tensor.domain().spaces();

  if constexpr (Side == MorphismSide::Domain)
  {
    if constexpr (End == BoundaryEnd::Left)
    {
      return domain_from_tuple(tuple_slice<1, domain_size - 1>(domain_spaces));
    }
    else
    {
      return domain_from_tuple(tuple_slice<0, domain_size - 1>(domain_spaces));
    }
  }
  else
  {
    auto const codomain_spaces = tensor.codomain().spaces();
    constexpr std::size_t moved_index = End == BoundaryEnd::Left ? 0 : codomain_size - 1;
    auto moved = dual(std::get<moved_index>(codomain_spaces));
    if constexpr (End == BoundaryEnd::Left)
    {
      return domain_from_tuple(std::tuple_cat(std::tuple{std::move(moved)}, domain_spaces));
    }
    else
    {
      return domain_from_tuple(std::tuple_cat(domain_spaces, std::tuple{std::move(moved)}));
    }
  }
}

template <MorphismSide Side, BoundaryEnd End, class Tensor> auto make_repartitioned_codomain(Tensor const& tensor)
{
  constexpr std::size_t domain_size = Tensor::domain_type::size();
  constexpr std::size_t codomain_size = Tensor::codomain_type::size();
  auto const codomain_spaces = tensor.codomain().spaces();

  if constexpr (Side == MorphismSide::Codomain)
  {
    if constexpr (End == BoundaryEnd::Left)
    {
      return codomain_from_tuple(tuple_slice<1, codomain_size - 1>(codomain_spaces));
    }
    else
    {
      return codomain_from_tuple(tuple_slice<0, codomain_size - 1>(codomain_spaces));
    }
  }
  else
  {
    auto const domain_spaces = tensor.domain().spaces();
    constexpr std::size_t moved_index = End == BoundaryEnd::Left ? 0 : domain_size - 1;
    auto moved = dual(std::get<moved_index>(domain_spaces));
    if constexpr (End == BoundaryEnd::Left)
    {
      return codomain_from_tuple(std::tuple_cat(std::tuple{std::move(moved)}, codomain_spaces));
    }
    else
    {
      return codomain_from_tuple(std::tuple_cat(codomain_spaces, std::tuple{std::move(moved)}));
    }
  }
}

template <MorphismSide Side, BoundaryEnd End, class DomainType, class CodomainType> struct MovedSpace;

template <BoundaryEnd End, class DomainType, class CodomainType>
struct MovedSpace<MorphismSide::Domain, End, DomainType, CodomainType>
{
    static constexpr std::size_t index = End == BoundaryEnd::Left ? 0 : DomainType::size() - 1;
    using type = typename DomainType::template space_type<index>;
};

template <BoundaryEnd End, class DomainType, class CodomainType>
struct MovedSpace<MorphismSide::Codomain, End, DomainType, CodomainType>
{
    static constexpr std::size_t index = End == BoundaryEnd::Left ? 0 : CodomainType::size() - 1;
    using type = typename CodomainType::template space_type<index>;
};

template <MorphismSide Side, BoundaryEnd End, std::size_t CoordinateCount>
consteval auto make_repartition_permutation(std::size_t domain_count, std::size_t codomain_count,
                                            bool moved_has_coordinate) -> std::array<std::size_t, CoordinateCount>
{
  std::array<std::size_t, CoordinateCount> permutation{};
  std::size_t output = 0;
  std::size_t const moved_count = moved_has_coordinate ? 1 : 0;
  auto append = [&](std::size_t first, std::size_t last) {
    for (std::size_t axis = first; axis < last; ++axis)
      permutation[output++] = axis;
  };

  if constexpr (Side == MorphismSide::Domain && End == BoundaryEnd::Left)
  {
    append(moved_count, domain_count);
    if (moved_has_coordinate) permutation[output++] = 0;
    append(domain_count, domain_count + codomain_count);
  }
  else if constexpr (Side == MorphismSide::Domain && End == BoundaryEnd::Right)
  {
    append(0, domain_count - moved_count);
    append(domain_count, domain_count + codomain_count);
    if (moved_has_coordinate) permutation[output++] = domain_count - 1;
  }
  else if constexpr (Side == MorphismSide::Codomain && End == BoundaryEnd::Left)
  {
    if (moved_has_coordinate) permutation[output++] = domain_count;
    append(0, domain_count);
    append(domain_count + moved_count, domain_count + codomain_count);
  }
  else
  {
    append(0, domain_count);
    if (moved_has_coordinate) permutation[output++] = domain_count + codomain_count - 1;
    append(domain_count, domain_count + codomain_count - moved_count);
  }

  if (output != CoordinateCount) throw "invalid repartition coordinate permutation";
  return permutation;
}

template <std::size_t CoordinateCount>
auto permute_key(BlockKey<CoordinateCount> const& source,
                 std::array<std::size_t, CoordinateCount> const& permutation) -> BlockKey<CoordinateCount>
{
  std::array<std::size_t, CoordinateCount> coordinates{};
  for (std::size_t axis = 0; axis < CoordinateCount; ++axis)
  {
    coordinates[axis] = source.coordinate(permutation[axis]);
  }
  return BlockKey<CoordinateCount>{coordinates};
}

template <class Block> struct StridedBlock
{
    using source_type = std::remove_cvref_t<Block>;
    using extents_type = stdex::dextents<typename source_type::index_type, source_type::rank()>;
    using type = stdex::mdspan<typename source_type::element_type, extents_type, stdex::layout_stride,
                               typename source_type::accessor_type>;
};

template <class Block> using strided_block_t = typename StridedBlock<Block>::type;

template <class Block, std::size_t DenseBlockOrder>
auto permute_block(Block block, std::array<std::size_t, DenseBlockOrder> const& permutation) -> strided_block_t<Block>
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
  return result_type(block.data_handle(), mapping_type(result_extents, strides), block.accessor());
}

template <class Tensor, MorphismSide Side>
concept RepartitionSourceHasLeg =
    (Side == MorphismSide::Domain && std::remove_cvref_t<Tensor>::domain_type::size() > 0) ||
    (Side == MorphismSide::Codomain && std::remove_cvref_t<Tensor>::codomain_type::size() > 0);

} // namespace detail

/// \brief Zero-copy view of one bosonic edge leg bent across a morphism boundary.
/// \details The view owns transformed boundary and logical-key metadata while
///          retaining a reference to the source tensor's numerical storage.
///          Dense axes are exposed through strided mdspans in transformed order.
///          Replacing or structurally modifying the source invalidates the view;
///          writes to existing payload elements do not.
/// \tparam SourceTensor Source BlockTensor-like value, optionally const.
/// \tparam Side Morphism side from which the edge factor moves.
/// \tparam End Planar boundary end at which the factor bends.
template <class SourceTensor, MorphismSide Side, BoundaryEnd End>
  requires detail::RepartitionSourceHasLeg<SourceTensor, Side>
class BlockTensorRepartitionView {
  public:
    using source_type = SourceTensor;
    using source_reference_type = SourceTensor&;
    using source_tensor_type = std::remove_const_t<SourceTensor>;
    using element_type = typename source_tensor_type::element_type;
    using value_type = typename source_tensor_type::value_type;
    using storage_policy = typename source_tensor_type::storage_policy;
    using backend_selector_type = typename source_tensor_type::backend_selector_type;
    using key_type = typename source_tensor_type::key_type;
    using domain_type =
        decltype(detail::make_repartitioned_domain<Side, End>(std::declval<source_tensor_type const&>()));
    using codomain_type =
        decltype(detail::make_repartitioned_codomain<Side, End>(std::declval<source_tensor_type const&>()));
    using source_mutable_access_block_type =
        decltype(std::declval<SourceTensor&>().block(std::declval<key_type const&>()));
    using source_const_access_block_type =
        decltype(std::declval<SourceTensor const&>().block(std::declval<key_type const&>()));
    using mutable_block_type = detail::strided_block_t<source_mutable_access_block_type>;
    using const_block_type = detail::strided_block_t<source_const_access_block_type>;

    static constexpr std::size_t static_order = source_tensor_type::order();
    static constexpr std::size_t static_key_coordinate_count = source_tensor_type::key_coordinate_count();
    static constexpr std::size_t static_dense_block_order = source_tensor_type::dense_block_order();

  private:
    using source_domain_type = typename source_tensor_type::domain_type;
    using source_codomain_type = typename source_tensor_type::codomain_type;
    using moved_space_type = typename detail::MovedSpace<Side, End, source_domain_type, source_codomain_type>::type;

    static constexpr std::size_t source_domain_key_count =
        detail::BoundaryBlockShape<source_domain_type>::key_coordinate_count;
    static constexpr std::size_t source_codomain_key_count =
        detail::BoundaryBlockShape<source_codomain_type>::key_coordinate_count;
    static constexpr std::size_t source_domain_dense_order =
        detail::BoundaryBlockShape<source_domain_type>::dense_block_order;
    static constexpr std::size_t source_codomain_dense_order =
        detail::BoundaryBlockShape<source_codomain_type>::dense_block_order;
    static constexpr bool moved_has_key_coordinate = BlockTensorSpaceTraits<moved_space_type>::has_block_coordinate;
    static constexpr bool moved_has_dense_axis = BlockTensorSpaceTraits<moved_space_type>::has_dense_axis;
    static constexpr bool source_blocks_writable =
        !std::is_const_v<typename std::remove_cvref_t<source_mutable_access_block_type>::element_type>;
    static constexpr auto key_permutation =
        detail::make_repartition_permutation<Side, End, static_key_coordinate_count>(
            source_domain_key_count, source_codomain_key_count, moved_has_key_coordinate);
    static constexpr auto dense_permutation = detail::make_repartition_permutation<Side, End, static_dense_block_order>(
        source_domain_dense_order, source_codomain_dense_order, moved_has_dense_axis);

    static_assert(detail::BoundaryBlockShape<domain_type>::key_coordinate_count +
                      detail::BoundaryBlockShape<codomain_type>::key_coordinate_count ==
                  static_key_coordinate_count);
    static_assert(detail::BoundaryBlockShape<domain_type>::dense_block_order +
                      detail::BoundaryBlockShape<codomain_type>::dense_block_order ==
                  static_dense_block_order);

  public:
    /// \brief Construct transformed metadata over an existing tensor value.
    /// \param source Tensor whose payload storage remains authoritative.
    explicit BlockTensorRepartitionView(SourceTensor& source)
        : source_(&source), domain_(detail::make_repartitioned_domain<Side, End>(source)),
          codomain_(detail::make_repartitioned_codomain<Side, End>(source))
    {
      this->build_key_index();
    }

    /// \brief Copy a view while retaining the same source binding.
    BlockTensorRepartitionView(BlockTensorRepartitionView const&) = default;

    /// \brief Move a view while retaining its source binding.
    BlockTensorRepartitionView(BlockTensorRepartitionView&&) = default;

    /// \brief Prevent replacing metadata observed by a dependent view.
    auto operator=(BlockTensorRepartitionView const&) -> BlockTensorRepartitionView& = delete;

    /// \brief Prevent replacing metadata observed by a dependent view.
    auto operator=(BlockTensorRepartitionView&&) -> BlockTensorRepartitionView& = delete;

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
      return detail::is_legal_block_key(this->symmetry(), domain_, codomain_, key);
    }

    /// \brief Return whether a transformed key has a source block binding.
    auto contains(key_type const& key) const -> bool { return this->ordinal(key).has_value(); }

    /// \brief Find a writable transformed block view.
    auto find_block(key_type const& key) -> std::optional<mutable_block_type>
      requires source_blocks_writable
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return detail::permute_block(source_->block(source_keys_[*found]), dense_permutation);
    }

    /// \brief Find a read-only transformed block view.
    auto find_block(key_type const& key) const -> std::optional<const_block_type>
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return detail::permute_block(std::as_const(*source_).block(source_keys_[*found]), dense_permutation);
    }

    /// \brief Return a writable transformed block view.
    /// \throws std::out_of_range If the transformed key is not stored.
    auto block(key_type const& key) -> mutable_block_type
      requires source_blocks_writable
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("repartitioned BlockTensor block key is not stored");
      return *found;
    }

    /// \brief Return a read-only transformed block view.
    /// \throws std::out_of_range If the transformed key is not stored.
    auto block(key_type const& key) const -> const_block_type
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("repartitioned BlockTensor block key is not stored");
      return *found;
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
        bindings.emplace_back(detail::permute_key(source_key, key_permutation), source_key);
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

/// \brief Bend one planar edge leg across a BlockTensor morphism boundary.
/// \details The returned lvalue view owns only transformed metadata; numerical
///          payload remains in \p tensor. Rvalues are rejected to prevent a
///          dangling storage reference. Payload element writes remain valid,
///          but replacing or structurally modifying \p tensor invalidates the
///          view and every view transitively built from it.
/// \tparam Side Source morphism side.
/// \tparam End Planar edge to bend.
/// \tparam Tensor BlockTensor-like source value.
/// \param tensor Source tensor whose lifetime must cover the returned view.
/// \return Zero-copy transformed BlockTensor view.
template <MorphismSide Side, BoundaryEnd End, class Tensor>
  requires detail::RepartitionSourceHasLeg<Tensor, Side>
auto repartition(Tensor& tensor) -> BlockTensorRepartitionView<Tensor, Side, End>
{
  return BlockTensorRepartitionView<Tensor, Side, End>(tensor);
}

template <MorphismSide Side, BoundaryEnd End, class Tensor>
  requires(!std::is_lvalue_reference_v<Tensor &&>)
auto repartition(Tensor&&) = delete;

} // namespace uni20
