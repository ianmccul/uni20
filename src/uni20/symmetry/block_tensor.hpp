/**
 * \file block_tensor.hpp
 * \ingroup symmetry
 * \brief Defines the first bosonic abelian BlockTensor container.
 */

#pragma once

#include <uni20/symmetry/block_space.hpp>
#include <uni20/symmetry/block_tensor_space_traits.hpp>
#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/symmetry/dual_space.hpp>
#include <uni20/symmetry/local_space.hpp>
#include <uni20/symmetry/morphism_boundary.hpp>
#include <uni20/symmetry/qnum_space.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{
namespace detail
{

template <class T> struct IsDomain : std::false_type
{};

template <class... Spaces> struct IsDomain<Domain<Spaces...>> : std::true_type
{};

template <class T> struct IsCodomain : std::false_type
{};

template <class... Spaces> struct IsCodomain<Codomain<Spaces...>> : std::true_type
{};

template <class T>
concept PrototypeBlockFactor =
    std::same_as<primal_space_t<T>, BlockSpace> || std::same_as<primal_space_t<T>, LocalSpace> ||
    std::same_as<primal_space_t<T>, QNumSpace>;

template <class Boundary> struct PrototypeBoundary;

template <template <class...> class Boundary, class... Spaces>
struct PrototypeBoundary<Boundary<Spaces...>> : std::bool_constant<(PrototypeBlockFactor<Spaces> && ...)>
{};

template <class Boundary> struct BoundaryBlockShape;

template <template <class...> class Boundary, class... Spaces> struct BoundaryBlockShape<Boundary<Spaces...>>
{
    static constexpr std::size_t key_coordinate_count =
        (std::size_t{0} + ... + (BlockTensorSpaceTraits<std::remove_cvref_t<Spaces>>::has_block_coordinate ? 1U : 0U));
    static constexpr std::size_t dense_block_order =
        (std::size_t{0} + ... + (BlockTensorSpaceTraits<std::remove_cvref_t<Spaces>>::has_dense_axis ? 1U : 0U));
};

inline auto factor_size(BlockSpace const& space) -> std::size_t { return space.size(); }
inline auto factor_size(LocalSpace const& space) -> std::size_t { return space.size(); }

template <Space SpaceType> auto factor_size(Dual<SpaceType> const& space) -> std::size_t
{
  return factor_size(space.base());
}

inline auto factor_qnum(BlockSpace const& space, std::size_t coordinate) -> QNum const& { return space[coordinate].q; }

inline auto factor_qnum(LocalSpace const& space, std::size_t coordinate) -> QNum const& { return space[coordinate]; }

inline auto factor_qnum(QNumSpace const& space, std::size_t) -> QNum const& { return space.qnum(); }

template <Space SpaceType> auto factor_qnum(Dual<SpaceType> const& space, std::size_t coordinate) -> QNum
{
  return dual(factor_qnum(space.base(), coordinate));
}

inline auto factor_extent(BlockSpace const& space, std::size_t coordinate) -> std::size_t
{
  return space[coordinate].dim;
}

template <Space SpaceType> auto factor_extent(Dual<SpaceType> const& space, std::size_t coordinate) -> std::size_t
{
  return factor_extent(space.base(), coordinate);
}

template <class Boundary, class Function> void for_each_boundary_space(Boundary const& boundary, Function&& function)
{
  std::apply([&](auto const&... spaces) { (function(spaces), ...); }, boundary.spaces());
}

template <std::size_t KeyCoordinateCount, class DomainType, class CodomainType>
auto boundary_factor_sizes(DomainType const& domain,
                           CodomainType const& codomain) -> std::array<std::size_t, KeyCoordinateCount>
{
  std::array<std::size_t, KeyCoordinateCount> sizes{};
  std::size_t key_axis = 0;
  auto append_size = [&](auto const& space) {
    using space_type = std::remove_cvref_t<decltype(space)>;
    if constexpr (BlockTensorSpaceTraits<space_type>::has_block_coordinate)
    {
      sizes[key_axis++] = factor_size(space);
    }
  };
  for_each_boundary_space(domain, append_size);
  for_each_boundary_space(codomain, append_size);
  return sizes;
}

template <std::size_t KeyCoordinateCount, class DomainType, class CodomainType>
auto is_legal_block_key(Symmetry const& symmetry, DomainType const& domain, CodomainType const& codomain,
                        BlockKey<KeyCoordinateCount> const& key) -> bool
{
  auto const sizes = boundary_factor_sizes<KeyCoordinateCount>(domain, codomain);
  for (std::size_t axis = 0; axis < KeyCoordinateCount; ++axis)
  {
    if (key.coordinate(axis) >= sizes[axis]) return false;
  }

  QNum domain_charge = QNum::identity(symmetry);
  QNum codomain_charge = QNum::identity(symmetry);
  std::size_t key_axis = 0;
  auto add_charge = [&](auto const& space, QNum& charge) {
    using space_type = std::remove_cvref_t<decltype(space)>;
    std::size_t coordinate = 0;
    if constexpr (BlockTensorSpaceTraits<space_type>::has_block_coordinate)
    {
      coordinate = key.coordinate(key_axis++);
    }
    charge = charge + factor_qnum(space, coordinate);
  };
  for_each_boundary_space(domain, [&](auto const& space) { add_charge(space, domain_charge); });
  for_each_boundary_space(codomain, [&](auto const& space) { add_charge(space, codomain_charge); });
  return domain_charge == codomain_charge;
}

} // namespace detail

/// \brief Fixed-order tensor whose blocks obey a bosonic abelian selection rule.
/// \details This first slice supports orders zero through four and boundaries
///          containing `BlockSpace`, `LocalSpace`, `QNumSpace`, and their
///          explicit `Dual<S>` adaptors.
/// \tparam T Numerical block element type.
/// \tparam DomainType Ordered domain value type.
/// \tparam CodomainType Ordered codomain value type.
/// \tparam Storage Block storage policy; explicit-key construction requires a
///                 sparse policy.
template <typename T, class DomainType, class CodomainType, BlockTensorStorage Storage>
  requires(detail::IsDomain<DomainType>::value && detail::IsCodomain<CodomainType>::value &&
           detail::PrototypeBoundary<DomainType>::value && detail::PrototypeBoundary<CodomainType>::value)
class BlockTensor {
  public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using domain_type = DomainType;
    using codomain_type = CodomainType;
    using storage_policy = Storage;
    static constexpr std::size_t static_order = domain_type::size() + codomain_type::size();
    static constexpr std::size_t static_key_coordinate_count =
        detail::BoundaryBlockShape<domain_type>::key_coordinate_count +
        detail::BoundaryBlockShape<codomain_type>::key_coordinate_count;
    static constexpr std::size_t static_dense_block_order =
        detail::BoundaryBlockShape<domain_type>::dense_block_order +
        detail::BoundaryBlockShape<codomain_type>::dense_block_order;
    static_assert(static_order <= 4, "the first BlockTensor slice supports orders 0 through 4");
    static_assert(
        BlockTensorStorageFor<storage_policy, element_type, static_key_coordinate_count, static_dense_block_order>,
        "BlockTensor storage does not provide the required block bindings");

    using key_type = BlockKey<static_key_coordinate_count>;
    using storage_type = typename storage_policy::template storage_t<element_type, static_key_coordinate_count,
                                                                     static_dense_block_order>;
    using mutable_block_type = typename storage_type::mutable_block_type;
    using const_block_type = typename storage_type::const_block_type;
    using backend_selector_type = typename storage_policy::backend_selector_type;

    /// \brief Construct a sparse tensor from an explicit stored-key list.
    /// \param symmetry Explicit tensor symmetry context.
    /// \param domain Ordered domain spaces.
    /// \param codomain Ordered codomain spaces.
    /// \param stored_keys Legal keys to allocate; order is canonicalized.
    BlockTensor(Symmetry symmetry, domain_type domain, codomain_type codomain, std::vector<key_type> stored_keys)
      requires SparseBlockStorage<storage_policy>
        : symmetry_(symmetry), domain_(std::move(domain)), codomain_(std::move(codomain)),
          storage_(this->make_storage(std::move(stored_keys)))
    {}

    /// \brief Construct a sparse tensor from an initializer list of stored keys.
    /// \param symmetry Explicit tensor symmetry context.
    /// \param domain Ordered domain spaces.
    /// \param codomain Ordered codomain spaces.
    /// \param stored_keys Legal keys to allocate; order is canonicalized.
    BlockTensor(Symmetry symmetry, domain_type domain, codomain_type codomain,
                std::initializer_list<key_type> stored_keys)
      requires SparseBlockStorage<storage_policy>
        : BlockTensor(symmetry, std::move(domain), std::move(codomain), std::vector<key_type>(stored_keys))
    {}

    /// \brief Return the explicit symmetry context.
    auto symmetry() const -> Symmetry { return symmetry_; }

    /// \brief Return the ordered domain value.
    auto domain() const -> domain_type const& { return domain_; }

    /// \brief Return the ordered codomain value.
    auto codomain() const -> codomain_type const& { return codomain_; }

    /// \brief Return the compile-time tensor order.
    static constexpr auto order() noexcept -> std::size_t { return static_order; }

    /// \brief Return the number of coordinates stored in each block key.
    static constexpr auto key_coordinate_count() noexcept -> std::size_t { return static_key_coordinate_count; }

    /// \brief Return the dense mdspan order of each numerical block.
    static constexpr auto dense_block_order() noexcept -> std::size_t { return static_dense_block_order; }

    /// \brief Return the number of stored blocks.
    auto stored_block_count() const noexcept -> std::size_t { return storage_.size(); }

    /// \brief Return the number of symmetry-legal blocks.
    auto legal_block_count() const -> std::size_t
    {
      std::size_t count = 0;
      this->for_each_possible_key([&](key_type const& key) {
        if (this->is_legal(key)) ++count;
      });
      return count;
    }

    /// \brief Return whether this value currently stores every legal block.
    /// \note This observation does not change the storage policy's compile-time
    ///       completeness guarantee.
    auto has_all_legal_blocks() const -> bool { return this->stored_block_count() == this->legal_block_count(); }

    /// \brief Return whether a key is in range and satisfies charge conservation.
    auto is_legal(key_type const& key) const -> bool
    {
      return detail::is_legal_block_key(symmetry_, domain_, codomain_, key);
    }

    /// \brief Return whether a key has an allocated numerical block.
    auto contains(key_type const& key) const -> bool { return this->ordinal(key).has_value(); }

    /// \brief Find a writable block without changing tensor structure.
    auto find_block(key_type const& key) -> std::optional<mutable_block_type>
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return this->block_by_ordinal(*found);
    }

    /// \brief Find a read-only block without changing tensor structure.
    auto find_block(key_type const& key) const -> std::optional<const_block_type>
    {
      auto const found = this->ordinal(key);
      if (!found) return std::nullopt;
      return this->block_by_ordinal(*found);
    }

    /// \brief Return a writable stored block.
    /// \throws std::out_of_range If the key is not stored.
    auto block(key_type const& key) -> mutable_block_type
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("BlockTensor block key is not stored");
      return *found;
    }

    /// \brief Return a read-only stored block.
    /// \throws std::out_of_range If the key is not stored.
    auto block(key_type const& key) const -> const_block_type
    {
      auto const found = this->find_block(key);
      if (!found) throw std::out_of_range("BlockTensor block key is not stored");
      return *found;
    }

    /// \brief Return a writable block descriptor by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Immediate mdspan or deferred mdspec selected by storage.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    auto block_by_ordinal(std::size_t ordinal) -> mutable_block_type
    {
      if (ordinal >= storage_.size()) throw std::out_of_range("BlockTensor block ordinal is out of range");
      auto const& key = storage_.keys()[ordinal];
      return storage_.block(ordinal, this->block_extents(key));
    }

    /// \brief Return a read-only block descriptor by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Immediate mdspan or deferred mdspec selected by storage.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    auto block_by_ordinal(std::size_t ordinal) const -> const_block_type
    {
      if (ordinal >= storage_.size()) throw std::out_of_range("BlockTensor block ordinal is out of range");
      auto const& key = storage_.keys()[ordinal];
      return storage_.block(ordinal, this->block_extents(key));
    }

    /// \brief Return the independently scheduled value for one stored key.
    /// \details The returned `Async<Tensor>` owns the block's dependency
    ///          timeline. Submitted work must retain a read or write buffer.
    /// \param key Stored logical block key.
    /// \return Mutable async dense-block value.
    /// \throws std::out_of_range If the key is not stored.
    decltype(auto) async_block(key_type const& key)
      requires AsyncLocalBlockStorageFor<storage_policy, element_type, static_key_coordinate_count,
                                         static_dense_block_order>
    {
      auto const found = this->ordinal(key);
      if (!found) throw std::out_of_range("BlockTensor block key is not stored");
      return storage_.async_block(*found);
    }

    /// \brief Return the independently scheduled value for one stored key.
    /// \param key Stored logical block key.
    /// \return Read-only async dense-block value.
    /// \throws std::out_of_range If the key is not stored.
    decltype(auto) async_block(key_type const& key) const
      requires AsyncLocalBlockStorageFor<storage_policy, element_type, static_key_coordinate_count,
                                         static_dense_block_order>
    {
      auto const found = this->ordinal(key);
      if (!found) throw std::out_of_range("BlockTensor block key is not stored");
      return storage_.async_block(*found);
    }

    /// \brief Return one independently scheduled value by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Mutable async dense-block value.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    decltype(auto) async_block_by_ordinal(std::size_t ordinal)
      requires AsyncLocalBlockStorageFor<storage_policy, element_type, static_key_coordinate_count,
                                         static_dense_block_order>
    {
      if (ordinal >= storage_.size()) throw std::out_of_range("BlockTensor block ordinal is out of range");
      return storage_.async_block(ordinal);
    }

    /// \brief Return one independently scheduled value by stable canonical ordinal.
    /// \param ordinal Position in `stored_keys()`.
    /// \return Read-only async dense-block value.
    /// \throws std::out_of_range If \p ordinal is not a stored-block position.
    decltype(auto) async_block_by_ordinal(std::size_t ordinal) const
      requires AsyncLocalBlockStorageFor<storage_policy, element_type, static_key_coordinate_count,
                                         static_dense_block_order>
    {
      if (ordinal >= storage_.size()) throw std::out_of_range("BlockTensor block ordinal is out of range");
      return storage_.async_block(ordinal);
    }

    /// \brief Return stored keys in canonical lexicographic order.
    auto stored_keys() const noexcept -> std::span<key_type const> { return storage_.keys(); }

    /// \brief Return the default dense-leaf backend selector for this storage.
    static constexpr auto backend_selector() noexcept -> backend_selector_type
    {
      return storage_policy::backend_selector();
    }

    /// \brief Return the concrete storage value.
    auto storage() noexcept -> storage_type& { return storage_; }

    /// \brief Return the concrete storage value.
    auto storage() const noexcept -> storage_type const& { return storage_; }

  private:
    auto make_storage(std::vector<key_type> stored_keys) -> storage_type
    {
      this->validate_boundaries();
      std::sort(stored_keys.begin(), stored_keys.end());
      if (std::adjacent_find(stored_keys.begin(), stored_keys.end()) != stored_keys.end())
      {
        throw std::invalid_argument("BlockTensor stored keys must be unique");
      }

      std::vector<detail::SparseBlockSpec<static_key_coordinate_count, static_dense_block_order>> specs;
      specs.reserve(stored_keys.size());
      for (key_type const& key : stored_keys)
      {
        if (!this->is_legal(key)) throw std::invalid_argument("BlockTensor stored key is not symmetry-legal");
        auto const extents = this->block_extents(key);
        static_cast<void>(detail::checked_block_size(extents));
        specs.push_back({key, extents});
      }
      return storage_type(specs);
    }

    void validate_boundaries() const
    {
      if (!symmetry_.valid()) throw std::invalid_argument("BlockTensor requires a valid symmetry");
      auto validate = [&](auto const& space) {
        if (space.symmetry() != symmetry_)
        {
          throw std::invalid_argument("BlockTensor boundary space has the wrong symmetry");
        }
      };
      detail::for_each_boundary_space(domain_, validate);
      detail::for_each_boundary_space(codomain_, validate);
    }

    auto factor_sizes() const -> std::array<std::size_t, static_key_coordinate_count>
    {
      return detail::boundary_factor_sizes<static_key_coordinate_count>(domain_, codomain_);
    }

    auto block_extents(key_type const& key) const -> std::array<std::size_t, static_dense_block_order>
    {
      std::array<std::size_t, static_dense_block_order> extents{};
      std::size_t key_axis = 0;
      std::size_t dense_axis = 0;
      auto append_extent = [&](auto const& space) {
        using space_type = std::remove_cvref_t<decltype(space)>;
        std::size_t coordinate = 0;
        if constexpr (BlockTensorSpaceTraits<space_type>::has_block_coordinate)
        {
          coordinate = key.coordinate(key_axis++);
        }
        if constexpr (BlockTensorSpaceTraits<space_type>::has_dense_axis)
        {
          std::size_t const extent = detail::factor_extent(space, coordinate);
          if (!std::in_range<index_type>(extent))
          {
            throw std::length_error("BlockTensor block extent does not fit index_type");
          }
          extents[dense_axis++] = extent;
        }
      };
      detail::for_each_boundary_space(domain_, append_extent);
      detail::for_each_boundary_space(codomain_, append_extent);
      return extents;
    }

    template <class Function> void for_each_possible_key(Function&& function) const
    {
      auto const sizes = this->factor_sizes();
      if (std::ranges::find(sizes, std::size_t{0}) != sizes.end()) return;

      if constexpr (static_key_coordinate_count == 0)
      {
        function(key_type{});
      }
      else
      {
        std::array<std::size_t, static_key_coordinate_count> coordinates{};
        while (true)
        {
          function(key_type{coordinates});
          std::size_t axis = static_key_coordinate_count;
          while (axis > 0)
          {
            --axis;
            if (++coordinates[axis] < sizes[axis]) break;
            coordinates[axis] = 0;
          }
          if (axis == 0 && coordinates[0] == 0) return;
        }
      }
    }

    auto ordinal(key_type const& key) const -> std::optional<std::size_t>
    {
      auto const keys = storage_.keys();
      auto const found = std::ranges::lower_bound(keys, key);
      if (found == keys.end() || *found != key) return std::nullopt;
      return static_cast<std::size_t>(found - keys.begin());
    }

    Symmetry symmetry_;
    domain_type domain_;
    codomain_type codomain_;
    storage_type storage_;
};

} // namespace uni20
