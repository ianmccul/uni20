/**
 * \file dual_space.hpp
 * \ingroup symmetry
 * \brief Defines the generic categorical dual-space adaptor.
 */

#pragma once

#include <uni20/symmetry/block_sector.hpp>
#include <uni20/symmetry/qnum.hpp>
#include <uni20/symmetry/space.hpp>

#include <concepts>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace uni20
{

template <class SpaceType> class Dual;

namespace detail
{

template <class T> struct IsDualSpace : std::false_type
{};

template <class SpaceType> struct IsDualSpace<Dual<SpaceType>> : std::true_type
{};

template <class T> struct PrimalSpaceType
{
    using type = std::remove_cvref_t<T>;
};

template <class SpaceType> struct PrimalSpaceType<Dual<SpaceType>>
{
    using type = SpaceType;
};

inline auto dual_space_entry(QNum const& q) -> QNum { return dual(q); }

inline auto dual_space_entry(BlockSector const& sector) -> BlockSector
{
  return BlockSector{dual(sector.q), sector.dim};
}

template <std::input_iterator Iterator> class DualSpaceIterator {
  public:
    using iterator_concept = std::input_iterator_tag;
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::iter_difference_t<Iterator>;
    using value_type = std::remove_cvref_t<decltype(dual_space_entry(*std::declval<Iterator const&>()))>;

    DualSpaceIterator()
      requires std::default_initializable<Iterator>
    = default;

    explicit DualSpaceIterator(Iterator iterator) : iterator_(std::move(iterator)) {}

    auto operator*() const -> value_type { return dual_space_entry(*iterator_); }

    auto operator++() -> DualSpaceIterator&
    {
      ++iterator_;
      return *this;
    }

    void operator++(int) { ++iterator_; }

    auto operator==(DualSpaceIterator const&) const -> bool = default;

  private:
    Iterator iterator_;
};

} // namespace detail

/// \brief Tensor space represented in an explicitly dual basis.
/// \details The adaptor preserves the underlying basis occurrence order,
///          dimensions, and label. Quantum-number observations return the
///          categorical dual without rebuilding or canonicalizing the space.
/// \tparam SpaceType Adapted concrete space type.
template <class SpaceType> class Dual {
    static_assert(Space<SpaceType>, "Dual requires a tensor-space value");
    static_assert(!detail::IsDualSpace<std::remove_cvref_t<SpaceType>>::value,
                  "use dual(Dual<S>) to recover S instead of nesting Dual adaptors");

  public:
    using base_space_type = SpaceType;

    /// \brief Construct a dual adaptor from its primal-basis value.
    /// \param space Underlying space value whose occurrence order is retained.
    explicit Dual(base_space_type space) : space_(std::move(space)) {}

    /// \brief Return the underlying primal-basis space.
    auto base() const& noexcept -> base_space_type const& { return space_; }

    /// \brief Return the symmetry shared with the underlying space.
    auto symmetry() const -> Symmetry
      requires SymmetrySpace<base_space_type>
    {
      return space_.symmetry();
    }

    /// \brief Return the semantic leg label.
    auto label() const -> std::string const& { return space_.label(); }

    /// \brief Replace the semantic leg label without changing the basis.
    /// \param label New label.
    void set_label(std::string label) { space_.set_label(std::move(label)); }

    /// \brief Forward the number of structural basis occurrences.
    auto size() const -> std::size_t
      requires requires(base_space_type const& space) {
        { space.size() } -> std::same_as<std::size_t>;
      }
    {
      return space_.size();
    }

    /// \brief Forward whether the underlying space is empty.
    auto empty() const -> bool
      requires requires(base_space_type const& space) {
        { space.empty() } -> std::convertible_to<bool>;
      }
    {
      return space_.empty();
    }

    /// \brief Forward the total degeneracy dimension.
    auto total_dim() const -> std::size_t
      requires requires(base_space_type const& space) {
        { space.total_dim() } -> std::same_as<std::size_t>;
      }
    {
      return space_.total_dim();
    }

    /// \brief Forward a symmetry-neutral dense extent.
    auto extent() const -> std::size_t
      requires requires(base_space_type const& space) {
        { space.extent() } -> std::same_as<std::size_t>;
      }
    {
      return space_.extent();
    }

    /// \brief Return the dual of a fixed irrep label.
    auto qnum() const -> QNum
      requires requires(base_space_type const& space) {
        { space.qnum() } -> std::convertible_to<QNum>;
      }
    {
      return dual(space_.qnum());
    }

    /// \brief Return one dualized basis occurrence without changing its index.
    /// \param index Zero-based occurrence index in the underlying space.
    auto operator[](std::size_t index) const
      requires requires(base_space_type const& space) { space[index]; }
    {
      using entry_type = std::remove_cvref_t<decltype(space_[index])>;
      static_assert(std::same_as<entry_type, QNum> || std::same_as<entry_type, BlockSector>,
                    "indexed Dual-space access currently supports QNum and BlockSector entries");
      return detail::dual_space_entry(space_[index]);
    }

    /// \brief Return an iterator which dualizes each observed basis occurrence.
    auto begin() const
      requires requires(base_space_type const& space) { space.begin(); }
    {
      return detail::DualSpaceIterator(space_.begin());
    }

    /// \brief Return the end iterator for dualized basis occurrences.
    auto end() const
      requires requires(base_space_type const& space) { space.end(); }
    {
      return detail::DualSpaceIterator(space_.end());
    }

    /// \brief Return a lazy range of dualized local-state quantum numbers.
    auto qnums() const
      requires requires(base_space_type const& space) { space.qnums(); }
    {
      return space_.qnums() | std::views::transform([](QNum const& q) { return dual(q); });
    }

    /// \brief Return a lazy range of dualized block sectors.
    auto sectors() const
      requires requires(base_space_type const& space) { space.sectors(); }
    {
      return space_.sectors() |
             std::views::transform([](BlockSector const& sector) { return BlockSector{dual(sector.q), sector.dim}; });
    }

    /// \brief Return whether the exposed dual basis contains an irrep.
    /// \param q Quantum number expressed in the dual basis.
    auto contains(QNum q) const -> bool
      requires SymmetrySpace<base_space_type> && requires(base_space_type const& space, QNum value) {
        { space.contains(value) } -> std::convertible_to<bool>;
      }
    {
      if (!q.valid() || q.symmetry() != this->symmetry())
      {
        throw std::invalid_argument("Dual-space query has the wrong symmetry");
      }
      return space_.contains(dual(q));
    }

    /// \brief Compare explicit dual-basis space values.
    auto operator==(Dual const& other) const -> bool = default;

    /// \brief Transfer the underlying primal-basis value out of this adaptor.
    auto release_base() && -> base_space_type { return std::move(space_); }

  private:
    base_space_type space_;
};

template <class SpaceType> Dual(SpaceType&&) -> Dual<std::remove_cvref_t<SpaceType>>;

/// \brief Space value whose object basis is explicitly dual.
/// \tparam T Candidate space type.
template <class T>
concept DualSpace = Space<T> && detail::IsDualSpace<std::remove_cvref_t<T>>::value;

/// \brief Remove one explicit dual-space adaptor from a space type.
/// \tparam T Space or dual-space type.
template <class T> using primal_space_t = typename detail::PrimalSpaceType<std::remove_cvref_t<T>>::type;

/// \brief Construct an explicitly dual basis without changing occurrence order.
/// \tparam SpaceType Concrete primal-basis space type.
/// \param space Space value to adapt.
/// \return Dual-basis adaptor owning a copy of the space value.
template <Space SpaceType>
  requires(!DualSpace<SpaceType>)
auto dual(SpaceType const& space) -> Dual<std::remove_cvref_t<SpaceType>>
{
  return Dual<std::remove_cvref_t<SpaceType>>(space);
}

/// \brief Construct an explicitly dual basis by moving a space value.
/// \tparam SpaceType Concrete primal-basis space type.
/// \param space Space value to adapt.
/// \return Dual-basis adaptor owning the moved space value.
template <Space SpaceType>
  requires(!DualSpace<SpaceType>)
auto dual(SpaceType&& space) -> Dual<std::remove_cvref_t<SpaceType>>
{
  return Dual<std::remove_cvref_t<SpaceType>>(std::forward<SpaceType>(space));
}

/// \brief Remove explicit duality from a dual-space value.
/// \tparam SpaceType Underlying primal-basis space type.
/// \param space Dual-basis space value.
/// \return Copy of the underlying primal-basis value.
template <class SpaceType> auto dual(Dual<SpaceType> const& space) -> SpaceType { return space.base(); }

/// \brief Remove explicit duality while moving a dual-space value.
/// \tparam SpaceType Underlying primal-basis space type.
/// \param space Dual-basis space value.
/// \return Moved underlying primal-basis value.
template <class SpaceType> auto dual(Dual<SpaceType>&& space) -> SpaceType { return std::move(space).release_base(); }

} // namespace uni20
