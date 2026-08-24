/**
 * \file local_space.hpp
 * \ingroup symmetry
 * \brief Defines immutable ordered local-state spaces.
 */

#pragma once

#include <uni20/symmetry/qnum.hpp>
#include <uni20/symmetry/space.hpp>

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Ordered local-state space whose quantum numbers may repeat.
/// \details The symmetry and ordered quantum-number structure are immutable.
///          The semantic leg label may be changed without altering that structure.
class LocalSpace {
  public:
    /// \brief Construct an empty local space in an explicit symmetry.
    /// \param sym Symmetry context retained even for an empty state list.
    /// \param label Semantic leg label.
    explicit LocalSpace(Symmetry sym, std::string label = {}) : sym_(sym), label_(std::move(label))
    {
      this->verify_symmetry();
    }

    /// \brief Construct a local space from an initializer list.
    /// \param sym Symmetry shared by every local state.
    /// \param qnums Ordered local-state quantum numbers; repeated values are preserved.
    /// \param label Semantic leg label.
    LocalSpace(Symmetry sym, std::initializer_list<QNum> qnums, std::string label = {})
        : LocalSpace(sym, std::move(label))
    {
      this->append_qnums(qnums);
    }

    /// \brief Construct a local space from an input range.
    /// \tparam QNums Input range whose elements construct `QNum`.
    /// \param sym Symmetry shared by every local state.
    /// \param qnums Ordered local-state quantum numbers; repeated values are preserved.
    /// \param label Semantic leg label.
    template <std::ranges::input_range QNums>
      requires std::constructible_from<QNum, std::ranges::range_reference_t<QNums>>
    LocalSpace(Symmetry sym, QNums&& qnums, std::string label = {}) : LocalSpace(sym, std::move(label))
    {
      this->append_qnums(std::forward<QNums>(qnums));
    }

    /// \brief Construct an immutable local space from a `QNumList`.
    /// \param qnums Ordered local-state quantum numbers and their symmetry.
    /// \param label Semantic leg label.
    explicit LocalSpace(QNumList const& qnums, std::string label = {}) : LocalSpace(qnums.symmetry(), std::move(label))
    {
      this->append_qnums(qnums);
    }

    /// \brief Return the symmetry shared by every local state.
    /// \return Local-space symmetry.
    auto symmetry() const -> Symmetry { return sym_; }

    /// \brief Return the number of explicit local states.
    /// \return Local-state count.
    auto size() const -> std::size_t { return qnums_.size(); }

    /// \brief Return whether the local space has no states.
    /// \return `true` when the local-state list is empty.
    auto empty() const -> bool { return qnums_.empty(); }

    /// \brief Return indexed local-state access.
    /// \param index Zero-based local-state index.
    /// \return Quantum number at the requested index.
    auto operator[](std::size_t index) const -> QNum const& { return qnums_[index]; }

    /// \brief Return the ordered local-state quantum numbers.
    /// \return Read-only span over the local-state list.
    auto qnums() const -> std::span<QNum const> { return qnums_; }

    /// \brief Return an iterator to the first local state.
    /// \return Begin iterator.
    auto begin() const { return qnums_.begin(); }

    /// \brief Return an iterator past the last local state.
    /// \return End iterator.
    auto end() const { return qnums_.end(); }

    /// \brief Return the semantic leg label.
    /// \return Label, which may be empty.
    auto label() const -> std::string const& { return label_; }

    /// \brief Replace the semantic leg label without changing the space structure.
    /// \param label New label.
    void set_label(std::string label) { label_ = std::move(label); }

    /// \brief Compare symmetry, ordered local states, and label.
    /// \param other Other local space.
    /// \return `true` when structure and label are equal.
    auto operator==(LocalSpace const& other) const -> bool = default;

  private:
    /// \brief Validate that the required symmetry handle is initialized.
    void verify_symmetry() const
    {
      if (!sym_.valid())
      {
        throw std::invalid_argument("LocalSpace requires a valid symmetry");
      }
    }

    /// \brief Append construction-time quantum numbers after validating symmetry.
    /// \tparam QNums Input range whose elements construct `QNum`.
    /// \param qnums Ordered quantum-number range.
    template <std::ranges::input_range QNums> void append_qnums(QNums&& qnums)
    {
      for (auto&& input : qnums)
      {
        QNum q(std::forward<decltype(input)>(input));
        if (!q.valid() || q.symmetry() != sym_)
        {
          throw std::invalid_argument("LocalSpace state has the wrong symmetry");
        }
        qnums_.push_back(std::move(q));
      }
    }

    Symmetry sym_;
    std::vector<QNum> qnums_;
    std::string label_;
};

static_assert(SymmetrySpace<LocalSpace>);

} // namespace uni20
