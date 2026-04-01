/**
 * \file local_space.hpp
 * \brief Explicit sparse operator-space wrapper built on `uni20::QNumList`.
 * \details See `docs/operators.md` for the current operator-layer design.
 */

#pragma once

#include <uni20/symmetry/qnum.hpp>

#include <cstddef>
#include <initializer_list>
#include <utility>

namespace uni20
{

/// \brief Explicit sparse operator-space represented as an ordered list of irreps.
/// \details `LocalSpace` is a semantic wrapper over `QNumList`. In the current
///          operator layer it is used both for physical local states and for MPO
///          auxiliary spaces, while preserving symmetry, ordering, and repeated labels.
class LocalSpace
{
  public:
    using value_type = QNum;

    /// \brief Construct an empty explicit sparse space for one symmetry.
    /// \param sym Symmetry carried by every state.
    explicit LocalSpace(Symmetry sym) : qnums_(sym) {}

    /// \brief Construct a singleton space from one irrep.
    /// \param q Quantum number of the single state.
    explicit LocalSpace(QNum q) : qnums_(q.symmetry(), {q}) {}

    /// \brief Construct a space from an explicit list of irreps.
    /// \param sym Symmetry carried by every state.
    /// \param values Ordered state irreps.
    LocalSpace(Symmetry sym, std::initializer_list<QNum> values) : qnums_(sym, values) {}

    /// \brief Construct a local space from an existing sparse irrep list.
    /// \param qnums Ordered sparse irrep list.
    explicit LocalSpace(QNumList qnums) : qnums_(std::move(qnums)) {}

    /// \brief Return the symmetry shared by all states.
    /// \return Space symmetry.
    auto symmetry() const -> Symmetry { return this->qnums_.symmetry(); }

    /// \brief Return the number of states.
    /// \return State count.
    auto size() const -> std::size_t { return this->qnums_.size(); }

    /// \brief Return whether the space is empty.
    /// \return `true` if the space contains no states.
    auto empty() const -> bool { return this->qnums_.empty(); }

    /// \brief Append one state irrep.
    /// \param q Quantum number to append.
    void push_back(QNum q) { this->qnums_.push_back(q); }

    /// \brief Remove all states while keeping the symmetry.
    void clear() { this->qnums_.clear(); }

    /// \brief Return whether the space contains a given irrep.
    /// \param q Quantum number to search for.
    /// \return `true` if the irrep appears in the state list.
    auto contains(QNum q) const -> bool { return this->qnums_.contains(q); }

    /// \brief Return one state irrep by index.
    /// \param index Zero-based state index.
    /// \return Quantum number stored at the requested position.
    auto operator[](std::size_t index) const -> QNum const& { return this->qnums_[index]; }

    /// \brief Return the underlying sparse irrep list.
    /// \return Reference to the wrapped `QNumList`.
    auto qnums() const -> QNumList const& { return this->qnums_; }

    /// \brief Return an iterator to the first state irrep.
    /// \return Begin iterator.
    auto begin() const { return this->qnums_.begin(); }

    /// \brief Return an iterator past the last state irrep.
    /// \return End iterator.
    auto end() const { return this->qnums_.end(); }

    /// \brief Compare two local spaces for exact equality.
    /// \param other Other local space.
    /// \return `true` if symmetry and ordered state list match.
    auto operator==(LocalSpace const& other) const -> bool = default;

  private:
    QNumList qnums_;
};

} // namespace uni20
