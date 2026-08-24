/**
 * \file qnum_space.hpp
 * \ingroup symmetry
 * \brief Defines single-irrep tensor spaces.
 */

#pragma once

#include <uni20/symmetry/qnum.hpp>
#include <uni20/symmetry/space.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace uni20
{

/// \brief Tensor space carrying one irreducible-representation label.
class QNumSpace {
  public:
    /// \brief Construct a single-irrep space.
    /// \param qnum Irrep label and its required symmetry context.
    /// \param label Semantic leg label.
    explicit QNumSpace(QNum qnum, std::string label = {}) : qnum_(qnum), label_(std::move(label))
    {
      if (!qnum_.valid())
      {
        throw std::invalid_argument("QNumSpace requires a valid quantum number");
      }
    }

    /// \brief Return the symmetry of the irrep label.
    /// \return QNum-space symmetry.
    auto symmetry() const -> Symmetry { return qnum_.symmetry(); }

    /// \brief Return the irrep label.
    /// \return Stored quantum number.
    auto qnum() const -> QNum const& { return qnum_; }

    /// \brief Return the semantic leg label.
    /// \return Label, which may be empty.
    auto label() const -> std::string const& { return label_; }

    /// \brief Replace the semantic leg label without changing the irrep.
    /// \param label New label.
    void set_label(std::string label) { label_ = std::move(label); }

    /// \brief Compare irrep and label.
    /// \param other Other QNum space.
    /// \return `true` when structure and label are equal.
    auto operator==(QNumSpace const& other) const -> bool = default;

  private:
    QNum qnum_;
    std::string label_;
};

static_assert(SymmetrySpace<QNumSpace>);

} // namespace uni20
