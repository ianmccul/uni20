/**
 * \file dense_space.hpp
 * \ingroup symmetry
 * \brief Defines symmetry-neutral dense tensor spaces.
 */

#pragma once

#include <uni20/symmetry/space.hpp>

#include <cstddef>
#include <string>
#include <utility>

namespace uni20
{

/// \brief Symmetry-neutral dense tensor space with one fixed extent.
/// \details A dense leg bundles structurally identical symmetric tensors without
///          contributing a quantum number to the block-selection rule.
class DenseSpace {
  public:
    /// \brief Construct a dense space.
    /// \param extent Number of dense entries; zero is permitted.
    /// \param label Semantic leg label.
    explicit DenseSpace(std::size_t extent, std::string label = {}) : extent_(extent), label_(std::move(label)) {}

    /// \brief Return the dense extent.
    /// \return Number of dense entries.
    auto extent() const -> std::size_t { return extent_; }

    /// \brief Return whether the dense space has zero extent.
    /// \return `true` when the extent is zero.
    auto empty() const -> bool { return extent_ == 0; }

    /// \brief Return the semantic leg label.
    /// \return Label, which may be empty.
    auto label() const -> std::string const& { return label_; }

    /// \brief Replace the semantic leg label without changing the extent.
    /// \param label New label.
    void set_label(std::string label) { label_ = std::move(label); }

    /// \brief Compare dense extent and label.
    /// \param other Other dense space.
    /// \return `true` when extent and label are equal.
    auto operator==(DenseSpace const& other) const -> bool = default;

  private:
    std::size_t extent_ = 0;
    std::string label_;
};

static_assert(Space<DenseSpace>);
static_assert(!SymmetrySpace<DenseSpace>);

} // namespace uni20
