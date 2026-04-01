/**
 * \file finite_triangular_mpo.hpp
 * \brief Minimal finite triangular MPO container for the first DMRG implementation.
 * \details See `docs/operators.md` for the current operator-layer design.
 */

#pragma once

#include <uni20/operator/operator_component.hpp>

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Finite MPO represented as a sequence of upper-triangular site components.
/// \details This is the first lattice-level MPO container needed by the DMRG path.
///          It intentionally models only the triangular finite case rather than a
///          larger MPO hierarchy.
class FiniteTriangularMPO
{
  public:
    using value_type = OperatorComponent;
    using container_type = std::vector<value_type>;
    using size_type = typename container_type::size_type;
    using const_iterator = typename container_type::const_iterator;

    /// \brief Construct an empty finite triangular MPO.
    FiniteTriangularMPO() = default;

    /// \brief Construct from an initializer list of site components.
    /// \param sites Ordered site components.
    FiniteTriangularMPO(std::initializer_list<value_type> sites) : sites_(sites) { this->check_structure(); }

    /// \brief Construct from a vector of site components.
    /// \param sites Ordered site components.
    explicit FiniteTriangularMPO(container_type sites) : sites_(std::move(sites)) { this->check_structure(); }

    /// \brief Return the number of sites.
    /// \return Site count.
    auto size() const -> size_type { return this->sites_.size(); }

    /// \brief Return whether the MPO has no sites.
    /// \return `true` if the MPO is empty.
    auto empty() const -> bool { return this->sites_.empty(); }

    /// \brief Return one site component by index.
    /// \param index Zero-based site index.
    /// \return Site component at the requested position.
    auto operator[](size_type index) const -> value_type const& { return this->sites_[index]; }

    /// \brief Return the first site component.
    /// \return Front site component.
    auto front() const -> value_type const& { return this->sites_.front(); }

    /// \brief Return the last site component.
    /// \return Back site component.
    auto back() const -> value_type const& { return this->sites_.back(); }

    /// \brief Return the iterator to the first site.
    /// \return Begin iterator.
    auto begin() const -> const_iterator { return this->sites_.begin(); }

    /// \brief Return the iterator past the last site.
    /// \return End iterator.
    auto end() const -> const_iterator { return this->sites_.end(); }

    /// \brief Append one site component and revalidate the chain structure.
    /// \param site Site component to append.
    void push_back(value_type site)
    {
        this->sites_.push_back(std::move(site));
        try
        {
            this->check_structure();
        }
        catch (...)
        {
            this->sites_.pop_back();
            throw;
        }
    }

    /// \brief Return the symmetry shared by all sites.
    /// \throws std::logic_error If the MPO is empty.
    /// \return Shared symmetry.
    auto symmetry() const -> Symmetry
    {
        if (this->empty())
        {
            throw std::logic_error("FiniteTriangularMPO::symmetry requires a non-empty MPO");
        }
        return this->front().symmetry();
    }

    /// \brief Return the left boundary virtual space.
    /// \throws std::logic_error If the MPO is empty.
    /// \return Left boundary virtual space.
    auto left_boundary_virtual_space() const -> LocalSpace const&
    {
        if (this->empty())
        {
            throw std::logic_error("FiniteTriangularMPO::left_boundary_virtual_space requires a non-empty MPO");
        }
        return this->front().left_virtual_space();
    }

    /// \brief Return the right boundary virtual space.
    /// \throws std::logic_error If the MPO is empty.
    /// \return Right boundary virtual space.
    auto right_boundary_virtual_space() const -> LocalSpace const&
    {
        if (this->empty())
        {
            throw std::logic_error("FiniteTriangularMPO::right_boundary_virtual_space requires a non-empty MPO");
        }
        return this->back().right_virtual_space();
    }

    /// \brief Validate adjacency, symmetry, and triangularity of the MPO.
    /// \throws std::invalid_argument If adjacent sites are incompatible or any site is not upper triangular.
    void check_structure() const
    {
        for (size_type i = 0; i < this->sites_.size(); ++i)
        {
            auto const& site = this->sites_[i];
            if (!is_upper_triangular(site))
            {
                throw std::invalid_argument("FiniteTriangularMPO site is not upper triangular in its virtual indices");
            }

            if (i == 0)
            {
                continue;
            }

            auto const& left = this->sites_[i - 1];
            if (left.right_virtual_space() != site.left_virtual_space())
            {
                throw std::invalid_argument("FiniteTriangularMPO adjacent virtual spaces do not match");
            }
            if (left.symmetry() != site.symmetry())
            {
                throw std::invalid_argument("FiniteTriangularMPO sites do not share one symmetry");
            }
        }
    }

  private:
    container_type sites_;
};

/// \brief Return whether every site component in a finite MPO is upper triangular.
/// \param mpo Finite triangular MPO to inspect.
/// \return `true` if every site is upper triangular in its virtual indices.
inline auto is_upper_triangular(FiniteTriangularMPO const& mpo) -> bool
{
    for (auto const& site : mpo)
    {
        if (!is_upper_triangular(site))
        {
            return false;
        }
    }
    return true;
}

} // namespace uni20
