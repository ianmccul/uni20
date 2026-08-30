/**
 * \file finite_chain.hpp
 * \ingroup tensor_network
 * \brief Owns validated finite MPS and MPO chains.
 */

#pragma once

#include <uni20/symmetry/block_tensor_storage.hpp>
#include <uni20/symmetry/space.hpp>
#include <uni20/tensor_network/site_types.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Homogeneous finite MPS owner with exact bond connectivity.
/// \details Sites are exposed read-only. Numerical or structural changes must
///          use the replacement members so dependent environment caches can
///          detect the changed sites through their revisions.
/// \tparam Scalar Numerical site element type.
/// \tparam Bond Concrete left and right bond-space type.
/// \tparam Physical Concrete physical-space type.
/// \tparam Storage Block storage policy used by every site.
template <typename Scalar, Space Bond, Space Physical, BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
class FiniteMps {
  public:
    using value_type = std::remove_cv_t<Scalar>;
    using bond_space_type = Bond;
    using physical_space_type = Physical;
    using storage_policy = Storage;
    using site_type = MpsSite<Scalar, Bond, Physical, Bond, Storage>;

    /// \brief Construct a nonempty chain and validate every shared bond.
    /// \param sites Ordered MPS sites from left to right.
    /// \throws std::invalid_argument If the chain is empty, mixes symmetries,
    ///         or contains unequal adjacent bond spaces.
    explicit FiniteMps(std::vector<site_type> sites) : sites_(std::move(sites)), revisions_(sites_.size())
    {
      this->validate();
    }

    /// \brief Return the number of sites.
    auto size() const noexcept -> std::size_t { return sites_.size(); }

    /// \brief Return the common symmetry context.
    auto symmetry() const -> Symmetry { return sites_.front().symmetry(); }

    /// \brief Return one read-only site.
    /// \param index Zero-based site index.
    /// \throws std::out_of_range If \p index is outside the chain.
    auto site(std::size_t index) const -> site_type const& { return sites_.at(index); }

    /// \brief Return all sites in left-to-right order.
    auto sites() const noexcept -> std::span<site_type const> { return sites_; }

    /// \brief Return the revision of one site.
    /// \param index Zero-based site index.
    /// \throws std::out_of_range If \p index is outside the chain.
    auto revision(std::size_t index) const -> std::size_t { return revisions_.at(index); }

    /// \brief Replace one site without changing any of its spaces.
    /// \details This operation is the numerical-update path. A bond-space
    ///          change requires `replace_pair()` so both ends change together.
    /// \param index Zero-based site index.
    /// \param replacement New site with exactly equal domain and codomain.
    /// \throws std::invalid_argument If symmetry or spaces differ.
    /// \throws std::out_of_range If \p index is outside the chain.
    void replace_site(std::size_t index, site_type replacement)
    {
      site_type const& current = sites_.at(index);
      if (replacement.symmetry() != this->symmetry() || replacement.domain() != current.domain() ||
          replacement.codomain() != current.codomain())
      {
        throw std::invalid_argument("FiniteMps replacement site must preserve every space");
      }
      sites_[index] = std::move(replacement);
      ++revisions_[index];
    }

    /// \brief Replace adjacent sites while permitting their shared bond to change.
    /// \details The pair's external bonds and physical spaces remain exactly
    ///          equal to the current pair. The two replacement sites must share
    ///          one exactly equal internal bond.
    /// \param first_index Index of the left site in the pair.
    /// \param first New left site.
    /// \param second New right site.
    /// \throws std::invalid_argument If symmetry, external spaces, physical
    ///         spaces, or the new internal connection are incompatible.
    /// \throws std::out_of_range If there is no adjacent pair at \p first_index.
    void replace_pair(std::size_t first_index, site_type first, site_type second)
    {
      if (first_index >= this->size() || first_index + 1 >= this->size())
        throw std::out_of_range("FiniteMps pair index is out of range");

      site_type const& current_first = sites_[first_index];
      site_type const& current_second = sites_[first_index + 1];
      if (first.symmetry() != this->symmetry() || second.symmetry() != this->symmetry() ||
          first.domain() != current_first.domain() || second.codomain() != current_second.codomain() ||
          second.domain().template space<1>() != current_second.domain().template space<1>() ||
          first.codomain().template space<0>() != second.domain().template space<0>())
      {
        throw std::invalid_argument("FiniteMps replacement pair has incompatible spaces");
      }

      sites_[first_index] = std::move(first);
      sites_[first_index + 1] = std::move(second);
      ++revisions_[first_index];
      ++revisions_[first_index + 1];
    }

  private:
    void validate() const
    {
      if (sites_.empty()) throw std::invalid_argument("FiniteMps requires at least one site");
      Symmetry const chain_symmetry = sites_.front().symmetry();
      for (std::size_t index = 0; index < sites_.size(); ++index)
      {
        if (sites_[index].symmetry() != chain_symmetry)
          throw std::invalid_argument("FiniteMps sites must share one symmetry");
        if (index + 1 < sites_.size() &&
            sites_[index].codomain().template space<0>() != sites_[index + 1].domain().template space<0>())
        {
          throw std::invalid_argument("FiniteMps adjacent bond spaces must compare exactly equal");
        }
      }
    }

    std::vector<site_type> sites_;
    std::vector<std::size_t> revisions_;
};

/// \brief Homogeneous finite MPO owner with exact auxiliary connectivity.
/// \details MPO sites are exposed read-only and may be replaced only without
///          changing their spaces.
/// \tparam Scalar Numerical MPO element type.
/// \tparam Auxiliary Concrete left and right auxiliary-space type.
/// \tparam Physical Concrete input and output physical-space type.
/// \tparam Storage Block storage policy used by every site.
template <typename Scalar, Space Auxiliary, Space Physical, BlockTensorStorage Storage = SeparateSparseBlockStorage<>>
class FiniteMpo {
  public:
    using value_type = std::remove_cv_t<Scalar>;
    using auxiliary_space_type = Auxiliary;
    using physical_space_type = Physical;
    using storage_policy = Storage;
    using site_type = MpoSite<Scalar, Auxiliary, Physical, Auxiliary, Physical, Storage>;

    /// \brief Construct a nonempty chain and validate every shared auxiliary.
    /// \param sites Ordered MPO sites from left to right.
    /// \throws std::invalid_argument If the chain is empty, mixes symmetries,
    ///         or contains unequal adjacent auxiliary spaces.
    explicit FiniteMpo(std::vector<site_type> sites) : sites_(std::move(sites)), revisions_(sites_.size())
    {
      this->validate();
    }

    /// \brief Return the number of sites.
    auto size() const noexcept -> std::size_t { return sites_.size(); }

    /// \brief Return the common symmetry context.
    auto symmetry() const -> Symmetry { return sites_.front().symmetry(); }

    /// \brief Return one read-only site.
    /// \param index Zero-based site index.
    /// \throws std::out_of_range If \p index is outside the chain.
    auto site(std::size_t index) const -> site_type const& { return sites_.at(index); }

    /// \brief Return all sites in left-to-right order.
    auto sites() const noexcept -> std::span<site_type const> { return sites_; }

    /// \brief Return the revision of one site.
    /// \param index Zero-based site index.
    /// \throws std::out_of_range If \p index is outside the chain.
    auto revision(std::size_t index) const -> std::size_t { return revisions_.at(index); }

    /// \brief Replace one MPO site without changing any of its spaces.
    /// \param index Zero-based site index.
    /// \param replacement New site with exactly equal domain and codomain.
    /// \throws std::invalid_argument If symmetry or spaces differ.
    /// \throws std::out_of_range If \p index is outside the chain.
    void replace_site(std::size_t index, site_type replacement)
    {
      site_type const& current = sites_.at(index);
      if (replacement.symmetry() != this->symmetry() || replacement.domain() != current.domain() ||
          replacement.codomain() != current.codomain())
      {
        throw std::invalid_argument("FiniteMpo replacement site must preserve every space");
      }
      sites_[index] = std::move(replacement);
      ++revisions_[index];
    }

  private:
    void validate() const
    {
      if (sites_.empty()) throw std::invalid_argument("FiniteMpo requires at least one site");
      Symmetry const chain_symmetry = sites_.front().symmetry();
      for (std::size_t index = 0; index < sites_.size(); ++index)
      {
        if (sites_[index].symmetry() != chain_symmetry)
          throw std::invalid_argument("FiniteMpo sites must share one symmetry");
        if (index + 1 < sites_.size() &&
            sites_[index].codomain().template space<0>() != sites_[index + 1].domain().template space<0>())
        {
          throw std::invalid_argument("FiniteMpo adjacent auxiliary spaces must compare exactly equal");
        }
      }
    }

    std::vector<site_type> sites_;
    std::vector<std::size_t> revisions_;
};

} // namespace uni20::tensor_network
