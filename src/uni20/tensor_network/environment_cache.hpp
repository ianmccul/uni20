/**
 * \file environment_cache.hpp
 * \ingroup tensor_network
 * \brief Builds and invalidates finite-chain MPO environments.
 */

#pragma once

#include <uni20/tensor_network/environment.hpp>
#include <uni20/tensor_network/finite_chain.hpp>

#include <concepts>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace uni20::tensor_network
{

/// \brief Directional MPO environment cache borrowed from finite MPS and MPO owners.
/// \details The cache stores one left and one right environment at every bond.
///          Entries are built lazily or with `build_all()`. Site revisions
///          automatically invalidate the exact directional suffix or prefix
///          that depends on a replacement.
/// \warning The referenced owners must outlive the cache, remain at their
///          original addresses, and be mutated only through their replacement
///          members while the cache exists.
/// \tparam MpsChain Concrete `FiniteMps` type.
/// \tparam MpoChain Concrete `FiniteMpo` type.
/// \tparam EnvironmentStorage Immediate sparse host storage for cached values.
template <class MpsChain, class MpoChain, SparseBlockStorage EnvironmentStorage = SeparateSparseBlockStorage<>>
class MpoEnvironmentCache {
  public:
    using mps_type = std::remove_cvref_t<MpsChain>;
    using mpo_type = std::remove_cvref_t<MpoChain>;
    using value_type = typename mps_type::value_type;
    using bond_space_type = typename mps_type::bond_space_type;
    using auxiliary_space_type = typename mpo_type::auxiliary_space_type;
    using environment_type =
        MpoEnvironment<value_type, bond_space_type, auxiliary_space_type, bond_space_type, EnvironmentStorage>;
    using mps_block_type = block_tensor_const_block_t<typename mps_type::site_type>;
    using mpo_block_type = block_tensor_const_block_t<typename mpo_type::site_type>;
    using environment_block_type =
        typename EnvironmentStorage::template storage_t<value_type, 3, 2>::mutable_block_type;

    static_assert(std::same_as<value_type, typename mpo_type::value_type>,
                  "MPS and MPO chains must have the same scalar type");
    static_assert(std::same_as<typename mps_type::physical_space_type, typename mpo_type::physical_space_type>,
                  "MPS and MPO chains must use the same physical-space type");
    static_assert(std::same_as<bond_space_type, BlockSpace>,
                  "the first finite environment cache requires BlockSpace bonds");
    static_assert(std::same_as<auxiliary_space_type, LocalSpace>,
                  "the first finite environment cache requires LocalSpace MPO auxiliaries");
    static_assert(ImmediateBlockTensorView<typename mps_type::site_type>);
    static_assert(HostAccessibleMdspan<immediate_tensor_mdspan_t<mps_block_type>>);
    static_assert(ImmediateBlockTensorView<typename mpo_type::site_type>);
    static_assert(HostAccessibleMdspan<immediate_tensor_mdspan_t<mpo_block_type>>);
    static_assert(ImmediateLocalBlockStorageFor<EnvironmentStorage, value_type, 3, 2>);
    static_assert(HostAccessibleMdspan<mutable_immediate_tensor_mdspan_t<environment_block_type>>);

    /// \brief Construct boundary entries and validate the coupled chains.
    /// \param mps Borrowed finite MPS owner.
    /// \param mpo Borrowed finite MPO owner of the same length.
    /// \param left_auxiliary_index Identity-charge state at the left MPO boundary.
    /// \param right_auxiliary_index Identity-charge state at the right MPO boundary.
    /// \throws std::invalid_argument If chain sizes, symmetries, or physical
    ///         spaces differ, or either boundary state is invalid.
    MpoEnvironmentCache(mps_type const& mps, mpo_type const& mpo, std::size_t left_auxiliary_index,
                        std::size_t right_auxiliary_index)
        : mps_(&mps), mpo_(&mpo), left_(mps.size() + 1), right_(mps.size() + 1), mps_revisions_(mps.size()),
          mpo_revisions_(mpo.size())
    {
      this->validate();
      for (std::size_t index = 0; index < this->size(); ++index)
      {
        mps_revisions_[index] = mps_->revision(index);
        mpo_revisions_[index] = mpo_->revision(index);
      }

      left_[0].emplace(make_identity_mpo_environment<value_type, EnvironmentStorage>(
          mps_->site(0).domain().template space<0>(), mpo_->site(0).domain().template space<0>(),
          left_auxiliary_index));
      right_[this->size()].emplace(make_identity_mpo_environment<value_type, EnvironmentStorage>(
          mps_->site(this->size() - 1).codomain().template space<0>(),
          mpo_->site(this->size() - 1).codomain().template space<0>(), right_auxiliary_index));
    }

    /// \brief Return the number of physical sites.
    auto size() const noexcept -> std::size_t { return mps_->size(); }

    /// \brief Return whether the left environment at a bond is currently cached.
    /// \param bond Bond index in `[0, size()]`.
    /// \throws std::out_of_range If \p bond is invalid.
    auto left_cached(std::size_t bond) -> bool
    {
      this->validate_bond(bond);
      this->synchronize_revisions();
      return left_[bond].has_value();
    }

    /// \brief Return whether the right environment at a bond is currently cached.
    /// \param bond Bond index in `[0, size()]`.
    /// \throws std::out_of_range If \p bond is invalid.
    auto right_cached(std::size_t bond) -> bool
    {
      this->validate_bond(bond);
      this->synchronize_revisions();
      return right_[bond].has_value();
    }

    /// \brief Build and return the left environment at one bond.
    /// \details Bond zero is the left identity boundary. Bond `i+1` extends
    ///          bond `i` across site `i`.
    /// \warning The returned reference is invalidated when a later cache
    ///          observation detects a replacement on which this entry depends.
    /// \param bond Bond index in `[0, size()]`.
    /// \throws std::out_of_range If \p bond is invalid.
    auto left_environment(std::size_t bond) -> environment_type const&
    {
      this->validate_bond(bond);
      this->synchronize_revisions();
      if (!left_[bond])
      {
        std::size_t first_missing = bond;
        while (!left_[first_missing])
          --first_missing;
        for (std::size_t site = first_missing; site < bond; ++site)
        {
          left_[site + 1].emplace(
              extend_left_environment<EnvironmentStorage>(*left_[site], mps_->site(site), mpo_->site(site)));
        }
      }
      return *left_[bond];
    }

    /// \brief Build and return the right environment at one bond.
    /// \details Bond `size()` is the right identity boundary. Bond `i`
    ///          extends bond `i+1` across site `i` from right to left.
    /// \warning The returned reference is invalidated when a later cache
    ///          observation detects a replacement on which this entry depends.
    /// \param bond Bond index in `[0, size()]`.
    /// \throws std::out_of_range If \p bond is invalid.
    auto right_environment(std::size_t bond) -> environment_type const&
    {
      this->validate_bond(bond);
      this->synchronize_revisions();
      if (!right_[bond])
      {
        std::size_t first_available = bond;
        while (!right_[first_available])
          ++first_available;
        for (std::size_t site = first_available; site > bond; --site)
        {
          std::size_t const site_index = site - 1;
          right_[site_index].emplace(extend_right_environment<EnvironmentStorage>(*right_[site], mps_->site(site_index),
                                                                                  mpo_->site(site_index)));
        }
      }
      return *right_[bond];
    }

    /// \brief Build every left and right environment entry.
    void build_all()
    {
      static_cast<void>(this->left_environment(this->size()));
      static_cast<void>(this->right_environment(0));
    }

  private:
    void validate() const
    {
      if (mps_->size() != mpo_->size())
        throw std::invalid_argument("MPO environment cache requires equal nonzero chain lengths");
      if (mps_->symmetry() != mpo_->symmetry())
        throw std::invalid_argument("MPO environment cache requires one symmetry");
      for (std::size_t index = 0; index < mps_->size(); ++index)
      {
        auto const& physical = mps_->site(index).domain().template space<1>();
        if (physical != mpo_->site(index).domain().template space<1>() ||
            physical != mpo_->site(index).codomain().template space<1>())
        {
          throw std::invalid_argument("MPO environment cache physical spaces must compare exactly equal");
        }
      }
    }

    void validate_bond(std::size_t bond) const
    {
      if (bond > this->size()) throw std::out_of_range("MPO environment cache bond index is out of range");
    }

    void synchronize_revisions()
    {
      for (std::size_t site = 0; site < this->size(); ++site)
      {
        std::size_t const mps_revision = mps_->revision(site);
        std::size_t const mpo_revision = mpo_->revision(site);
        if (mps_revision == mps_revisions_[site] && mpo_revision == mpo_revisions_[site]) continue;

        for (std::size_t bond = site + 1; bond <= this->size(); ++bond)
          left_[bond].reset();
        for (std::size_t bond = 0; bond <= site; ++bond)
          right_[bond].reset();
        mps_revisions_[site] = mps_revision;
        mpo_revisions_[site] = mpo_revision;
      }
    }

    mps_type const* mps_;
    mpo_type const* mpo_;
    std::vector<std::optional<environment_type>> left_;
    std::vector<std::optional<environment_type>> right_;
    std::vector<std::size_t> mps_revisions_;
    std::vector<std::size_t> mpo_revisions_;
};

/// \brief Deduce chain owner types while retaining the default environment storage.
template <class MpsChain, class MpoChain>
MpoEnvironmentCache(MpsChain const&, MpoChain const&, std::size_t, std::size_t)
    -> MpoEnvironmentCache<MpsChain, MpoChain>;

} // namespace uni20::tensor_network
