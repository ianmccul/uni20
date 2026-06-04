/**
 * \file sparse_mpo_site.hpp
 * \brief Compiled block-sparse MPO-site view for symmetry-aware DMRG prototypes.
 */

#pragma once

#include <uni20/operator/operator_component.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Logical key for one scalar entry in a sparse MPO site tensor.
/// \details The key represents `(left virtual, bra physical, ket physical, right virtual)`.
struct SparseMpoEntryKey
{
    std::size_t left_virtual = 0;
    std::size_t bra = 0;
    std::size_t ket = 0;
    std::size_t right_virtual = 0;

    /// \brief Compare two keys for exact identity.
    /// \param other Other key.
    /// \return `true` if all coordinates match.
    auto operator==(SparseMpoEntryKey const& other) const -> bool = default;
};

/// \brief Hash functor for `SparseMpoEntryKey`.
struct SparseMpoEntryKeyHash
{
    /// \brief Hash one sparse MPO key.
    /// \param key Key to hash.
    /// \return Combined hash value.
    auto operator()(SparseMpoEntryKey const& key) const noexcept -> std::size_t
    {
      auto seed = std::hash<std::size_t>{}(key.left_virtual);
      auto combine = [&](std::size_t value) {
        seed ^= std::hash<std::size_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
      };
      combine(key.bra);
      combine(key.ket);
      combine(key.right_virtual);
      return seed;
    }
};

/// \brief Stored scalar entry in a compiled sparse MPO site tensor.
struct SparseMpoEntry
{
    SparseMpoEntryKey key;
    double value = 0.0;
};

/// \brief Return whether a local-operator coefficient satisfies its transform charge.
/// \details The convention is `q_bra = q_ket + q_operator`.
/// \param op Local operator carrying the transform charge.
/// \param bra Bra physical index.
/// \param ket Ket physical index.
/// \return `true` when the local coefficient is symmetry-allowed.
inline auto local_operator_entry_allowed(LocalOperator const& op, std::size_t bra, std::size_t ket) -> bool
{
  return op.ket_space()[ket] + op.transforms_as() == op.bra_space()[bra];
}

/// \brief Return whether a sparse MPO scalar entry satisfies the zero-flux rule.
/// \details The convention is `q_left_virtual + q_ket = q_right_virtual + q_bra`.
/// \param left_virtual_space Left MPO virtual space.
/// \param bra_space Bra physical space.
/// \param ket_space Ket physical space.
/// \param right_virtual_space Right MPO virtual space.
/// \param key Entry key to test.
/// \return `true` when the MPO scalar entry is symmetry-allowed.
inline auto sparse_mpo_entry_allowed(LocalSpace const& left_virtual_space, LocalSpace const& bra_space,
                                     LocalSpace const& ket_space, LocalSpace const& right_virtual_space,
                                     SparseMpoEntryKey key) -> bool
{
  return left_virtual_space[key.left_virtual] + ket_space[key.ket] ==
         right_virtual_space[key.right_virtual] + bra_space[key.bra];
}

/// \brief Compiled scalar-entry representation of one sparse MPO site tensor.
/// \details `OperatorComponent` remains the model-construction format. This view
///          flattens each stored `LocalOperator` coefficient into one scalar
///          four-leg MPO entry and validates the U(1)-style charge flow used by
///          the block-sparse MPS prototype.
class SparseMpoSite {
  public:
    using key_type = SparseMpoEntryKey;
    using entry_type = SparseMpoEntry;
    using index_type = std::size_t;

    /// \brief Construct a compiled view from an operator component.
    /// \throws std::invalid_argument If any stored coefficient violates charge flow.
    /// \param component Source operator component.
    explicit SparseMpoSite(OperatorComponent const& component)
        : bra_space_(component.local_bra_space()), ket_space_(component.local_ket_space()),
          left_virtual_space_(component.left_virtual_space()), right_virtual_space_(component.right_virtual_space()),
          entries_by_left_virtual_(left_virtual_space_.size()), entries_by_right_virtual_(right_virtual_space_.size()),
          entries_by_bra_(bra_space_.size()), entries_by_ket_(ket_space_.size())
    {
      this->verify_spaces();
      for (std::size_t left = 0; left < component.rows(); ++left)
      {
        for (auto const& virtual_entry : component.data().row(left))
        {
          auto const right = virtual_entry.column;
          auto const& op = virtual_entry.value;
          for (std::size_t bra = 0; bra < op.rows(); ++bra)
          {
            for (auto const& coefficient : op.coefficients().row(bra))
            {
              if (!local_operator_entry_allowed(op, bra, coefficient.column))
              {
                throw std::invalid_argument("local operator coefficient violates q_bra = q_ket + q_operator");
              }
              SparseMpoEntryKey const key{
                  .left_virtual = left,
                  .bra = bra,
                  .ket = coefficient.column,
                  .right_virtual = right,
              };
              this->insert_entry(key, coefficient.value);
            }
          }
        }
      }
    }

    /// \brief Return the bra physical space.
    /// \return Bra physical space.
    auto bra_space() const -> LocalSpace const& { return bra_space_; }

    /// \brief Return the ket physical space.
    /// \return Ket physical space.
    auto ket_space() const -> LocalSpace const& { return ket_space_; }

    /// \brief Return the left virtual space.
    /// \return Left virtual space.
    auto left_virtual_space() const -> LocalSpace const& { return left_virtual_space_; }

    /// \brief Return the right virtual space.
    /// \return Right virtual space.
    auto right_virtual_space() const -> LocalSpace const& { return right_virtual_space_; }

    /// \brief Return all stored scalar entries.
    /// \return Read-only entry span.
    auto entries() const -> std::span<entry_type const> { return entries_; }

    /// \brief Return the number of stored scalar entries.
    /// \return Entry count.
    auto nnz() const -> std::size_t { return entries_.size(); }

    /// \brief Return whether one scalar entry exists.
    /// \param key Entry coordinate.
    /// \return `true` if the entry is stored.
    auto contains(key_type key) const -> bool { return lookup_.contains(key); }

    /// \brief Return one scalar coefficient.
    /// \throws std::out_of_range If no entry exists at `key`.
    /// \param key Entry coordinate.
    /// \return Stored coefficient.
    auto at(key_type key) const -> double { return entries_.at(this->entry_index(key)).value; }

    /// \brief Return entry indexes with one left virtual coordinate.
    /// \param left_virtual Left virtual index.
    /// \return Entry indexes.
    auto entries_from_left_virtual(index_type left_virtual) const -> std::span<index_type const>
    {
      return entries_by_left_virtual_.at(left_virtual);
    }

    /// \brief Return entry indexes with one right virtual coordinate.
    /// \param right_virtual Right virtual index.
    /// \return Entry indexes.
    auto entries_to_right_virtual(index_type right_virtual) const -> std::span<index_type const>
    {
      return entries_by_right_virtual_.at(right_virtual);
    }

    /// \brief Return entry indexes with one bra physical coordinate.
    /// \param bra Bra physical index.
    /// \return Entry indexes.
    auto entries_for_bra(index_type bra) const -> std::span<index_type const> { return entries_by_bra_.at(bra); }

    /// \brief Return entry indexes with one ket physical coordinate.
    /// \param ket Ket physical index.
    /// \return Entry indexes.
    auto entries_for_ket(index_type ket) const -> std::span<index_type const> { return entries_by_ket_.at(ket); }

  private:
    void verify_spaces() const
    {
      auto const sym = ket_space_.symmetry();
      if (bra_space_.symmetry() != sym || left_virtual_space_.symmetry() != sym ||
          right_virtual_space_.symmetry() != sym)
      {
        throw std::invalid_argument("SparseMpoSite spaces must share one symmetry");
      }
    }

    void validate_key_ranges(key_type key) const
    {
      if (key.left_virtual >= left_virtual_space_.size() || key.bra >= bra_space_.size() ||
          key.ket >= ket_space_.size() || key.right_virtual >= right_virtual_space_.size())
      {
        throw std::out_of_range("sparse MPO entry key is out of range");
      }
    }

    void insert_entry(key_type key, double value)
    {
      this->validate_key_ranges(key);
      if (!sparse_mpo_entry_allowed(left_virtual_space_, bra_space_, ket_space_, right_virtual_space_, key))
      {
        throw std::invalid_argument("sparse MPO entry violates q_left + q_ket = q_right + q_bra");
      }
      if (this->contains(key))
      {
        throw std::invalid_argument("duplicate sparse MPO scalar entry");
      }

      auto const index = entries_.size();
      entries_.push_back(entry_type{.key = key, .value = value});
      lookup_.emplace(key, index);
      entries_by_left_virtual_[key.left_virtual].push_back(index);
      entries_by_right_virtual_[key.right_virtual].push_back(index);
      entries_by_bra_[key.bra].push_back(index);
      entries_by_ket_[key.ket].push_back(index);
    }

    auto entry_index(key_type key) const -> index_type
    {
      this->validate_key_ranges(key);
      auto const it = lookup_.find(key);
      if (it == lookup_.end())
      {
        throw std::out_of_range("sparse MPO entry is not present");
      }
      return it->second;
    }

    LocalSpace bra_space_;
    LocalSpace ket_space_;
    LocalSpace left_virtual_space_;
    LocalSpace right_virtual_space_;
    std::vector<entry_type> entries_;
    std::unordered_map<key_type, index_type, SparseMpoEntryKeyHash> lookup_;
    std::vector<std::vector<index_type>> entries_by_left_virtual_;
    std::vector<std::vector<index_type>> entries_by_right_virtual_;
    std::vector<std::vector<index_type>> entries_by_bra_;
    std::vector<std::vector<index_type>> entries_by_ket_;
};

} // namespace uni20
