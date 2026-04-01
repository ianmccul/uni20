/**
 * \file operator_component.hpp
 * \brief Per-site MPO component represented as a sparse matrix of local operators.
 * \details See `docs/operators.md` for the current operator-layer design.
 */

#pragma once

#include <uni20/matrix/sparse_matrix.hpp>
#include <uni20/operator/local_operator.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace uni20
{

/// \brief One site component of an MPO-like object.
/// \details `OperatorComponent` is the fundamental per-site object above
///          `LocalOperator`. It stores left/right virtual spaces together with a
///          sparse matrix of local operators.
class OperatorComponent
{
  public:
    using value_type = LocalOperator;
    using matrix_type = SparseMatrix<value_type>;
    using index_type = typename matrix_type::index_type;

    /// \brief Construct an empty operator component.
    /// \param local_bra_space Output local space shared by every matrix entry.
    /// \param local_ket_space Input local space shared by every matrix entry.
    /// \param left_virtual_space Left virtual space.
    /// \param right_virtual_space Right virtual space.
    OperatorComponent(LocalSpace local_bra_space, LocalSpace local_ket_space, LocalSpace left_virtual_space,
                      LocalSpace right_virtual_space)
        : local_bra_space_(std::move(local_bra_space)),
          local_ket_space_(std::move(local_ket_space)),
          left_virtual_space_(std::move(left_virtual_space)),
          right_virtual_space_(std::move(right_virtual_space)),
          data_(this->left_virtual_space_.size(), this->right_virtual_space_.size())
    {
        this->verify_symmetry();
    }

    /// \brief Construct an operator component from an existing sparse matrix of local operators.
    /// \param local_bra_space Output local space shared by every matrix entry.
    /// \param local_ket_space Input local space shared by every matrix entry.
    /// \param left_virtual_space Left virtual space.
    /// \param right_virtual_space Right virtual space.
    /// \param data Sparse matrix of local operators with shape
    ///             `(left_virtual_space.size(), right_virtual_space.size())`.
    OperatorComponent(LocalSpace local_bra_space, LocalSpace local_ket_space, LocalSpace left_virtual_space,
                      LocalSpace right_virtual_space, matrix_type data)
        : local_bra_space_(std::move(local_bra_space)),
          local_ket_space_(std::move(local_ket_space)),
          left_virtual_space_(std::move(left_virtual_space)),
          right_virtual_space_(std::move(right_virtual_space)),
          data_(std::move(data))
    {
        this->verify_symmetry();
        if (this->data_.rows() != this->left_virtual_space_.size() || this->data_.cols() != this->right_virtual_space_.size())
        {
            throw std::invalid_argument("OperatorComponent sparse matrix shape does not match its virtual spaces");
        }
        this->verify_entries();
    }

    /// \brief Return the output local space shared by every entry.
    /// \return Local bra space.
    auto local_bra_space() const -> LocalSpace const& { return this->local_bra_space_; }

    /// \brief Return the input local space shared by every entry.
    /// \return Local ket space.
    auto local_ket_space() const -> LocalSpace const& { return this->local_ket_space_; }

    /// \brief Return the left virtual space.
    /// \return Left bond space.
    auto left_virtual_space() const -> LocalSpace const& { return this->left_virtual_space_; }

    /// \brief Return the right virtual space.
    /// \return Right bond space.
    auto right_virtual_space() const -> LocalSpace const& { return this->right_virtual_space_; }

    /// \brief Return the shared symmetry of local and virtual spaces.
    /// \return Component symmetry.
    auto symmetry() const -> Symmetry { return this->left_virtual_space_.symmetry(); }

    /// \brief Return the number of left virtual states.
    /// \return Sparse matrix row count.
    auto rows() const -> index_type { return this->data_.rows(); }

    /// \brief Return the number of right virtual states.
    /// \return Sparse matrix column count.
    auto cols() const -> index_type { return this->data_.cols(); }

    /// \brief Return the number of stored local-operator entries.
    /// \return Count of sparse matrix entries.
    auto nnz() const -> index_type { return this->data_.nnz(); }

    /// \brief Return whether an entry exists at one virtual-space coordinate.
    /// \param row Left virtual-space index.
    /// \param col Right virtual-space index.
    /// \return `true` if a local operator is stored at `(row, col)`.
    auto contains(index_type row, index_type col) const -> bool { return this->data_.contains(row, col); }

    /// \brief Return the sparse matrix of local operators.
    /// \return Reference to the stored sparse matrix.
    auto data() const -> matrix_type const& { return this->data_; }

    /// \brief Return the sparse matrix of local operators.
    /// \return Mutable reference to the stored sparse matrix.
    auto data() -> matrix_type& { return this->data_; }

    /// \brief Return a stored local operator.
    /// \throws std::out_of_range If no entry is stored at `(row, col)`.
    /// \param row Left virtual-space index.
    /// \param col Right virtual-space index.
    /// \return Stored local operator at `(row, col)`.
    auto at(index_type row, index_type col) const -> value_type const& { return this->data_.at(row, col); }

    /// \brief Insert or overwrite one local-operator entry.
    /// \throws std::invalid_argument If the local operator is incompatible with this component.
    /// \param row Left virtual-space index.
    /// \param col Right virtual-space index.
    /// \param op Local operator to store.
    /// \return Reference to the stored entry.
    auto insert_or_assign(index_type row, index_type col, value_type op) -> value_type&
    {
        this->verify_entry(op);
        return this->data_.insert_or_assign(row, col, std::move(op));
    }

    /// \brief Remove one stored local-operator entry if present.
    /// \param row Left virtual-space index.
    /// \param col Right virtual-space index.
    /// \return `true` if an entry was removed.
    auto erase(index_type row, index_type col) -> bool { return this->data_.erase(row, col); }

    /// \brief Remove all stored local-operator entries.
    void clear() { this->data_.clear(); }

  private:
    /// \brief Validate that local and virtual spaces share one symmetry.
    void verify_symmetry() const
    {
        auto const sym = this->local_ket_space_.symmetry();
        if (this->local_bra_space_.symmetry() != sym)
        {
            throw std::invalid_argument("OperatorComponent local bra and ket spaces must share one symmetry");
        }
        if (this->left_virtual_space_.symmetry() != sym || this->right_virtual_space_.symmetry() != sym)
        {
            throw std::invalid_argument("OperatorComponent virtual spaces must match the local symmetry");
        }
    }

    /// \brief Validate one local operator against the component spaces.
    /// \param op Local operator to validate.
    void verify_entry(LocalOperator const& op) const
    {
        if (op.bra_space() != this->local_bra_space_ || op.ket_space() != this->local_ket_space_)
        {
            throw std::invalid_argument("OperatorComponent entry uses incompatible local spaces");
        }
    }

    /// \brief Validate every stored entry in the sparse matrix.
    void verify_entries() const
    {
        for (index_type row = 0; row < this->data_.rows(); ++row)
        {
            for (auto const& entry : this->data_.row(row))
            {
                this->verify_entry(entry.value);
            }
        }
    }

    LocalSpace local_bra_space_;
    LocalSpace local_ket_space_;
    LocalSpace left_virtual_space_;
    LocalSpace right_virtual_space_;
    matrix_type data_;
};

/// \brief Return whether an operator component is upper triangular in its virtual indices.
/// \details For rectangular components, this uses the natural convention that entries
///          strictly below the main diagonal are forbidden, i.e. stored entries must
///          satisfy `row <= column`.
/// \param component Operator component to inspect.
/// \return `true` if no stored entry lies below the main diagonal.
inline auto is_upper_triangular(OperatorComponent const& component) -> bool
{
    for (OperatorComponent::index_type row = 0; row < component.rows(); ++row)
    {
        for (auto const& entry : component.data().row(row))
        {
            if (row > entry.column)
            {
                return false;
            }
        }
    }
    return true;
}

} // namespace uni20
