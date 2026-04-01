/**
 * \file local_operator.hpp
 * \brief Sparse coefficient representation of local symmetry operators.
 * \details See `docs/operators.md` for the current operator-layer design.
 */

#pragma once

#include <uni20/matrix/sparse_matrix.hpp>
#include <uni20/operator/local_space.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace uni20
{

/// \brief Sparse local operator transforming as one symmetry irrep.
/// \details Conceptually this is the first concrete realization of
///          `Tensor<co<LocalSpace>, QNum, LocalSpace>`. The implementation stores
///          only a sparse coefficient matrix over explicit local states.
class LocalOperator
{
  public:
    using value_type = double;
    using matrix_type = SparseMatrix<value_type>;
    using index_type = typename matrix_type::index_type;

    /// \brief Construct an empty local operator with sparse coefficient storage.
    /// \param bra_space Output local space.
    /// \param ket_space Input local space.
    /// \param transforms_as Irrep under which the operator transforms.
    LocalOperator(LocalSpace bra_space, LocalSpace ket_space, QNum transforms_as)
        : bra_space_(std::move(bra_space)),
          ket_space_(std::move(ket_space)),
          transforms_as_(transforms_as),
          coefficients_(this->bra_space_.size(), this->ket_space_.size())
    {
        this->verify_symmetry();
    }

    /// \brief Construct a local operator from existing sparse coefficients.
    /// \param bra_space Output local space.
    /// \param ket_space Input local space.
    /// \param transforms_as Irrep under which the operator transforms.
    /// \param coefficients Sparse coefficient matrix with shape `(bra_space.size(), ket_space.size())`.
    LocalOperator(LocalSpace bra_space, LocalSpace ket_space, QNum transforms_as, matrix_type coefficients)
        : bra_space_(std::move(bra_space)),
          ket_space_(std::move(ket_space)),
          transforms_as_(transforms_as),
          coefficients_(std::move(coefficients))
    {
        this->verify_symmetry();
        if (this->coefficients_.rows() != this->bra_space_.size() || this->coefficients_.cols() != this->ket_space_.size())
        {
            throw std::invalid_argument("LocalOperator coefficient matrix shape does not match its spaces");
        }
    }

    /// \brief Return the output local space.
    /// \return Bra-space metadata.
    auto bra_space() const -> LocalSpace const& { return this->bra_space_; }

    /// \brief Return the input local space.
    /// \return Ket-space metadata.
    auto ket_space() const -> LocalSpace const& { return this->ket_space_; }

    /// \brief Return the symmetry irrep carried by this operator.
    /// \return Transform-as quantum number.
    auto transforms_as() const -> QNum const& { return this->transforms_as_; }

    /// \brief Return the common symmetry of the bra and ket spaces.
    /// \return Operator symmetry.
    auto symmetry() const -> Symmetry { return this->ket_space_.symmetry(); }

    /// \brief Return the number of bra-space states.
    /// \return Row count of the coefficient matrix.
    auto rows() const -> index_type { return this->coefficients_.rows(); }

    /// \brief Return the number of ket-space states.
    /// \return Column count of the coefficient matrix.
    auto cols() const -> index_type { return this->coefficients_.cols(); }

    /// \brief Return the number of stored sparse coefficients.
    /// \return Count of stored entries.
    auto nnz() const -> index_type { return this->coefficients_.nnz(); }

    /// \brief Return whether a coefficient is explicitly stored.
    /// \param row Bra-space state index.
    /// \param col Ket-space state index.
    /// \return `true` if a coefficient is stored at `(row, col)`.
    auto contains(index_type row, index_type col) const -> bool { return this->coefficients_.contains(row, col); }

    /// \brief Return the sparse coefficient matrix.
    /// \return Reference to the stored matrix.
    auto coefficients() const -> matrix_type const& { return this->coefficients_; }

    /// \brief Return the sparse coefficient matrix.
    /// \return Mutable reference to the stored matrix.
    auto coefficients() -> matrix_type& { return this->coefficients_; }

    /// \brief Return a stored coefficient.
    /// \throws std::out_of_range If the entry is absent.
    /// \param row Bra-space state index.
    /// \param col Ket-space state index.
    /// \return Stored coefficient at `(row, col)`.
    auto at(index_type row, index_type col) const -> value_type const& { return this->coefficients_.at(row, col); }

    /// \brief Insert or overwrite one sparse coefficient.
    /// \param row Bra-space state index.
    /// \param col Ket-space state index.
    /// \param value Coefficient value to store.
    /// \return Reference to the stored coefficient.
    auto insert_or_assign(index_type row, index_type col, value_type value) -> value_type&
    {
        return this->coefficients_.insert_or_assign(row, col, value);
    }

    /// \brief Remove one stored coefficient if present.
    /// \param row Bra-space state index.
    /// \param col Ket-space state index.
    /// \return `true` if a coefficient was removed.
    auto erase(index_type row, index_type col) -> bool { return this->coefficients_.erase(row, col); }

    /// \brief Remove all stored coefficients.
    void clear() { this->coefficients_.clear(); }

  private:
    /// \brief Validate that spaces and transform label share the same symmetry.
    void verify_symmetry() const
    {
        if (this->bra_space_.symmetry() != this->ket_space_.symmetry())
        {
            throw std::invalid_argument("LocalOperator bra and ket spaces must share one symmetry");
        }
        if (this->transforms_as_.symmetry() != this->ket_space_.symmetry())
        {
            throw std::invalid_argument("LocalOperator transform irrep must match the operator symmetry");
        }
    }

    LocalSpace bra_space_;
    LocalSpace ket_space_;
    QNum transforms_as_;
    matrix_type coefficients_;
};

} // namespace uni20
