/**
 * \file sparse_matrix.hpp
 * \brief Row-oriented sparse matrix container used by operator and MPO infrastructure.
 * \details See `docs/matrix.md` for the current scope and planned use.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief Row-oriented sparse matrix with sorted entries inside each row.
/// \details The matrix stores each row as a sorted list of `(column, value)` entries.
///          This keeps construction and mutation simple while preserving the access
///          pattern needed for sparse local operators and MPO rows.
/// \tparam T Value type stored in each nonzero entry.
template <typename T>
class SparseMatrix
{
  public:
    using value_type = T;
    using index_type = std::size_t;

    /// \brief Sparse matrix entry inside a row.
    struct Entry
    {
        index_type column{};
        value_type value;

        friend auto operator==(Entry const&, Entry const&) -> bool = default;
    };

    /// \brief Construct an empty sparse matrix with the requested shape.
    /// \param rows Number of matrix rows.
    /// \param cols Number of matrix columns.
    explicit SparseMatrix(index_type rows, index_type cols) : rows_(rows), cols_(cols), rows_data_(rows) {}

    /// \brief Return the number of matrix rows.
    /// \return Row count.
    [[nodiscard]] auto rows() const noexcept -> index_type { return this->rows_; }

    /// \brief Return the number of matrix columns.
    /// \return Column count.
    [[nodiscard]] auto cols() const noexcept -> index_type { return this->cols_; }

    /// \brief Return the matrix shape.
    /// \return Pair `(rows, cols)`.
    [[nodiscard]] auto shape() const noexcept -> std::pair<index_type, index_type> { return {this->rows_, this->cols_}; }

    /// \brief Return whether the matrix has zero rows or zero columns.
    /// \return `true` if at least one dimension is zero.
    [[nodiscard]] auto empty() const noexcept -> bool { return this->rows_ == 0 || this->cols_ == 0; }

    /// \brief Return the number of stored entries.
    /// \return Total count of nonzero slots currently stored.
    [[nodiscard]] auto nnz() const noexcept -> index_type { return this->nnz_; }

    /// \brief Return the number of stored entries in a row.
    /// \param row Row index.
    /// \return Number of stored entries in the requested row.
    [[nodiscard]] auto row_size(index_type row) const -> index_type { return this->row_ref(row).size(); }

    /// \brief Return a read-only view of a stored row.
    /// \param row Row index.
    /// \return Span of sorted entries for the requested row.
    [[nodiscard]] auto row(index_type row) const -> std::span<Entry const> { return this->row_ref(row); }

    /// \brief Reserve capacity for one row.
    /// \param row Row index.
    /// \param capacity Desired row capacity.
    void reserve_row(index_type row, index_type capacity) { this->row_ref(row).reserve(capacity); }

    /// \brief Return whether an entry exists at `(row, col)`.
    /// \param row Row index.
    /// \param col Column index.
    /// \return `true` if the entry is present.
    [[nodiscard]] auto contains(index_type row, index_type col) const -> bool
    {
        auto const& row_data = this->row_ref(row);
        auto const it = this->lower_bound(row_data, col);
        return it != row_data.end() && it->column == col;
    }

    /// \brief Return a pointer to the stored value at `(row, col)` if present.
    /// \param row Row index.
    /// \param col Column index.
    /// \return Pointer to the stored value, or `nullptr` if the entry is absent.
    [[nodiscard]] auto find(index_type row, index_type col) -> value_type*
    {
        auto& row_data = this->row_ref(row);
        auto const it = this->lower_bound(row_data, col);
        if (it == row_data.end() || it->column != col)
        {
            return nullptr;
        }
        return &it->value;
    }

    /// \brief Return a pointer to the stored value at `(row, col)` if present.
    /// \param row Row index.
    /// \param col Column index.
    /// \return Pointer to the stored value, or `nullptr` if the entry is absent.
    [[nodiscard]] auto find(index_type row, index_type col) const -> value_type const*
    {
        this->check_column(col);
        auto const& row_data = this->row_ref(row);
        auto const it = this->lower_bound(row_data, col);
        if (it == row_data.end() || it->column != col)
        {
            return nullptr;
        }
        return &it->value;
    }

    /// \brief Return the stored value at `(row, col)`.
    /// \throws std::out_of_range If the row or column index is invalid or the entry is absent.
    /// \param row Row index.
    /// \param col Column index.
    /// \return Reference to the stored value.
    auto at(index_type row, index_type col) -> value_type&
    {
        auto* ptr = this->find(row, col);
        if (ptr == nullptr)
        {
            throw std::out_of_range("SparseMatrix::at: entry not present");
        }
        return *ptr;
    }

    /// \brief Return the stored value at `(row, col)`.
    /// \throws std::out_of_range If the row or column index is invalid or the entry is absent.
    /// \param row Row index.
    /// \param col Column index.
    /// \return Reference to the stored value.
    auto at(index_type row, index_type col) const -> value_type const&
    {
        auto const* ptr = this->find(row, col);
        if (ptr == nullptr)
        {
            throw std::out_of_range("SparseMatrix::at: entry not present");
        }
        return *ptr;
    }

    /// \brief Insert a new entry or overwrite an existing one.
    /// \param row Row index.
    /// \param col Column index.
    /// \param value Value to store.
    /// \return Reference to the stored value.
    auto insert_or_assign(index_type row, index_type col, value_type value) -> value_type&
    {
        this->check_column(col);
        auto& row_data = this->row_ref(row);
        auto const it = this->lower_bound(row_data, col);
        if (it != row_data.end() && it->column == col)
        {
            it->value = std::move(value);
            return it->value;
        }

        auto const inserted = row_data.insert(it, Entry{col, std::move(value)});
        ++this->nnz_;
        return inserted->value;
    }

    /// \brief Remove an entry if it is present.
    /// \param row Row index.
    /// \param col Column index.
    /// \return `true` if an entry was removed.
    auto erase(index_type row, index_type col) -> bool
    {
        this->check_column(col);
        auto& row_data = this->row_ref(row);
        auto const it = this->lower_bound(row_data, col);
        if (it == row_data.end() || it->column != col)
        {
            return false;
        }

        row_data.erase(it);
        --this->nnz_;
        return true;
    }

    /// \brief Remove all stored entries.
    void clear() noexcept
    {
        for (auto& row_data : this->rows_data_)
        {
            row_data.clear();
        }
        this->nnz_ = 0;
    }

    /// \brief Remove all stored entries from one row.
    /// \param row Row index.
    void clear_row(index_type row)
    {
        auto& row_data = this->row_ref(row);
        this->nnz_ -= row_data.size();
        row_data.clear();
    }

    /// \brief Return the transpose of the sparse matrix.
    /// \return New sparse matrix with rows and columns exchanged.
    [[nodiscard]] auto transpose() const -> SparseMatrix<value_type>
    {
        SparseMatrix<value_type> result(this->cols_, this->rows_);
        for (index_type row = 0; row < this->rows_; ++row)
        {
            for (auto const& entry : this->rows_data_[row])
            {
                result.insert_or_assign(entry.column, row, entry.value);
            }
        }
        return result;
    }

  private:
    using row_type = std::vector<Entry>;

    template <typename Row>
    [[nodiscard]] static auto lower_bound(Row& row_data, index_type col)
    {
        return std::ranges::lower_bound(row_data, col, {}, &Entry::column);
    }

    void check_column(index_type col) const
    {
        if (col >= this->cols_)
        {
            throw std::out_of_range("SparseMatrix: column index out of range");
        }
    }

    [[nodiscard]] auto row_ref(index_type row) -> row_type&
    {
        if (row >= this->rows_)
        {
            throw std::out_of_range("SparseMatrix: row index out of range");
        }
        return this->rows_data_[row];
    }

    [[nodiscard]] auto row_ref(index_type row) const -> row_type const&
    {
        if (row >= this->rows_)
        {
            throw std::out_of_range("SparseMatrix: row index out of range");
        }
        return this->rows_data_[row];
    }

    index_type rows_{};
    index_type cols_{};
    index_type nnz_{};
    std::vector<row_type> rows_data_{};
};

} // namespace uni20
