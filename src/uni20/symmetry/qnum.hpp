/**
 * \file qnum.hpp
 * \brief Defines packed quantum numbers and same-symmetry quantum-number containers.
 *
 * \details See `docs/qnum.md` for the implemented `QNum` and `QNumList` API,
 *          along with the planned non-abelian, braiding, and coupling-data extensions.
 */

#pragma once

#include <uni20/symmetry/symmetry.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace uni20
{

/// \brief One irrep label together with the symmetry context needed to interpret it.
/// \details `QNum` is the packed runtime label used by the tensor code. See `docs/qnum.md`.
class QNum {
  public:
    /// \brief Construct an invalid quantum number.
    QNum() = default;

    /// \brief Construct from a symmetry handle and a packed code.
    /// \param sym Symmetry context.
    /// \param code Packed irrep code.
    explicit QNum(Symmetry sym, std::uint64_t code) : sym_(sym.impl()), code_(code) {}

    /// \brief Return whether the quantum number carries a valid symmetry context.
    /// \return `true` if this quantum number is initialized.
    auto valid() const -> bool { return sym_ != nullptr; }

    /// \brief Return the symmetry context of this quantum number.
    /// \return Symmetry handle corresponding to this quantum number.
    auto symmetry() const -> Symmetry
    {
        this->ensure_valid();
        return Symmetry{sym_};
    }

    /// \brief Return the packed irrep code.
    /// \return Canonical packed code.
    auto raw_code() const -> std::uint64_t
    {
        this->ensure_valid();
        return code_;
    }

    /// \brief Compare two quantum numbers for exact identity.
    /// \param other Other quantum number.
    /// \return `true` if both symmetry context and packed code match.
    auto operator==(QNum const& other) const -> bool = default;

    /// \brief Return the identity irrep of a given symmetry.
    /// \param sym Symmetry context.
    /// \return Identity quantum number.
    static auto identity(Symmetry sym) -> QNum { return QNum(sym, 0); }

  private:
    friend auto dual(QNum const&) -> QNum;
    friend auto is_scalar(QNum const&) -> bool;
    friend auto qdim(QNum const&) -> double;
    friend auto degree(QNum const&) -> int;
    friend auto to_string(QNum const&) -> std::string;
    friend auto make_qnum(Symmetry, std::initializer_list<std::pair<std::string_view, U1>>) -> QNum;
    friend auto coerce(QNum const&, Symmetry) -> QNum;
    friend auto u1_component(QNum const&, std::string_view) -> U1;
    friend auto operator+(QNum const&, QNum const&) -> QNum;

    /// \brief Throw if this quantum number is invalid.
    void ensure_valid() const
    {
        if (!this->valid())
        {
            throw std::logic_error("QNum is not initialized");
        }
    }

    detail::SymmetryImpl const* sym_ = nullptr;
    std::uint64_t code_ = 0;
};

/// \brief Explicit sparse tensor space represented as quantum numbers with one shared symmetry.
/// \details `QNumList` keeps ordering and repeated labels explicit. See `docs/qnum.md`.
class QNumList {
  public:
    /// \brief Construct an empty list tied to one symmetry.
    /// \param sym Symmetry carried by every element of the list.
    explicit QNumList(Symmetry sym) : sym_(sym) {}

    /// \brief Construct a list from initial values.
    /// \param sym Symmetry carried by every element.
    /// \param values Initial list contents.
    QNumList(Symmetry sym, std::initializer_list<QNum> values) : sym_(sym), values_(values)
    {
        for (QNum const& q : values_)
        {
            this->verify_symmetry(q);
        }
    }

    /// \brief Return the symmetry shared by all entries.
    /// \return List symmetry.
    auto symmetry() const -> Symmetry { return sym_; }

    /// \brief Return the number of stored quantum numbers.
    /// \return Element count.
    auto size() const -> std::size_t { return values_.size(); }

    /// \brief Return whether the list is empty.
    /// \return `true` if the list contains no entries.
    auto empty() const -> bool { return values_.empty(); }

    /// \brief Append a quantum number with the matching symmetry.
    /// \param q Quantum number to append.
    void push_back(QNum q)
    {
        this->verify_symmetry(q);
        values_.push_back(q);
    }

    /// \brief Remove all quantum numbers while keeping the list symmetry.
    void clear() { values_.clear(); }

    /// \brief Return whether the list already contains the given quantum number.
    /// \param q Quantum number to search for.
    /// \return `true` if the quantum number is present.
    auto contains(QNum q) const -> bool
    {
        this->verify_symmetry(q);
        return std::find(values_.begin(), values_.end(), q) != values_.end();
    }

    /// \brief Sort entries by canonical packed order.
    void sort()
    {
        std::sort(values_.begin(), values_.end(),
                  [](QNum const& lhs, QNum const& rhs) { return lhs.raw_code() < rhs.raw_code(); });
    }

    /// \brief Sort entries and remove duplicates.
    void normalize()
    {
        this->sort();
        values_.erase(std::unique(values_.begin(), values_.end()), values_.end());
    }

    /// \brief Return indexed element access.
    /// \param index Zero-based index.
    /// \return Quantum number at the requested position.
    auto operator[](std::size_t index) const -> QNum const& { return values_[index]; }

    /// \brief Return an iterator to the first element.
    /// \return Begin iterator.
    auto begin() const { return values_.begin(); }

    /// \brief Return an iterator past the last element.
    /// \return End iterator.
    auto end() const { return values_.end(); }

    /// \brief Compare two sparse irrep lists for exact equality.
    /// \param other Other sparse irrep list.
    /// \return `true` if symmetry and ordered values match.
    auto operator==(QNumList const& other) const -> bool = default;

  private:
    /// \brief Check that a quantum number matches the list symmetry.
    /// \param q Quantum number to validate.
    void verify_symmetry(QNum const& q) const
    {
        if (q.symmetry() != sym_)
        {
            throw std::invalid_argument("QNumList element has the wrong symmetry");
        }
    }

    Symmetry sym_;
    std::vector<QNum> values_;
};

/// \brief Construct a packed `QNum` from named U(1) component values.
/// \param sym Target symmetry.
/// \param values Named component assignments. Missing components default to the identity.
/// \return Packed quantum number in the requested symmetry.
inline auto make_qnum(Symmetry sym, std::initializer_list<std::pair<std::string_view, U1>> values) -> QNum
{
    auto const* impl = sym.impl();
    std::vector<std::uint64_t> codes(impl->factor_count(), 0);
    std::vector<bool> seen(impl->factor_count(), false);

    for (auto const& [name, value] : values)
    {
        auto const index = impl->find_factor(name);
        if (!index.has_value())
        {
            throw std::invalid_argument("unknown symmetry component name: " + std::string{name});
        }
        if (seen[*index])
        {
            throw std::invalid_argument("duplicate symmetry component assignment: " + std::string{name});
        }
        seen[*index] = true;

        auto const& factor = impl->factors()[*index];
        auto const* typed_factor = dynamic_cast<detail::SymmetryFactor<U1> const*>(factor.factor);
        if (typed_factor == nullptr)
        {
            throw std::invalid_argument("make_qnum(U1) used with a non-U(1) symmetry component");
        }
        codes[*index] = typed_factor->encode(value);
    }

    return QNum(sym, impl->pack(codes));
}

/// \brief Return the dual irrep of a quantum number.
/// \param q Quantum number to dualize.
/// \return Dual quantum number in the same symmetry.
inline auto dual(QNum const& q) -> QNum
{
    auto const* impl = q.sym_;
    auto codes = impl->unpack(q.code_);
    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        codes[i] = impl->factors()[i].factor->dual_code(codes[i]);
    }
    return QNum(q.symmetry(), impl->pack(codes));
}

/// \brief Return whether a quantum number is the identity irrep.
/// \param q Quantum number to test.
/// \return `true` if every factor is the identity.
inline auto is_scalar(QNum const& q) -> bool { return q.raw_code() == 0; }

/// \brief Return the quantum dimension of a quantum number.
/// \param q Quantum number to inspect.
/// \return Product of factor quantum dimensions.
inline auto qdim(QNum const& q) -> double
{
    auto const* impl = q.sym_;
    auto const codes = impl->unpack(q.code_);
    double result = 1.0;
    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        result *= impl->factors()[i].factor->qdim_code(codes[i]);
    }
    return result;
}

/// \brief Return the integral degree of a quantum number.
/// \param q Quantum number to inspect.
/// \return Product of factor degrees.
inline auto degree(QNum const& q) -> int
{
    auto const* impl = q.sym_;
    auto const codes = impl->unpack(q.code_);
    int result = 1;
    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        result *= impl->factors()[i].factor->degree_code(codes[i]);
    }
    return result;
}

/// \brief Read one named U(1) component from a quantum number.
/// \param q Quantum number to decode.
/// \param name Name of the requested symmetry component.
/// \return U(1) charge for that component.
inline auto u1_component(QNum const& q, std::string_view name) -> U1
{
    auto const* impl = q.sym_;
    auto const index = impl->find_factor(name);
    if (!index.has_value())
    {
        throw std::invalid_argument("unknown symmetry component name: " + std::string{name});
    }

    auto const& factor = impl->factors()[*index];
    auto const* typed_factor = dynamic_cast<detail::SymmetryFactor<U1> const*>(factor.factor);
    if (typed_factor == nullptr)
    {
        throw std::invalid_argument("requested U(1) component from a non-U(1) factor");
    }

    auto const codes = impl->unpack(q.code_);
    return typed_factor->decode(codes[*index]);
}

/// \brief Coerce a quantum number into a related symmetry by named components.
/// \param q Source quantum number.
/// \param target Target symmetry.
/// \return Coerced quantum number in the target symmetry.
inline auto coerce(QNum const& q, Symmetry target) -> QNum
{
    auto const source = q.symmetry();
    if (source == target)
    {
        return q;
    }

    auto const* source_impl = source.impl();
    auto const* target_impl = target.impl();
    auto const source_codes = source_impl->unpack(q.code_);

    std::vector<std::uint64_t> target_codes(target_impl->factor_count(), 0);
    for (std::size_t i = 0; i < target_impl->factor_count(); ++i)
    {
        auto const& target_factor = target_impl->factors()[i];
        auto const source_index = source_impl->find_factor(target_factor.name);
        if (!source_index.has_value())
        {
            target_codes[i] = 0;
            continue;
        }

        auto const& source_factor = source_impl->factors()[*source_index];
        if (source_factor.factor != target_factor.factor)
        {
            throw std::invalid_argument("cannot coerce between differently typed symmetry components");
        }
        target_codes[i] = source_codes[*source_index];
    }

    for (std::size_t i = 0; i < source_impl->factor_count(); ++i)
    {
        auto const& source_factor = source_impl->factors()[i];
        auto const target_index = target_impl->find_factor(source_factor.name);
        if (!target_index.has_value() && source_codes[i] != 0)
        {
            throw std::invalid_argument("cannot drop a non-identity symmetry component during coercion");
        }
    }

    return QNum(target, target_impl->pack(target_codes));
}

/// \brief Add two quantum numbers when the fusion result is uniquely defined.
/// \param lhs Left operand.
/// \param rhs Right operand.
/// \return Sum in the shared symmetry context.
inline auto operator+(QNum const& lhs, QNum const& rhs) -> QNum
{
    if (lhs.symmetry() != rhs.symmetry())
    {
        throw std::invalid_argument("QNum operator+ requires identical symmetry contexts");
    }

    auto const* impl = lhs.sym_;
    auto const lhs_codes = impl->unpack(lhs.code_);
    auto const rhs_codes = impl->unpack(rhs.code_);
    std::vector<std::uint64_t> out_codes(impl->factor_count(), 0);

    for (std::size_t i = 0; i < out_codes.size(); ++i)
    {
        auto const* factor = impl->factors()[i].factor;
        out_codes[i] = factor->add_codes(lhs_codes[i], rhs_codes[i]);
    }

    return QNum(lhs.symmetry(), impl->pack(out_codes));
}

/// \brief Format a quantum number as comma-separated named components.
/// \param q Quantum number to format.
/// \return Human-readable string form.
inline auto to_string(QNum const& q) -> std::string
{
    auto const* impl = q.sym_;
    auto const codes = impl->unpack(q.code_);
    std::string result;
    for (std::size_t i = 0; i < codes.size(); ++i)
    {
        if (!result.empty())
        {
            result += ',';
        }
        result += impl->factors()[i].name;
        result += '=';
        result += impl->factors()[i].factor->format_code(codes[i]);
    }
    return result;
}

} // namespace uni20

namespace std
{
template <> struct hash<uni20::QNum>
{
    /// \brief Hash a packed quantum number by its symmetry context and raw code.
    /// \param q Quantum number to hash.
    /// \return Combined hash value.
    auto operator()(uni20::QNum const& q) const noexcept -> std::size_t
    {
        auto const sym_hash = std::hash<void const*>{}(q.valid() ? q.symmetry().impl() : nullptr);
        auto const code_hash = std::hash<std::uint64_t>{}(q.valid() ? q.raw_code() : 0);
        return sym_hash ^ (code_hash + 0x9e3779b97f4a7c15ULL + (sym_hash << 6) + (sym_hash >> 2));
    }
};
} // namespace std
