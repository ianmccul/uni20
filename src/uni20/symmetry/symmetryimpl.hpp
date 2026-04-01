/**
 * \file symmetryimpl.hpp
 * \brief Defines the internal canonicalized representation of direct-product symmetries.
 */

#pragma once

#include <uni20/common/trace.hpp>
#include <uni20/symmetry/u1.hpp>

#include <cctype>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace uni20::detail
{

/// \brief Describe one named factor of a direct-product symmetry.
struct SymmetryFactorSpec
{
    std::string name;
    SymmetryFactor const* factor = nullptr;
};

/// \brief Internal canonicalized description of a direct-product symmetry.
///
/// \internal
/// The first implementation supports only named U(1) factors. Canonicalization is
/// process-local and intentionally leaks the interned `SymmetryImpl` instances.
/// \endinternal
class SymmetryImpl {
  public:
    /// \brief Construct a symmetry implementation from factor specifications.
    /// \param factors Named direct-product factors in canonical order.
    explicit SymmetryImpl(std::vector<SymmetryFactorSpec> factors) : factors_(std::move(factors))
    {
        for (std::size_t i = 0; i < factors_.size(); ++i)
        {
            auto const [it, inserted] = index_by_name_.emplace(factors_[i].name, i);
            if (!inserted)
            {
                throw std::invalid_argument("duplicate symmetry component name: " + it->first);
            }
        }
        canonical_string_ = build_canonical_string(factors_);
    }

    /// \brief Return the factor list in positional order.
    /// \return Read-only view of the factor specifications.
    auto factors() const -> std::span<SymmetryFactorSpec const> { return factors_; }

    /// \brief Return the number of direct-product factors.
    /// \return Number of named factors.
    auto factor_count() const -> std::size_t { return factors_.size(); }

    /// \brief Look up the position of a named factor.
    /// \param name Component name to search for.
    /// \return Zero-based factor index if present.
    auto find_factor(std::string_view name) const -> std::optional<std::size_t>
    {
        auto const it = index_by_name_.find(std::string{name});
        if (it == index_by_name_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    /// \brief Return the canonical string form of this symmetry.
    /// \return Canonical string representation.
    auto to_string() const -> std::string const& { return canonical_string_; }

    /// \brief Intern a parsed direct-product symmetry.
    /// \param factors Named factor list in canonical order.
    /// \return Canonical process-global `SymmetryImpl` instance.
    static auto intern(std::vector<SymmetryFactorSpec> factors) -> SymmetryImpl const*
    {
        auto const key = build_canonical_string(factors);

        auto& mutex = intern_mutex();
        std::lock_guard<std::mutex> lock(mutex);

        auto& map = intern_map();
        if (auto const it = map.find(key); it != map.end())
        {
            return it->second;
        }

        auto* impl = new SymmetryImpl(std::move(factors));
        map.emplace(impl->to_string(), impl);
        return impl;
    }

    /// \brief Parse a string specification and return the canonical symmetry instance.
    /// \param spec Comma-separated list such as `"N:U(1),Sz:U(1)"`.
    /// \return Canonical process-global `SymmetryImpl` instance.
    static auto parse(std::string_view spec) -> SymmetryImpl const*
    {
        register_builtin_symmetry_factors();

        auto trimmed = trim(spec);
        if (trimmed.empty())
        {
            throw std::invalid_argument("symmetry specification must not be empty");
        }

        std::vector<SymmetryFactorSpec> factors;
        std::size_t cursor = 0;
        while (cursor < trimmed.size())
        {
            auto const next = trimmed.find(',', cursor);
            auto const token =
                trim(trimmed.substr(cursor, next == std::string_view::npos ? std::string_view::npos : next - cursor));
            if (token.empty())
            {
                throw std::invalid_argument("symmetry specification contains an empty factor");
            }

            auto const colon = token.find(':');
            if (colon == std::string_view::npos)
            {
                throw std::invalid_argument("symmetry factor is missing ':' separator: " + std::string{token});
            }

            auto const name = trim(token.substr(0, colon));
            auto const kind = trim(token.substr(colon + 1));
            if (name.empty())
            {
                throw std::invalid_argument("symmetry factor has an empty name");
            }
            if (kind.empty())
            {
                throw std::invalid_argument("symmetry factor has an empty type");
            }

            factors.push_back({std::string{name}, find_symmetry_factor(kind)});

            if (next == std::string_view::npos)
            {
                break;
            }
            cursor = next + 1;
        }

        return intern(std::move(factors));
    }

    /// \brief Pack factor-local irrep codes into one `uint64_t`.
    /// \param factor_codes One packed local code per direct-product factor.
    /// \return Combined packed `QNum` code.
    auto pack(std::span<std::uint64_t const> factor_codes) const -> std::uint64_t
    {
        if (factor_codes.size() != factors_.size())
        {
            throw std::invalid_argument("wrong number of factor codes for symmetry packing");
        }
        if (factor_codes.empty())
        {
            return 0;
        }

        std::uint64_t combined = factor_codes.front();
        for (std::size_t i = 1; i < factor_codes.size(); ++i)
        {
            combined = cantor_pair(combined, factor_codes[i]);
        }
        return combined;
    }

    /// \brief Unpack a combined code into per-factor local irrep codes.
    /// \param code Combined packed `QNum` code.
    /// \return One packed local code per direct-product factor.
    auto unpack(std::uint64_t code) const -> std::vector<std::uint64_t>
    {
        if (factors_.empty())
        {
            return {};
        }
        if (factors_.size() == 1)
        {
            return {code};
        }

        std::vector<std::uint64_t> codes(factors_.size());
        auto current = code;
        for (std::size_t i = factors_.size(); i-- > 1;)
        {
            auto const [left, right] = cantor_unpair(current);
            codes[i] = right;
            current = left;
        }
        codes[0] = current;
        return codes;
    }

  private:
    /// \brief Trim leading and trailing ASCII whitespace from a string view.
    /// \param text Input text view.
    /// \return Trimmed view into the original string.
    static auto trim(std::string_view text) -> std::string_view
    {
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0)
        {
            text.remove_prefix(1);
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0)
        {
            text.remove_suffix(1);
        }
        return text;
    }

    /// \brief Build the canonical string representation from the factor list.
    /// \param factors Named factor list.
    /// \return Canonical comma-separated string form.
    static auto build_canonical_string(std::span<SymmetryFactorSpec const> factors) -> std::string
    {
        std::ostringstream out;
        for (std::size_t i = 0; i < factors.size(); ++i)
        {
            if (i != 0)
            {
                out << ',';
            }
            out << factors[i].name << ':' << factors[i].factor->type_name();
        }
        return out.str();
    }

    /// \brief Return the global intern map.
    /// \return Process-global mapping from canonical strings to canonical instances.
    static auto intern_map() -> std::unordered_map<std::string, SymmetryImpl const*>&
    {
        static std::unordered_map<std::string, SymmetryImpl const*> map;
        return map;
    }

    /// \brief Return the mutex protecting the global intern map.
    /// \return Process-global mutex for canonicalization.
    static auto intern_mutex() -> std::mutex&
    {
        static std::mutex mutex;
        return mutex;
    }

    /// \brief Return whether the triangular number `value * (value + 1) / 2` fits in `limit`.
    /// \param value Triangular index to test.
    /// \param limit Upper bound to compare against.
    /// \return `true` if the triangular number is less than or equal to `limit`.
    static auto triangular_leq(std::uint64_t value, std::uint64_t limit) -> bool
    {
        auto const left = (value % 2U == 0U) ? value / 2 : value / 2 + 1;
        auto const right = (value % 2U == 0U) ? value + 1 : value;
        if (left == 0)
        {
            return true;
        }
        return left <= limit / right;
    }

    /// \brief Compute a triangular number exactly in `uint64_t`.
    /// \param value Triangular index.
    /// \return `value * (value + 1) / 2`.
    static auto triangular(std::uint64_t value) -> std::uint64_t
    {
        auto const left = (value % 2U == 0U) ? value / 2 : value / 2 + 1;
        auto const right = (value % 2U == 0U) ? value + 1 : value;
        return left * right;
    }

    /// \brief Cantor-pair two packed factor codes.
    /// \param left Left input.
    /// \param right Right input.
    /// \return Combined packed code.
    static auto cantor_pair(std::uint64_t left, std::uint64_t right) -> std::uint64_t
    {
        if (left > std::numeric_limits<std::uint64_t>::max() - right)
        {
            throw std::overflow_error("quantum number code overflow while pairing direct-product components");
        }

        auto const sum = left + right;
        if (!triangular_leq(sum, std::numeric_limits<std::uint64_t>::max() - right))
        {
            throw std::overflow_error("quantum number code overflow while pairing direct-product components");
        }
        return triangular(sum) + right;
    }

    /// \brief Invert the Cantor pairing function.
    /// \param code Paired code.
    /// \return Original left and right codes.
    static auto cantor_unpair(std::uint64_t code) -> std::pair<std::uint64_t, std::uint64_t>
    {
        std::uint64_t lo = 0;
        std::uint64_t hi = code;
        while (lo < hi)
        {
            auto const mid = lo + (hi - lo + 1) / 2;
            if (triangular_leq(mid, code))
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        auto const sum = lo;
        auto const triangle = triangular(sum);
        auto const right = code - triangle;
        auto const left = sum - right;
        return {left, right};
    }

    std::vector<SymmetryFactorSpec> factors_;
    std::unordered_map<std::string, std::size_t> index_by_name_;
    std::string canonical_string_;
};

} // namespace uni20::detail
