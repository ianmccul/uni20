/**
 * \file u1.hpp
 * \brief Defines the first concrete symmetry factor, U(1), and its packed encoding helpers.
 */

#pragma once

#include <uni20/symmetry/symmetryfactor.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace uni20::detail
{

/// \brief Implements the packed irrep encoding rules for a U(1) symmetry factor.
class U1 final : public SymmetryFactor {
  public:
    /// \brief Packed representation type for one U(1) irrep.
    using code_type = std::uint64_t;

    /// \brief Native representation label type for U(1).
    using value_type = std::int64_t;

    /// \brief Encode a signed U(1) charge into canonical display order.
    /// \param value U(1) charge to encode.
    /// \return Encoded unsigned representation with `0` reserved for the identity.
    static constexpr auto encode(value_type value) -> code_type
    {
        if (value == 0)
        {
            return 0;
        }
        if (value > 0)
        {
            auto const magnitude = static_cast<code_type>(value);
            return magnitude * 2 - 1;
        }

        if (value == std::numeric_limits<value_type>::min())
        {
            throw std::overflow_error("U1::encode cannot represent the most negative int64_t value");
        }

        auto const magnitude = static_cast<code_type>(-value);
        return magnitude * 2;
    }

    /// \brief Decode a packed U(1) charge back to its signed value.
    /// \param code Packed U(1) code.
    /// \return Decoded signed charge.
    static constexpr auto decode(code_type code) -> value_type
    {
        if (code == 0)
        {
            return 0;
        }
        if ((code & 1U) != 0U)
        {
            return static_cast<value_type>((code + 1) / 2);
        }
        return -static_cast<value_type>(code / 2);
    }

    /// \brief Return the dual irrep of a U(1) charge.
    /// \param code Packed U(1) code.
    /// \return Packed code of the dual charge.
    static constexpr auto dual_code(code_type code) -> code_type { return encode(-decode(code)); }

    /// \brief Return the quantum dimension of a U(1) irrep.
    /// \param code Packed U(1) code.
    /// \return Quantum dimension, always `1.0` for U(1).
    static constexpr auto qdim_value(code_type code) -> double
    {
        static_cast<void>(code);
        return 1.0;
    }

    /// \brief Return the integral degree of a U(1) irrep.
    /// \param code Packed U(1) code.
    /// \return Integral degree, always `1` for U(1).
    static constexpr auto degree_value(code_type code) -> int
    {
        static_cast<void>(code);
        return 1;
    }

    /// \brief Format a packed U(1) code as a signed decimal string.
    /// \param code Packed U(1) code.
    /// \return Signed decimal representation of the decoded charge.
    static auto format(code_type code) -> std::string { return std::to_string(decode(code)); }

    /// \brief Return the canonical type name used in symmetry specifications.
    /// \return Canonical factor type string.
    auto type_name() const -> std::string_view override { return "U(1)"; }

    /// \brief Encode a signed U(1) charge into canonical display order.
    /// \param value Integer-valued U(1) charge.
    /// \return Packed factor-local code.
    auto encode_int64(std::int64_t value) const -> std::uint64_t override { return encode(value); }

    /// \brief Decode a packed factor-local U(1) code.
    /// \param code Packed factor-local code.
    /// \return Signed U(1) charge.
    auto decode_int64(std::uint64_t code) const -> std::int64_t override { return decode(code); }

    /// \brief Return the dual of a packed U(1) code.
    /// \param code Packed factor-local code.
    /// \return Packed factor-local code of the dual charge.
    auto dual(std::uint64_t code) const -> std::uint64_t override { return dual_code(code); }

    /// \brief Return the quantum dimension of a packed U(1) code.
    /// \param code Packed factor-local code.
    /// \return Quantum dimension, always `1.0`.
    auto qdim(std::uint64_t code) const -> double override { return qdim_value(code); }

    /// \brief Return the integral degree of a packed U(1) code.
    /// \param code Packed factor-local code.
    /// \return Integral degree, always `1`.
    auto degree(std::uint64_t code) const -> int override { return degree_value(code); }

    /// \brief Format a packed U(1) code as a signed decimal string.
    /// \param code Packed factor-local code.
    /// \return Signed decimal representation of the decoded charge.
    auto to_string(std::uint64_t code) const -> std::string override { return format(code); }
};

/// \brief Return the canonical U(1) symmetry factor instance.
/// \return Process-global U(1) factor implementation.
inline auto u1_factor() -> U1 const&
{
    static U1 factor;
    static auto const* registered = register_symmetry_factor(&factor);
    static_cast<void>(registered);
    return factor;
}

/// \brief Register the built-in symmetry factors needed by the first implementation.
inline void register_builtin_symmetry_factors() { static_cast<void>(u1_factor()); }

} // namespace uni20::detail
