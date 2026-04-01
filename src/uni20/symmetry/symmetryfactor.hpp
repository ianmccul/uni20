/**
 * \file symmetryfactor.hpp
 * \brief Defines the internal virtual interface for concrete symmetry factors.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace uni20::detail
{

/// \brief Virtual interface implemented by each concrete symmetry factor.
class SymmetryFactor {
  public:
    virtual ~SymmetryFactor() = default;

    /// \brief Return the canonical type name used in symmetry specifications.
    /// \return Canonical factor type string.
    virtual auto type_name() const -> std::string_view = 0;

    /// \brief Encode a first-pass signed integer irrep label.
    /// \param value Integer-valued irrep label.
    /// \return Packed factor-local code.
    virtual auto encode_int64(std::int64_t value) const -> std::uint64_t = 0;

    /// \brief Decode a first-pass signed integer irrep label.
    /// \param code Packed factor-local code.
    /// \return Decoded integer-valued irrep label.
    virtual auto decode_int64(std::uint64_t code) const -> std::int64_t = 0;

    /// \brief Return the dual irrep of a factor-local code.
    /// \param code Packed factor-local code.
    /// \return Packed factor-local code of the dual irrep.
    virtual auto dual(std::uint64_t code) const -> std::uint64_t = 0;

    /// \brief Return the quantum dimension of a factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Quantum dimension.
    virtual auto qdim(std::uint64_t code) const -> double = 0;

    /// \brief Return the integral degree of a factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Integral degree.
    virtual auto degree(std::uint64_t code) const -> int = 0;

    /// \brief Format a factor-local irrep code.
    /// \param code Packed factor-local code.
    /// \return Human-readable string form.
    virtual auto to_string(std::uint64_t code) const -> std::string = 0;
};

/// \brief Return the process-local symmetry factor registry.
/// \return Mapping from canonical factor type names to concrete factor implementations.
inline auto symmetry_factor_registry() -> std::unordered_map<std::string, SymmetryFactor const*>&
{
    static std::unordered_map<std::string, SymmetryFactor const*> registry;
    return registry;
}

/// \brief Register a concrete symmetry factor type in the process-local registry.
/// \param factor Concrete factor implementation.
/// \return The registered factor pointer.
inline auto register_symmetry_factor(SymmetryFactor const* factor) -> SymmetryFactor const*
{
    symmetry_factor_registry().emplace(std::string{factor->type_name()}, factor);
    return factor;
}

/// \brief Look up a registered symmetry factor by its canonical type name.
/// \param type_name Canonical factor type string.
/// \return Registered factor implementation.
inline auto find_symmetry_factor(std::string_view type_name) -> SymmetryFactor const*
{
    auto const it = symmetry_factor_registry().find(std::string{type_name});
    if (it == symmetry_factor_registry().end())
    {
        throw std::invalid_argument("unsupported symmetry factor type: " + std::string{type_name});
    }
    return it->second;
}

} // namespace uni20::detail
