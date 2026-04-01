/**
 * \file symmetryfactor.hpp
 * \brief Defines the internal runtime interface for concrete symmetry-factor adapters.
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace uni20::detail
{

/// \brief Runtime-erased interface for a concrete symmetry factor.
class SymmetryFactorBase {
  public:
    virtual ~SymmetryFactorBase() = default;

    /// \brief Return the canonical type name used in symmetry specifications.
    /// \return Canonical factor type string.
    virtual auto type_name() const -> std::string_view = 0;

    /// \brief Return the dual irrep of a packed factor-local code.
    /// \param code Packed factor-local code.
    /// \return Packed factor-local code of the dual irrep.
    virtual auto dual_code(std::uint64_t code) const -> std::uint64_t = 0;

    /// \brief Return the quantum dimension of a packed factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Quantum dimension.
    virtual auto qdim_code(std::uint64_t code) const -> double = 0;

    /// \brief Return the integral degree of a packed factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Integral degree.
    virtual auto degree_code(std::uint64_t code) const -> int = 0;

    /// \brief Add two packed factor-local codes using the unique-fusion shortcut.
    /// \param lhs Left packed factor-local code.
    /// \param rhs Right packed factor-local code.
    /// \return Packed factor-local code of the unique sum.
    virtual auto add_codes(std::uint64_t lhs, std::uint64_t rhs) const -> std::uint64_t = 0;

    /// \brief Format a packed factor-local irrep code.
    /// \param code Packed factor-local code.
    /// \return Human-readable string form.
    virtual auto format_code(std::uint64_t code) const -> std::string = 0;
};

/// \brief Traits that teach `SymmetryFactor<T>` how to encode and decode one concrete factor type.
/// \tparam Q Concrete factor value type.
template <typename Q> struct SymmetryFactorTraits;

/// \brief Typed adapter that forwards runtime-erased operations to free functions on `Q`.
/// \tparam Q Concrete factor value type.
template <typename Q> class SymmetryFactor final : public SymmetryFactorBase {
  public:
    /// \brief Encode a concrete factor value into its packed factor-local representation.
    /// \param value Concrete factor value.
    /// \return Packed factor-local code.
    auto encode(Q const& value) const -> std::uint64_t { return SymmetryFactorTraits<Q>::encode(value); }

    /// \brief Decode a packed factor-local representation into a concrete factor value.
    /// \param code Packed factor-local code.
    /// \return Concrete factor value.
    auto decode(std::uint64_t code) const -> Q { return SymmetryFactorTraits<Q>::decode(code); }

    /// \brief Return the canonical type name used in symmetry specifications.
    /// \return Canonical factor type string.
    auto type_name() const -> std::string_view override { return SymmetryFactorTraits<Q>::type_name(); }

    /// \brief Return the dual irrep of a packed factor-local code.
    /// \param code Packed factor-local code.
    /// \return Packed factor-local code of the dual irrep.
    auto dual_code(std::uint64_t code) const -> std::uint64_t override
    {
        return this->encode(dual(this->decode(code)));
    }

    /// \brief Return the quantum dimension of a packed factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Quantum dimension.
    auto qdim_code(std::uint64_t code) const -> double override { return qdim(this->decode(code)); }

    /// \brief Return the integral degree of a packed factor-local irrep.
    /// \param code Packed factor-local code.
    /// \return Integral degree.
    auto degree_code(std::uint64_t code) const -> int override { return degree(this->decode(code)); }

    /// \brief Add two packed factor-local codes using the unique-fusion shortcut.
    /// \param lhs Left packed factor-local code.
    /// \param rhs Right packed factor-local code.
    /// \return Packed factor-local code of the unique sum.
    auto add_codes(std::uint64_t lhs, std::uint64_t rhs) const -> std::uint64_t override
    {
        return this->encode(this->decode(lhs) + this->decode(rhs));
    }

    /// \brief Format a packed factor-local irrep code.
    /// \param code Packed factor-local code.
    /// \return Human-readable string form.
    auto format_code(std::uint64_t code) const -> std::string override { return to_string(this->decode(code)); }
};

/// \brief Return the process-local symmetry factor registry.
/// \return Mapping from canonical factor type names to concrete factor adapters.
inline auto symmetry_factor_registry() -> std::unordered_map<std::string, SymmetryFactorBase const*>&
{
    static std::unordered_map<std::string, SymmetryFactorBase const*> registry;
    return registry;
}

/// \brief Register a concrete symmetry factor adapter in the process-local registry.
/// \param factor Concrete factor adapter.
/// \return The registered factor pointer.
inline auto register_symmetry_factor(SymmetryFactorBase const* factor) -> SymmetryFactorBase const*
{
    symmetry_factor_registry().emplace(std::string{factor->type_name()}, factor);
    return factor;
}

/// \brief Look up a registered symmetry factor by its canonical type name.
/// \param type_name Canonical factor type string.
/// \return Registered factor adapter.
inline auto find_symmetry_factor(std::string_view type_name) -> SymmetryFactorBase const*
{
    auto const it = symmetry_factor_registry().find(std::string{type_name});
    if (it == symmetry_factor_registry().end())
    {
        throw std::invalid_argument("unsupported symmetry factor type: " + std::string{type_name});
    }
    return it->second;
}

} // namespace uni20::detail
