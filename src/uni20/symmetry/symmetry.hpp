/**
 * \file symmetry.hpp
 * \brief Defines the lightweight public handle for canonicalized direct-product symmetries.
 *
 * \details See `docs/qnum.md` for the current user-facing symmetry and quantum-number API and
 *          the planned extension points for non-abelian and braided cases.
 */

#pragma once

#include <uni20/symmetry/symmetryimpl.hpp>

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace uni20
{

/// \brief Public handle to a canonicalized direct-product symmetry specification.
/// \details The intended frontend construction path is string-based, for example
///          `"N:U(1),Sz:U(1)"`. See `docs/qnum.md`.
class Symmetry {
  public:
    /// \brief Construct an invalid symmetry handle.
    Symmetry() = default;

    /// \brief Construct a symmetry from its string specification.
    /// \param spec String specification such as `"N:U(1),Sz:U(1)"`.
    explicit Symmetry(std::string_view spec) : impl_(detail::SymmetryImpl::parse(spec)) {}

    /// \brief Construct from a canonical implementation pointer.
    /// \param impl Canonical implementation pointer.
    explicit Symmetry(detail::SymmetryImpl const* impl) : impl_(impl) {}

    /// \brief Parse a string specification such as `"N:U(1),Sz:U(1)"`.
    /// \param spec String specification to parse.
    /// \return Canonicalized symmetry handle.
    static auto parse(std::string_view spec) -> Symmetry { return Symmetry{detail::SymmetryImpl::parse(spec)}; }

    /// \brief Return whether this handle refers to a valid symmetry.
    /// \return `true` if this handle refers to canonical symmetry state.
    auto valid() const -> bool { return impl_ != nullptr; }

    /// \brief Return the number of direct-product factors.
    /// \return Number of named factors.
    auto factor_count() const -> std::size_t
    {
        this->ensure_valid();
        return impl_->factor_count();
    }

    /// \brief Return the named factor list.
    /// \return Read-only view of the factor specifications.
    auto factors() const -> std::span<detail::SymmetryFactorSpec const>
    {
        this->ensure_valid();
        return impl_->factors();
    }

    /// \brief Return the canonical string spelling of this symmetry.
    /// \return Canonical string form.
    auto to_string() const -> std::string
    {
        this->ensure_valid();
        return impl_->to_string();
    }

    /// \brief Return the internal canonical symmetry implementation pointer.
    /// \return Canonical implementation pointer used for equality and dispatch.
    auto impl() const -> detail::SymmetryImpl const*
    {
        this->ensure_valid();
        return impl_;
    }

    /// \brief Compare two symmetry handles for canonical identity.
    /// \param other Other symmetry handle.
    /// \return `true` when both handles refer to the same canonical symmetry.
    auto operator==(Symmetry const& other) const -> bool = default;

  private:
    /// \brief Throw if the symmetry handle is invalid.
    void ensure_valid() const
    {
        if (!this->valid())
        {
            throw std::logic_error("Symmetry handle is not initialized");
        }
    }

    detail::SymmetryImpl const* impl_ = nullptr;
};

/// \brief Format a symmetry handle as its canonical string form.
/// \param symmetry Symmetry handle to format.
/// \return Canonical string spelling of the symmetry.
inline auto to_string(Symmetry const& symmetry) -> std::string { return symmetry.to_string(); }

} // namespace uni20
