#pragma once

/**
 * \file diagnostic_error.hpp
 * \ingroup common
 * \brief Structured exception base carrying diagnostic source context.
 */

#include <uni20/config.hpp>

#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <utility>

#if UNI20_HAS_STACKTRACE
#include <stacktrace>
#endif

namespace uni20
{

/// \brief Base class for exceptions raised through Uni20's diagnostic boundary.
class diagnostic_error : public std::runtime_error {
  public:
    /// \brief Construct an exception with its concise plain-text message.
    explicit diagnostic_error(std::string message) : std::runtime_error(std::move(message)) {}

    /// \brief Attach the source location at which the exception is raised.
    void set_source_location(std::source_location where) noexcept { source_location_ = where; }

    /// \brief Return the attached source location, if any.
    [[nodiscard]] std::optional<std::source_location> const& source_location() const noexcept
    {
      return source_location_;
    }

#if UNI20_HAS_STACKTRACE
    /// \brief Attach the stacktrace captured when the exception is raised.
    void set_stacktrace(std::stacktrace stacktrace) { stacktrace_ = std::move(stacktrace); }

    /// \brief Return the attached stacktrace, if any.
    [[nodiscard]] std::optional<std::stacktrace> const& stacktrace() const noexcept { return stacktrace_; }
#endif

  private:
    std::optional<std::source_location> source_location_;
#if UNI20_HAS_STACKTRACE
    std::optional<std::stacktrace> stacktrace_;
#endif
};

} // namespace uni20
