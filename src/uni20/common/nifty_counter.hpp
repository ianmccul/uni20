#pragma once

/**
 * \file nifty_counter.hpp
 * \ingroup common
 * \brief Provides ordered one-time initialization across translation units.
 */

#include <cstddef>

namespace uni20
{

/// \brief Runs shared initialization before namespace-scope users and shared cleanup after them.
/// \details Place one guard with internal linkage in every header that exposes state requiring ordered
///          initialization. Named callback functions with external linkage ensure that all translation units
///          instantiate the same counter. Header-local lambdas must not be used for that purpose because they
///          can produce distinct counter specializations.
/// \tparam Init No-throw callable invoked when the first guard is constructed.
/// \tparam Exit No-throw callable invoked when the last guard is destroyed.
template <auto Init, auto Exit = []() noexcept {}>
  requires requires {
    { Init() } noexcept;
    { Exit() } noexcept;
  }
class nifty_counter {
  public:
    nifty_counter() noexcept
    {
      if (count_++ == 0) Init();
    }

    nifty_counter(nifty_counter const&) = delete;
    nifty_counter(nifty_counter&&) = delete;
    nifty_counter& operator=(nifty_counter const&) = delete;
    nifty_counter& operator=(nifty_counter&&) = delete;

    ~nifty_counter() noexcept
    {
      if (--count_ == 0) Exit();
    }

  private:
    static constinit inline std::size_t count_{0};
};

} // namespace uni20
