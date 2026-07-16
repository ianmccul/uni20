#pragma once

/**
 * \file mdspan.hpp
 * \ingroup mdspan_ext
 * \brief Configured gateway to Uni20's mdspan implementation.
 */

#include <uni20/config.hpp>

#ifndef MDSPAN_IMPL_STANDARD_NAMESPACE
#error "cmake configuration error: mdspan namespace is not defined"
#endif

#include <mdspan/mdspan.hpp>
