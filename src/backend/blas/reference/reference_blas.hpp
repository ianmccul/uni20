#pragma once

/**
 * \defgroup backend_blas_reference Reference BLAS backend
 * \ingroup backend_blas
 * \brief Header-only wrappers that provide a portable BLAS fallback implementation.
 * \details The reference backend expands the generic BLAS shims defined in
 *          `detail/blasproto.hpp` for each supported scalar type. These wrappers are primarily
 *          used for tests and environments where no optimized vendor library is available.
 */

/// \file reference_blas.hpp
/// \brief Instantiates the reference BLAS backend over all supported scalar types.
/// \ingroup backend_blas_reference

#include "backend/backend.hpp"
#include "config.hpp"
#include "core/types.hpp"

/// \brief Internal macro specifying the BLAS routine prefix for single-precision instantiations.
/// \ingroup internal
#define BLASCHAR s
/// \brief Internal macro mapping the BLAS prefix to the associated Uni20 scalar type.
/// \ingroup internal
#define BLASTYPE float32
/// \brief Internal flag indicating that the single-precision instantiation is real-valued.
/// \ingroup internal
#define BLASCOMPLEX 0
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

/// \brief Internal macro specifying the BLAS routine prefix for double-precision instantiations.
/// \ingroup internal
#define BLASCHAR d
/// \brief Internal macro mapping the BLAS prefix to the associated Uni20 scalar type.
/// \ingroup internal
#define BLASTYPE float64
/// \brief Internal flag indicating that the double-precision instantiation is real-valued.
/// \ingroup internal
#define BLASCOMPLEX 0
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASCOMPLEX

/// \brief Internal macro specifying the BLAS routine prefix for complex single-precision instantiations.
/// \ingroup internal
#define BLASCHAR c
/// \brief Internal macro mapping the complex BLAS prefix to the associated Uni20 scalar type.
/// \ingroup internal
#define BLASTYPE complex64
/// \brief Internal macro naming the real-valued companion type for complex single-precision instantiations.
/// \ingroup internal
#define BLASREALTYPE float32
/// \brief Internal flag indicating that the complex single-precision instantiation uses complex arithmetic.
/// \ingroup internal
#define BLASCOMPLEX 1
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASREALTYPE
#undef BLASCOMPLEX

/// \brief Internal macro specifying the BLAS routine prefix for complex double-precision instantiations.
/// \ingroup internal
#define BLASCHAR z
/// \brief Internal macro mapping the complex double-precision BLAS prefix to the associated Uni20 scalar type.
/// \ingroup internal
#define BLASTYPE complex128
/// \brief Internal macro naming the real-valued companion type for complex double-precision instantiations.
/// \ingroup internal
#define BLASREALTYPE float64
/// \brief Internal flag indicating that the complex double-precision instantiation uses complex arithmetic.
/// \ingroup internal
#define BLASCOMPLEX 1
#include "detail/blasproto.hpp"
#undef BLASCHAR
#undef BLASTYPE
#undef BLASREALTYPE
#undef BLASCOMPLEX
