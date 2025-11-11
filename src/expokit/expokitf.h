// -*- C++ -*-
//----------------------------------------------------------------------------
// Matrix Product Toolkit http://mptoolkit.qusim.net/
//
// expokit/expokitf.h
//
// Copyright (C) 2004-2016 Ian McCulloch <ian@qusim.net>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Research publications making use of this software should include
// appropriate citations and acknowledgements as described in
// the file CITATIONS in the main source directory.
//----------------------------------------------------------------------------
// ENDHEADER

/*
  expokitf.h

  C++ interface to EXPOKIT.
*/

#pragma once

#include <complex>

#include "core/scalar_concepts.hpp"
#include "expokit/matrix.hpp"

namespace EXPOKIT
{

/// \brief Compute the matrix exponential using the adaptive scaling-and-squaring algorithm.
/// \details Follows the Pad\'e-based scaling and squaring strategy of Higham (2005) and
///          Al-Mohy & Higham (2011) while maintaining compatibility with the classical
///          EXPOKIT interfaces derived from Sidje (1998). The routine automatically selects
///          between Pad\'e degrees {3, 5, 7, 9, 13} based on matrix norms.
/// \tparam Scalar Matrix element type; may be real or complex in single or double precision.
/// \param matrix Matrix whose exponential will be evaluated.
/// \param t Scalar multiplier applied to \p matrix before exponentiation.
/// \param ideg Legacy Pad\'e degree hint retained for compatibility.
/// \return The matrix exponential of \f$\exp(t \cdot \text{matrix})\f$.
template <uni20::RealOrComplex Scalar>
Matrix<Scalar> expm(Matrix<Scalar> const& matrix, uni20::make_real_t<Scalar> t, int ideg = 9);

} // namespace EXPOKIT
