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

#include "expokit/matrix.hpp"

namespace EXPOKIT
{

Matrix<std::complex<double>> expm(Matrix<std::complex<double>> const& matrix, double t,
                                  int ideg = 9);

} // namespace EXPOKIT
