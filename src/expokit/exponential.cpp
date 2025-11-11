// -*- C++ -*-
//----------------------------------------------------------------------------
// Matrix Product Toolkit http://mptoolkit.qusim.net/
//
// linearalgebra/exponential.cpp
//
// Copyright (C) 2006-2016 Ian McCulloch <ian@qusim.net>
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
//
// implementation of eigen.h EXPOKIT wrappers
//
// Created 2006-05-12 Ian McCulloch
//

#include "eigen.h"
#include "expokit/expokitf.h"

namespace LinearAlgebra
{

namespace Private
{

using Complex = std::complex<double>;
using Matrix = EXPOKIT::Matrix<Complex>;

Matrix expm(Matrix const& matrix, double t, int ideg)
{
   return EXPOKIT::expm(matrix, t, ideg);
}

Matrix expm(Matrix const& matrix, double t)
{
   int constexpr pade_degree = 9;
   return EXPOKIT::expm(matrix, t, pade_degree);
}

} // namespace Private

} // namespace LinearAlgebra
