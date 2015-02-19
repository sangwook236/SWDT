///
/// \file   util.cpp
/// \brief  Collects a variety of constants and function in namespace Util.
///
/// Provides definitions for a variety of numerical constants related to 
/// double precision and integer arithmetic and for a small collection of 
/// utility functions.
///
/// All are declared in namespace Util with an eye towards avoiding
/// naming conflicts.
///
/// \author Kent Holsinger
/// \date   2005-05-18
///

// This file is part of MCMC++, a library for constructing C++ programs
// that implement MCMC analyses of Bayesian statistical models.
// Copyright (c) 2004-2006 Kent E. Holsinger
//
// MCMC++ is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// MCMC++ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with MCMC++; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include <cassert>
// local includes
#include "mcmc++/util.h"

/// Round a floating point number
///
/// Formula:
///
/// \f[\mbox{floor}(x + 0.5)\f]
///
/// \param x
///
double
Util::round(const double x) {
  return floor(x + 0.5);
}

/// Returns log_dbl_min if x < log_dbl_min
///
/// \param x   The logarithm to check
///
double 
Util::safeLog(const double x) {
  return (x > log_dbl_min) ? x : log_dbl_min;
}

/// Returns \f$x^2\f$
///
/// \param x
///
double 
Util::sqr(const double x) {
  return x*x;
}

