///
/// \file   intervals.h
/// \brief  declarations for quantile(), p_value(), and hpd()
/// \author Kent Holsinger
/// \date   2004-06-26
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

#if !defined(__INTERVALS_H)
#define __INTERVALS_H


// standard includes
#include <cmath>
#include <vector>

double quantile(std::vector<double>& x, double p);
double p_value(std::vector<double>& x, double p);
std::vector<double> hpd(std::vector<double>& x, double p);

#endif

// Local Variables: //
// mode: c++ //
// End: //

