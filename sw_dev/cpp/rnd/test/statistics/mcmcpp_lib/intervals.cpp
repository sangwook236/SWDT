///
/// \file   intervals.cpp
/// \brief  defintions for quantile(), p_value(), and hpd()
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

// standard includes
#include <algorithm>
// local includes
#include "mcmc++/intervals.h"

using std::vector;

/// Returns the sample quantile corresponding to p.
///
/// This implementation corresponds to Type 8 continuous sample types
/// of Hyndman and Fan (American Statistician, 50:361-365; 1996)
///
/// Notice that as a side effect of calling this routine, the input
/// vector is sorted.
///
/// \param ox  The vector from which to calculate the quantile
/// \param p   The desired quantile
///
double quantile(vector<double>& ox, const double p) {
  unsigned int n = ox.size();
  double pn = n*p;
  double m = (p + 1.0)/3.0;
  unsigned int j = static_cast<int>(floor(pn + m));
  double q;
  std::sort(ox.begin(), ox.end());
  if (j == 0) {
    q = ox[0];
  } else if (j == n) {
    q = ox[n-1];
  } else {
    double g = pn + m - j;
    q = (1.0 - g)*ox[j - 1] + g*ox[j];
  }
  return q;
}

/// Returns P(x <= p).
///
/// Notice that as a side effect of calling this routine, the input
/// vector is sorted.
///
/// \param x  The vector
/// \param p  The desired P-value
///
double p_value(vector<double>& x, const double p) {
  unsigned i;
  std::sort(x.begin(), x.end());
  for (i = 0; i < x.size(); i++) {
    if (x[i] > p) {
      break;
    }
  }
  return static_cast<double>(i) / x.size();
}

/// Returns HPD interval, low in x[0], high in x[1].
///
/// Returns lower limit of HPD interval in first element of vector, upper 
/// limit of HPD interval in second element of vector. This method is an
/// implementation of the Chen-Shao HPD estimation algorithm.
///
/// Notice that as a side effect of calling this routine, the input
/// vector is sorted.
///
/// \param x  The vector
/// \param p  The credible interval desired, e.g., 0.95 for 95%
///
vector<double> hpd(vector<double>& x, const double p) {
  vector<double> interval;
  std::sort(x.begin(), x.end());
  // x.size()*p is the number of elements desired
  // x.size()*p - 1 is the index of the first (starting from 0)
  int idx = static_cast<int>(floor(x.size() * p - 1 + 0.5));
  double shortest = x[idx] - x[0];
  interval.push_back(x[0]);
  interval.push_back(x[idx]);

  for (unsigned int i = 1; i + idx < x.size(); i++) {
    if (x[idx + i] - x[i] < shortest) {
      shortest = x[idx + i] - x[i];
      interval[0] = x[i];
      interval[1] = x[i + idx];
    }
  }

  return interval;
}




