// File: newbessel.cc
//  Author: Suvrit Sra <suvrit@cs.utexas.edu>
//  (c) Copyright 2006,2007   Suvrit Sra
//  The University of Texas at Austin
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.


#include "newbessel.h"

// Just use the powerseries of I_s[x] and truncate
#define HALF_E to_RR("1.35914091422952")
#define HALF_ED 1.35914091422952
#define INV_2ROOTPI to_RR("0.39894228040143")
#define TOL to_RR("1e-30")

RR BesselI(double& s, double& x)
{

  if (x == 0) return to_RR("0.0");
  if (s == 0) return to_RR("1.0");

  RR scale_term;
  RR srr = to_RR(s);

  scale_term = pow ( to_RR(HALF_ED*x / s), srr);
  RR tmp;
  tmp = 1 + 1.0/(12*s) + 1.0/(288*s*s) - 139.0/(51840*s*s*s);
  scale_term *= sqrt(s) * INV_2ROOTPI / tmp;

  double ratio;
  RR tol = TOL;

  RR aterm;
  aterm = 1.0/s;
  RR sum; sum = aterm;
  int k = 1;

  while (true) {
    ratio = ((0.25*x*x) / (k * (s+k)));
    aterm *= ratio;
    if (aterm  < tol*sum) break;
    sum += aterm;
    ++k;
  }
  sum *= scale_term;
  return sum;
}

RR BesselI(RR& s, RR& x)
{
  RR scale_term;
  RR tol;
  tol = TOL;

  scale_term = pow ( HALF_E*x / s, s);
  RR tmp;
  tmp = 1 + 1.0/(12*s) + 1.0/(288*s*s) - 139.0/(51840*s*s*s);
  scale_term *= sqrt(s) * INV_2ROOTPI / tmp;

  //std::cerr << "scale term = " << scale_term << "\n";
  // Now do the series computation ...
  RR aterm;
  aterm = 1.0/s;
  RR sum; sum = aterm;
  int k = 1;

  while (true) {

    aterm *= ((0.25*x*x) / (k * (s+k)));
    if (aterm / sum < tol) break;
    sum += aterm;
    ++k;
  }
  sum *= scale_term;
  return sum;
}
