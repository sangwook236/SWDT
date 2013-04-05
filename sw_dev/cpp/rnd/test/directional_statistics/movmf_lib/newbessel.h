// File: new_bessel.h  -*- c++ -*-
// Author: Suvrit Sra
// Interface for bessel function routines of arbitrary precision.
/*  Author: Suvrit Sra <suvrit@cs.utexas.edu> */
/*  (c) Copyright 2006,2007   Suvrit Sra */
/*  The University of Texas at Austin */
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


#ifndef _NEWBESSEL_H
#define _NEWBESSEL_H


#include <NTL/RR.h>

using namespace NTL;

// Only for real arguments, extends trivially for complex
//, but i don't care for complex arguments as of now!
RR BesselI(RR& s, RR& x);

RR BesselI(double& s, double& x);

#endif // _NEWBESSEL_H
