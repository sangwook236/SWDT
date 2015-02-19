///
/// \file   ratio.h
/// \brief  Provides a simple ratio class.
///
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

#if !defined(__RATIO_H)
#define __RATIO_H

// local includes
#include "mcmc++/util.h"

class ratio {
public:
  ratio(void);
  ratio(const ratio& r);

  ratio& operator +=(const ratio& r);
  ratio& operator +=(const double d);
  ratio& operator /=(const ratio& r);
  ratio& operator /=(double d);
  ratio& operator =(const ratio& r);
  bool operator ==(const ratio& r) const;
  bool operator !=(const ratio& r) const;

  double make_double(void) const;
  double Top(void) const;
  double Bottom(void) const;

private:
  double top_;
  double bottom_;
  
};


#endif

// Local Variables: //
// mode: c++ //
// End: //
