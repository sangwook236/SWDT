///
/// \file   ratio.cpp
/// \brief  Provides a simple ratio class.
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

// local includes
#include "mcmc++/ratio.h"

/// \class ratio
/// \brief Provides a ratio class
///
/// The ratio class allows numerators and denominators to be summed separately.
/// Access to each is provided.
///

/// Default constructor.
///
/// Initializes private variables 
///
ratio::ratio(void) : 
  top_(0.0), bottom_(0.0) 
{}

/// Copy constructor.
///
/// Initializes private variables from existing ratio
///
ratio::ratio(const ratio& r) 
  : top_(r.top_), bottom_(r.bottom_) 
{}

/// Add ratio to current ratio.
///
/// \param r  The ratio to add
///
ratio& 
ratio::operator +=(const ratio& r) {
  top_ += r.top_;
  bottom_ += r.bottom_;
  return *this;
}

/// Add double to current ratio.
///
/// Adds the d to both numerator and denominator
///
/// \param d  The double value to add
///
ratio& 
ratio::operator +=(const double d) {
  top_ += d;
  bottom_ += d;
  return *this;
}

/// Divide one ratio by another.
///
/// Numerator divided by numerator. Denominator divided by denominator.
///
/// \param r  The ratio to be used in the "denominator" of the division
///
ratio& 
ratio::operator /=(const ratio& r) {
  top_ /= r.top_;
  bottom_ /= r.bottom_;
  return *this;
}

/// Divide a ratio by a double.
///
/// Numerator and denominator both divided by d.
/// 
/// \param d
///
ratio& 
ratio::operator /=(const double d) {
  top_ /= d;
  bottom_ /= d;
  return *this;
}

/// Assignment operator.
///
/// \param r   The value being assigned
///
ratio& 
ratio::operator =(const ratio& r) {
  top_ = r.top_;
  bottom_ = r.bottom_;
  return *this;
}

/// Equality test.
///
/// Equal if and only if top_ and bottom_ of both ratios are equal.
///
/// \param r   The value being compared
///
bool 
ratio::operator ==(const ratio& r) const {
  return ( top_ == r.top_ && bottom_ == r.bottom_ );
}

/// Inequality test.
///
/// Not equal if top_s or bottom_s are unequal
///
/// \param r   The value being compared
///
bool 
ratio::operator !=(const ratio& r) const {
  return ( top_ != r.top_ || bottom_ != r.bottom_ );
}

/// Value of ratio
///
double 
ratio::make_double(void) const {
  double retval;
  if ( bottom_ != 0.0 ) {
    retval = top_/bottom_;
  } else if ( top_ < 0.0 ) {
    retval = Util::long_min;
  } else if ( top_ > 0.0 ) {
    retval = Util::long_max;
  }
  return retval;
}

/// Numerator of ratio
///
double 
ratio::Top(void) const {
  return top_;
}

/// Denominator of ratio
///
double 
ratio::Bottom(void) const {
  return bottom_;
}
