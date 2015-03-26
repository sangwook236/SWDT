///
/// \file   util.h
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

#if !defined(__UTIL_H)
#define __UTIL_H

// standard includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

/// \namespace Util
///
/// Used to isolate utility functions and numerical constants.
///
/// Provides a variety of numerical constants related to double precision 
/// and integer arithmetic, a small collection of utility functions, and 
/// for probability density functions.
///
namespace Util {

  const double dbl_eps = std::numeric_limits<double>::epsilon();   ///< minimum representable value of 1.0 - x
  const double dbl_max = std::numeric_limits<double>::max(); ///< maximum value of double
  const double dbl_min = std::numeric_limits<double>::min(); ///< minimum (positive) value of double
  const int int_max = std::numeric_limits<int>::max(); ///< maximum value of integer
  const int int_min = std::numeric_limits<int>::min(); ///< minimum value of integer
  const unsigned uint_max = std::numeric_limits<unsigned>::max(); ///< maximum value of unsigned integer
  const long long_max = std::numeric_limits<long>::max(); ///< maximum value of long integer
  const long long_min = std::numeric_limits<long>::min(); ///< minimum value of long integer
  const unsigned long ulong_max = std::numeric_limits<unsigned long>::max(); ///< maximum value of unsigned long integer
  const double log_dbl_max = log(dbl_max); ///< log(dbl_max)
  const double log_dbl_min = log(dbl_min); ///< log(dbl_min)

  /// Returns minimum element in v.
  ///
  /// This is a simple wrapper around std::min_element()
  ///
  /// \param v  The vector whose minimum element is sought
  ///
  template <class C>
  inline C vectorMin(std::vector<C>& v) {
    return *std::min_element(v.begin(), v.end());
  }

  /// Returns maximum element in v.
  ///
  /// This is a simple wrapper around std::max_element()
  ///
  /// \param v  The vector whose maximum element is sought
  ///
  template <class C>
  inline C vectorMax(std::vector<C>& v) {
    return *std::max_element(v.begin(), v.end());
  }

  double round(double x);
  double safeLog(double x);
  double sqr(double x);
  
  /// Empty a vector, and ensure that its capacity is zero
  ///
  /// \param v  The vector to be emptied
  ///
  /// Uses the trick on p. 487 of Stroustrup (3rd ed.): create a temporary
  /// vector with zero capacity and swap it with the one to be erased.
  ///
  template <class C>
  void FlushVector(std::vector<C>& v) {
    std::vector<C> tmp;
    v.swap(tmp);
  }

  /// Assertion template for error checking
  ///
  /// \param assert     the assertion to check
  ///
  /// This template throws an exception of type Except when the assertion
  /// of type Assertion is false. It is shamelessly adapted (stolen) from 
  /// Stroustrup, The C++ Programming Language, 3rd ed.
  ///
  template <class Except, class Assertion>
  inline void Assert(Assertion assert) {
    if (!assert) throw Except();
  }

  /// Cast all elements of a vector from one type to another
  ///
  /// \param x   the vector with elements to cast
  ///
  /// This template will cast elements of type From to elements of type
  /// To and return the resulting vector
  ///
  template <class To, class From>
  inline std::vector<To> vector_cast(std::vector<From>& x) {
    unsigned end = x.size();
    std::vector<To> y(end);
    for (unsigned i = 0; i < end; ++i) {
      y[i] = static_cast<To>(x[i]);
    }
    return y;
  }
  
  /// \class PrintForVector
  ///
  /// This template provides a helper class for the template function
  /// below for std::vector<T> operator <<
  ///
  template <class T>
  class PrintForVector {
  public:
    /// Constructor
    ///
    explicit PrintForVector(std::ostream& os) 
      : os_(os), first_(true)
    {}
    /// allows for_each to provide appropriate output for vector elements
    ///
    void operator()(const T& x) {
      if (!first_) {
        os_ << ", ";
      } else {
        first_ = false;
      }
      os_ << x;
    }
  private:
    std::ostream& os_;
    bool first_;
  }; 

}

/// ostream& operator<< for vectors
///
/// \param os  the ostream
/// \param x   the vector
///
template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> x) {
  os << "(";
  std::for_each(x.begin(), x.end(), Util::PrintForVector<T>(os));
  os << ")";
  return os;
}

#endif

// Local Variables: //
// mode: c++ //
// End: //
