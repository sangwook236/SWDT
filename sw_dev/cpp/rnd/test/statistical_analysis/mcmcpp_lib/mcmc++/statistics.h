///
/// \file  statistics.h
/// \brief Classes for descriptive statistics
///
/// This header file provides two statistical classes: Statistic and
/// SimpleStatistic. As the names suggest, Statistic is more complete. It 
/// includes methods for standard deviation and coefficient of variation
/// as well as mean and variance. It can also calculate statistics on 
/// ratios (using ratio.h)
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

#if !defined(__STATISTI_H)
#define __STATISTI_H

// standard includes
#include <iostream>
// boost includes
#include <boost/type_traits.hpp>
// local includes
#include "mcmc++/ratio.h"


namespace keh { // to make sure that we avoid collisions

  /// Base template for Accumulate (used only for definition
  ///
  template <bool U, class T>
  class Accumulate;
  
  /// Designed for use with a vector<T>, where a method in T is called
  /// before this constructor to provide an implementation of operator* that
  /// returns an appropriate member from T (one that can be converted via
  /// default conversions to a double)
  ///
  template <class T>
  class Accumulate<false, T> {
  public:
    /// Constructor
    ///
    /// \param x     vector of values
    /// \param sum   sum of values
    /// \param sumsq sum of squared values
    /// \param n     number of values, i.e., length of the vector
    ///
    /// The constructor is useful because of its side effects, namely 
    /// sum, sumsq, and n are adjusted and are accesible from the calling
    /// function
    ///
    Accumulate(std::vector<T>& x, double& sum, double& sumsq, 
               unsigned long& n)
    {
      sum = sumsq = 0.0;
      typedef typename std::vector<T>::const_iterator iter;
      iter begin = x.begin();
      iter end = x.end();
      for (iter i = begin; i != end; ++i) {
        double value = **i;
        sum += value;
        sumsq += sqr(value);
      }
      n = x.size();
    }
  private:
    static double sqr(const double x) {
      return x*x;
    }
  };
  
  /// Specialization of Accumulate for arithmetic types
  ///
  template <class T>
  class Accumulate<true, T> {
  public:
    /// Constructor
    ///
    /// \param x     vector of values
    /// \param sum   sum of values
    /// \param sumsq sum of squared values
    /// \param n     number of values, i.e., length of the vector
    ///
    /// The constructor is useful because of its side effects, namely 
    /// sum, sumsq, and n are adjusted and are accesible from the calling
    /// function
    ///
    Accumulate(std::vector<T>& x, double& sum, double& sumsq, 
               unsigned long& n) 
    {
      n = x.size();
      sum = sumsq = 0.0;
      typedef typename std::vector<T>::const_iterator iter;
      iter end = x.end();
      for (iter i = x.begin(); i != end; ++i) {
        double value = *i;
        sum += value;
        sumsq += sqr(value);
      }
    }
  private:
    static double sqr(const double x) {
      return x*x;
    }

  };

} // end of namespace keh

class Statistic {
  double sum;
  double lsum; // lower sum used only for ratios
  double sumSq;
  double lsumSq; // lower sum of squares used only for ratios
  double mean;
  double variance;
  double stddev;
  double cv;
  long n;
  int dirty;
  void CalcMean(void);
  void CalcVariance(void);
  void CalcStdDev(void) {
    stddev = (variance <= 0.0 ? 0.0 : sqrt(variance));
  }
  void CalcCV(void) {
    cv = (mean == 0.0 ? 0.0 : stddev/mean);
  }
  void CalcAll(void);

public:
  Statistic(void);
  void Add(double);
  void Add(ratio);
  /// returns sample size
  ///
  long N(void) {
    return n;
  }
  /// returns sum of sample values
  ///
  double Sum(void) {
    return sum;
  }
  /// returns sum of squared sample values
  ///
  double SumSq(void) {
    return sumSq;
  }
  double Mean(void);
  double Variance(void);
  double StdDev(void);
  double CV(void);
  Statistic& operator +=(double v);
  Statistic& operator +=(ratio r);
  friend std::ostream& operator <<(std::ostream&, Statistic&);
};

class SimpleStatistic {
public:
  SimpleStatistic(void);

  /// Constructor -- initialize with a vector.
  ///
  /// May be used with any vector having an iterator that can produce a
  /// double. This may be a simple vector<double> (or any other vector 
  /// whose elements can be converted to double by default conversions,
  /// but it could also be a vector<T>, where a method in T is called 
  /// before this constructor to provide an implementation of operator* 
  /// that returns an appropriate member from T (one that can be converted 
  /// via default conversions to a double).
  ///
  /// \param x   The vector to use in calculations
  ///
  template <class T>
  SimpleStatistic(std::vector<T>& x) {
    keh::Accumulate<boost::is_arithmetic<T>::value, T> 
      stats(x, sum_, sumsq_, n_);
  }
  void Add(double x);
  double Mean(void) const;
  double Variance(void) const;
  double StdDev(void) const;
  void Clear(void);

private:
  double sqr(const double x) const;

  double sum_;
  double sumsq_;
  unsigned long n_;

};

#endif

// Local Variables: //
// mode: c++ //
// End: //
