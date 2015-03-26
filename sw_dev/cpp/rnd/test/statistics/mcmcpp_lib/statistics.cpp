///
/// \file   statistics.cpp
/// \brief Classes for descriptive statistics
///
/// This file provides two statistical classes: Statistic and
/// SimpleStatistic. As the names suggest, Statistic is more complete. It 
/// includes methods for standard deviation and coefficient of variation
/// as well as mean and variance. It can also calculate statistics on 
/// ratios (using ratio.h)
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
#include "mcmc++/statistics.h"

/// \class Statistic
/// \brief Implements a class for summary statistics.
///
/// The Statistic class allows easy calculation of summary statistics.
/// When applied to ratio data the mean is the ratio of the means, and the
/// variance is calculated from the ratio of the sums of squares. 
/// Specifically, let
/// \f[ s  = \frac{\sum_i x_{i,top}}{\sum_i x_{i,bottom}} \f]
/// \f[ ss = \frac{\sum_i x_{i,top}^2}{\sum_i x_{i,bottom}^2} \f]
/// then Mean() returns
/// \f[ s/n \f]
/// and Variance() returns
/// \f[ (ss - (s*s)/n)/(n-1) \f]
///
/// Example:
///
/// \code
/// Statistic stat;
/// stat.Add(1.0);
/// stat.Add(2.0);
/// stat.Add(3.0);
/// stat.Add(4.0);
/// stat.Add(5.0);
/// long  n    = stat.N();    // n = 5
/// double mean   = stat.Mean();   // mean = 3.000
/// double variance  = stat.Variance(); // variance = 2.500
/// double stddev  = stat.StdDev();  // stddev = 1.581
/// double cv    = stat.CV();   // cv = 0.527
/// \endcode
///

/// Constructor
///
/// The constructor for statistic simply initializes all internal
/// values in preparation for calculating statistics. After initializtion,
/// data is added with Add().
///
Statistic::Statistic(void)
  : sum(0.0), lsum(0.0), sumSq(0.0), lsumSq(0.0), mean(0.0), 
    variance(0.0), stddev(0.0), cv(0.0), n(0L), dirty(0) 
{}

void 
Statistic::CalcMean(void) {
  if (lsum > 0.0) {
    mean = sum/lsum;
  } else {
    mean = sum/static_cast<double>(n);
  }
}

void 
Statistic::CalcVariance(void) {
  double s = sum;
  double ss = sumSq;

  if (lsum > 0.0) {
    s /= lsum;
    s *= static_cast<double>(n);
    ss /= lsumSq;
    ss *= static_cast<double>(n);
  }
  variance = (ss - s*s/static_cast<double>(n))/static_cast<double>(n - 1);
}

void 
Statistic::CalcAll(void) {
  if (dirty && n > 0)
    CalcMean();
  if (dirty && n > 1) {
    CalcMean();
    CalcVariance();
    CalcStdDev();
    CalcCV();
  }
  dirty = 0;
}

/// Returns arithmetic mean of the data.
///
double 
Statistic::Mean(void) {
  CalcAll();
  return mean;
}

/// Returns variance of the data.
///
double 
Statistic::Variance(void) {
  CalcAll();
  return variance;
}

/// Returns standard deviation of the data.
///
double 
Statistic::StdDev() {
  CalcAll();
  return stddev;
}

/// Returns coefficient of variation of the data.
///
double 
Statistic::CV() {
  CalcAll();
  return cv;
}

/// Add a double value to the statistic.
///
/// \param v  The value to add
///
void 
Statistic::Add(const double v) {
  sum += v;
  sumSq += v * v;
  n++;
  dirty = 1;
}

/// Add a ratio to the statistic.
///
/// \param r   The ratio to add
///
void 
Statistic::Add(const ratio r) {
  sum += r.Top();
  lsum += r.Bottom();
  sumSq += r.Top() * r.Top();
  lsumSq += r.Bottom() * r.Bottom();
  n++;
  dirty = 1;
}

/// Add a double value to the statistic
///
/// \param v   The value to add
///
Statistic& Statistic::operator +=(const double v) {
  Add(v);
  return *this;
}

/// Add a ratio to the statistic.
///
/// \param r   The ratio to add
///
Statistic& 
Statistic::operator +=(const ratio r) {
  Add(r);
  return *this;
}

/// Stream output for Statistic.
///
/// Reports sample size, mean, variance, standard deviation, and
/// coefficient of variation, each preceded by a tab and appearing
/// on a new line.
///
/// \param out   The output stream
/// \param st    The statistics
///
std::ostream& 
operator <<(std::ostream& out, Statistic& st) {
  out << "\n\tN     = " << st.N();
  out << "\n\tmean   = " << st.Mean();
  out << "\n\tvariance = " << st.Variance();
  out << "\n\tstd. dev. = " << st.StdDev();
  out << "\n\tCV    = " << st.CV();
  return out;
}


/// \class SimpleStatistic
/// \brief Implements a class for summary statistics.
///
/// The SimpleStatistic class allows easy calculation of simple summary 
/// statistics. Unlike Statistic, it cannot be applied to ratio data. In
/// addition to adding single values, with Add(), an entire vector of
/// values can be added in the constructor.
///
/// Example:
///
/// \code
/// SimpleStatistic stat;
/// stat.Add(1.0);
/// stat.Add(2.0);
/// stat.Add(3.0);
/// stat.Add(4.0);
/// stat.Add(5.0);
/// double mean   = stat.Mean();   // mean = 3.000
/// double variance  = stat.Variance(); // variance = 2.500
/// double stddev  = stat.StdDev();  // stddev = 1.581
/// stat.Clear();  // clears internal data for new calculation
///
/// vector<double> x(5);
/// for (int i = 0; i < 5; ++i) {
///   x[i] = i;
/// }
/// SimpleStatistic vecStat(x);
/// double mean   = stat.Mean();   // mean = 3.000
/// double variance  = stat.Variance(); // variance = 2.500
/// double stddev  = stat.StdDev();  // stddev = 1.581
/// \endcode
///

/// Default constructor.
///
/// The default constructor simply initializes the internal data.
/// After initializaiton, data is added with Add().
///
SimpleStatistic::SimpleStatistic(void)
  : sum_(0.0), sumsq_(0.0), n_(0) 
{}

/// Add a double value to the statistic.
///
/// \param x  Value to add
///
void 
SimpleStatistic::Add(const double x) {
  sum_ += x;
  sumsq_ += sqr(x);
  n_++;
}

/// Returns mean of the data.
///
double 
SimpleStatistic::Mean(void) const {
  return sum_/n_;
}

/// Returns variance of the data.
///
double 
SimpleStatistic::Variance(void) const {
  return (sumsq_ - sqr(sum_)/n_)/(n_ - 1.0);
}

/// Returns standard deviation of the data.
///
double 
SimpleStatistic::StdDev(void) const {
  return sqrt(Variance());
}

/// Re-initializes internal data for new calculations.
///
void 
SimpleStatistic::Clear(void) {
  sum_ = sumsq_ = 0.0;
  n_ = 0;
}

double 
SimpleStatistic::sqr(const double x) const {
  return x*x;
}




  


