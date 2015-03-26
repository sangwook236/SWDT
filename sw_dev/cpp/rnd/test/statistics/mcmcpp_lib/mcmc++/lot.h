///
/// \file   lot.h
/// \brief  Random number generators
///
/// A series of random number generators are provided here. Any one of three
/// uniform random number generators can be chosen. By default the Mersenne
/// twister is used.
///
/// \author Kent Holsinger & Paul Lewis
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

#ifndef __LOT_H
#define __LOT_H

// standard includes
#include <cassert>
// local includes
#include "mcmc++/util.h"

/// typedef to allow more succint declarations for some variables
///
typedef std::vector<double> DblVect;

/// \class lot
/// \brief Provides a series of random number generators
///
/// This class provides a series of random number generators. Three different
/// uniform random number generators are provided. By default the Mersenne
/// twister is used. Sources are acknowledged in the source code by including 
/// copyright information (where appropriate).
///
/// It is given the name "lot" because the noun lot is defined as "an object 
/// used in deciding something by chance" according to The New Merriam-Webster 
/// Dictionary,
///
/// To seed the random number generator with a specific value (useful for
/// debugging stochastic simulations), do the following:
///
/// \code
///     lot rng;
///     rng.set_seed(1234L);
/// \endcode
///
/// You can, of course pick any number you like.
///

class lot {
public:
  enum {
    RAN_POL = 1,   ///< Paul's original RNG from J. Monahan, NCSU
    RAN_KNU,       ///< Knuth's rng.c translated to C++
    RAN_MT         ///< Mersenne twister MT19937 
  };

  enum {
    PRECISE = 100, ///< use double version of uniform with RAN_KNU
    FAST           ///< retrieve double from uniform long with RAN_KNU
  };

  enum {
    OPEN = 1000,   ///< uniform on (0,1) with RAN_MT
    ZERO,          ///< uniform on [0,1) with RAN_MT
    ZERO_ONE       ///< uniform on [0,1] with RAN_MT
  };

  lot(int type = RAN_MT, int gType = ZERO);
  ~lot(void);
  void set_generator(int type, int gType);

  /// First seed from RAN_POL
  ///
  inline long seed(void) {
    return ix;
  }

  /// Second seed from RAN_POL
  ///
  inline long init_seed(void) {
    return ix0;
  }

  void randomize (int spin = 100);
  void set_seed (long s);

  void dememorize (int spin = 100);

  void ran_start(long seed);
  void ranf_start(long seed);

  int random_int(int);
  long random_long(long maxval = Util::int_max);

  /// Uniform random integer in [0, Util::int_max-1] with RAN_KNU
  ///
  inline long ran_knu(void) {
    return *ran_arr_ptr >= 0 ? *ran_arr_ptr++ : ran_arr_cycle();
  }

  /// Uniform random number
  ///
  /// RAN_POL & RAN_KNU produce uniform on [0,1)
  /// RAN_MT produces uniform on [0,1) by default
  /// RAN_MT can produce uniform on (0,1) or [0,1]
  /// \see Set_MT
  ///
  inline double uniform(void) {
    return (this->*do_uniform)();
  }

  void ran_array(std::vector<long>& x, int n);
  void ranf_array(std::vector<double>& aa, int n);

  void MT_sgenrand(long seed);
  void MT_init_by_array(unsigned long* init_key, int key_length);
  void MT_R_initialize(int seed);
  unsigned long MT_genrand_int(void);
  // uniform (0,1): mapped to uniform() by default
  double MT_genrand(void);
  // uniform [0,1)
  double MT_genrand_with_zero(void);
  // uniform [0,1]
  double MT_genrand_with_zero_one(void);
  bool Set_MT(int type);

  double beta(double aa, double bb);
  int binom(double nin, double pp);
  double cauchy(double l, double s);
  double chisq(double n);
  std::vector<double> dirichlet(std::vector<double> c);
  double expon(void);
  /// Exponential random deviate
  ///
  /// \param lambda   The exponential parameter (\f$\lambda\f$)
  ///
  /// \f[ f(x) = \lambda e^{-\lambda x} \f]
  ///
  /// \f[ \mu = \frac{1}{\lambda} \f]
  /// \f[ \sigma^2 = \left(\frac{1}{\lambda}\right)^2 \f]
  ///
  /// Calls expon() to do the real work
  ///
  inline double exponential(const double lambda) {
    return (lambda*expon());
  }
  double f(double m, double n);
  double gamma(double a, double scale = 1.0);
  double geom(double p);
  int hypergeom(int nn1, int nn2, int kk);
  /// Inverse gamma random deviate
  ///
  /// \f[ f(1/x) = \frac{1}{s^a \Gamma(a)}x^{a-1}e^{-x/s} \f]
  ///
  /// or equivalently
  ///
  /// \f[ f(y) = \frac{\lambda^a (1/y)^{a+1} e^{-\lambda/y}}{\Gamma(a)} \f]
  /// \f[ \lambda = 1/s \f]
  ///
  /// \param a   Shape (\f$a\f$)
  /// \param s   Scale (\f$s\f$)
  ///
  /// \f[ \mbox{E}(x) = \frac{\lambda}{a-1} \quad , \quad a > 1 \f]
  /// \f[ \mbox{Var}(x) = \frac{\lambda^2}{(a-1)^2(a-2)} \quad , \quad a > 2 \f]
  ///
  inline double igamma(const double a, const double s = 1.0) {
    return 1.0/gamma(a, s);
  }
  double lnorm(double logmean, double logsd);
  double logis(double location, double scale);
  std::vector<int> multinom(unsigned n, const std::vector<double>& p);
  double nbinom(double n, double p);
  int poisson(double mu);
  /// Normal random deviate
  ///
  /// \f[ f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}} \f]
  ///
  /// \param mu   Mean (\f$\mu\f$)
  /// \param sd   Standard deviation (\f$\sigma\f$)
  ///
  /// \f[ \mbox{E}(x) = \mu \f]
  /// \f[ \mbox{Var}(x) = \sigma^2 \f]
  ///
  /// Calls snorm() to do the real work.
  ///
  inline double norm(const double mu, const double sd) {
    return sd*snorm() + mu;
  }
  double snorm(void);
  double t(double df);
  double weibull(double shape, double scale);

  /// First element of internal array in RAN_KNU
  ///
  /// Public only to allow testing of PRECISE 
  ///
  inline double get_ran_u_0(void) const {
    return ran_u[0];
  }

private:
  double POL_uniform(void);
  double POL_uniform_open(void);
  double KNU_uniform(void);
  double KNU_uniform_from_long(void);

  // The following methods are from the numerical math library in R
  // and are used by some of the RNGs derived from R.
  //
  inline double fmax2(const double x, const double y) {
    return (x < y) ? y : x;
  }
  inline double fmin2(const double x, const double y) {
    return (x < y) ? x : y;
  }
  inline int imax2(int x, int y) {
    return (x < y) ? y : x;
  }
  inline int imin2(int x, int y) {
    return (x < y) ? x : y;
  }
  inline double fsign(double x, double y) {
    return ((y >= 0) ? fabs(x) : -fabs(x));
  }

  inline long get_ran_x_0(void) const {
    return ran_x[0];
  }

  static double ppchi2(double p, double v);

  // #define mod_diff(x,y) (((x)-(y))&(MM-1))
  // subtraction mod MM
  inline long mod_diff(const long x, const long y) const {
    return ((x - y) & (MM - 1));
  }

  // #define mod_sum(x,y) (((x)+(y))-(int)((x)+(y)))
  inline double mod_sum(const double x, const double y) const {
    return (x + y) - static_cast<int>(x + y);
  }

  // #define is_odd(x)  ((s)&1)
  inline bool is_odd(const long s) const {
    return (s & 1);
  }

  long ran_arr_cycle(void);
  double ranf_arr_cycle(void);

  double fixup(double x);

  static const int QUALITY = 1009;
  static const int KK = 100;  // the long lag
  static const int LL = 37;   // the short lag
  static const int MM = (1L << 30);
  static const int TT = 70;

  bool ready;                 // true if normal deviate ready
  double y_norm;              // stored normal deviate

  double (lot::*do_uniform)(void);
  double (lot::*uniform_no_zero_generator)(void);
  int pos;
  std::vector<long> ran_arr_buf;
  std::vector<double> ranf_arr_buf;
  std::vector<long> ran_x;
  std::vector<double> ran_u;
  long* ran_arr_ptr;
  double* ranf_arr_ptr;
  long ran_arr_sentinel;
  double ranf_arr_sentinel;

  // for Mersenne twister
  // Period parameters
  static const int MT_N = 624;
  static const int MT_M = 397;

  std::vector<unsigned long> mt; // the array for the state vector
  int mti; // mti==MT_N+1 means mt[MT_N] is not initialized

  int rngType_;

  inline double sqr(const double x) {
    return x*x;
  }

  long ix0, ix;

  double e;
  double prev_alpha;
  double c1;
  double c2;
  double c3;
  double c4;
  double c5;

};

#endif

// Local Variables: //
// mode: c++ //
// End: //
