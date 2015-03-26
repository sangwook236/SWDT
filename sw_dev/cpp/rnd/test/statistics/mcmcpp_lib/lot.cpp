///
/// \file   lot.cpp
/// \brief  Random number generators
///
/// A series of random number generators are provided here. Any one of three
/// uniform random number generators can be chosen. By default the Mersenne
/// twister is used.
///
/// \author Kent Holsinger & Paul Lewis
/// \date   2005-05-18
///
/// The random number generators from R have been checked for numerical
/// accuracy with the routines in R v2.0. See lotTest.cpp for the specific
/// small set of test run. In every case the results differ from those
/// reported by R by less than 1.0e-11, and are exact for integer random
/// variables.

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
#include <cfloat>
#include <ctime>
#include <iostream>
// boost includes
#include <boost/static_assert.hpp>
// local includes
#include "mcmc++/lot.h"
#include <cmath>

//--S [] 2015/02/15 : Sang-Wook Lee
#if _MSC_VER < 1800
#include <boost/math/special_functions.hpp>
using boost::math::log1p;
#endif
//--E [] 2015/02/15 : Sang-Wook Lee

using std::cerr;
using std::endl;
using std::vector;

namespace lot_conditions {

  BOOST_STATIC_ASSERT(sizeof(long) * CHAR_BIT == 32); // 32 bit longs required

}

// defines necessary (or convenient) for functions imported from R
/// Macro for compatibility with RNGs from R
///
#define unif_rand() uniform()
/// Macro for compatibility with RNGs from R
///
#define exp_rand() expon()
/// Macro for compatibility with RNGs from R
///
#define norm_rand() snorm()
/// Macro for compatibility with RNGs from R
///
#define repeat for (;;)

namespace {

  // for Mersenne twister
  // Period parameters
  const unsigned long MATRIX_A = 0x9908b0dfUL;   /* constant vector a */
  const unsigned long UPPER_MASK = 0x80000000UL; /* most significant w-r bits */
  const unsigned long LOWER_MASK = 0x7fffffffUL; /* least significant r bits */
  // Tempering parameters
  const unsigned long TEMPERING_MASK_B = 0x9d2c5680;
  const unsigned long TEMPERING_MASK_C = 0xefc60000;
  inline long TEMPERING_SHIFT_U(const long y) {
    return (y >> 11);
  }
  inline long TEMPERING_SHIFT_S(const long y) {
    return (y << 7);
  }
  inline long TEMPERING_SHIFT_T(const long y) {
    return (y << 15);
  }
  inline long TEMPERING_SHIFT_L(const long y) {
    return (y >> 18);
  }

  // constsnts and function for Kinderman-Ramage standard normal generator
  // from R v1.9.0
  const double C1 = 0.398942280401433;
  const double C2 = 0.180025191068563;
  const double A =  2.216035867166471;
  inline double g(const double x) {
		return (C1*exp(-x*x/2.0)-C2*(A-x));
  }

  // constants for Poisson generator from R v1.9.0
  const double a0	= -0.5;
  const double a1 = 0.3333333;
  const double a2 =	-0.2500068;
  const double a3 = 0.2000118;
  const double a4	= -0.1661269;
  const double a5 = 0.1421878;
  const double a6	= -0.1384794;
  const double a7 = 0.1250060;
  const double one_7 = 0.1428571428571428571;
  const double one_12 = 0.0833333333333333333;
  const double one_24	= 0.0416666666666666667;
  const double M_1_SQRT_2PI	= 0.398942280401432677939946059934;	/* 1/sqrt(2pi) */

  // From R v2.0
#if defined(M_PI)
#undef M_PI
#endif
  const double M_PI = 3.141592653589793238462643383280;

  // computes e^x - 1 more accurately than exp(x) - 1 for small values
  // of x, i.e., x <= 0.697
  //
  // modified from R v1.9.0
  /*
   *  Mathlib : A C Library of Special Functions
   *  Copyright (C) 2002 The R Development Core Team
   *
   *  This program is free software; you can redistribute it and/or modify
   *  it under the terms of the GNU General Public License as published by
   *  the Free Software Foundation; either version 2 of the License, or
   *  (at your option) any later version.
   *
   *  This program is distributed in the hope that it will be useful,
   *  but WITHOUT ANY WARRANTY; without even the implied warranty of
   *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   *  GNU General Public License for more details.
   *
   *  You should have received a copy of the GNU General Public License
   *  along with this program; if not, write to the Free Software
   *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
   *
   *  SYNOPSIS
   *
   *	#include <Rmath.h>
   *	double expm1(double x);
   *
   *  DESCRIPTION
   *
   *	Compute the Exponential minus 1
   *
   *			exp(x) - 1
   *
   *      accurately also when x is close to zero, i.e. |x| << 1
   *
   *  NOTES
   *
   *	As log1p(), this is a standard function in some C libraries,
   *	particularly GNU and BSD (but is neither ISO/ANSI C nor POSIX).
   *
   *  We supply a substitute for the case when there is no system one.
   */
#if !defined(HAVE_EXPM1)
  double expm1(double x) {
    double y, a = fabs(x);
    if (a < Util::dbl_eps) {
      return x;
    }
    if (a > 0.697) {
      return exp(x) - 1;  /* negligible cancellation */
    }
    if (a > 1e-8) {
      y = exp(x) - 1;
    } else { /* Taylor expansion, more accurate in this range */
      y = (x / 2 + 1) * x;
    }
    /* Newton step for solving   log(1 + y) = x   for y : */
    /* WARNING: does not work for y ~ -1: bug in 1.5.0 */
    y -= (1 + y) * (log1p (y) - x);
    return y;
  }
#endif

  // From R v2.0
  //
  // afc(i) :=  ln( i! )	[logarithm of the factorial i.
  //	   If (i > 7), use Stirling's approximation, otherwise use table lookup.
  //
  double afc(const int i) {
    const double al[9] =
      {
        0.0,
        0.0,/*ln(0!)=ln(1)*/
        0.0,/*ln(1!)=ln(1)*/
        0.69314718055994530941723212145817,/*ln(2) */
        1.79175946922805500081247735838070,/*ln(6) */
        3.17805383034794561964694160129705,/*ln(24)*/
        4.78749174278204599424770093452324,
        6.57925121201010099506017829290394,
        8.52516136106541430016553103634712
        /*, 10.60460290274525022841722740072165*/
      };
    double di, value;

    // check eliminated: caller must guarantee i >= 0
    //
    //     if (i < 0) {
    //       MATHLIB_WARNING("rhyper.c: afc(i), i=%d < 0 -- SHOULD NOT HAPPEN!\n",i);
    //       return -1;/* unreached (Wall) */
    //     } else if (i <= 7) {
    if (i <= 7) {
      value = al[i + 1];
    } else {
      di = i;
      value = (di + 0.5) * log(di) - di + 0.08333333333333 / di
        - 0.00277777777777 / di / di / di + 0.9189385332;
    }
    return value;
  }

}

/// Default constructor
///
/// Uses Mersenne twister on [0,1) by default and selects integer
/// implementation of Knuth generator by default, using long -> double
/// conversion for uniform on [0,1) instead of direct calculations with
/// floating point
///
/// \param type   RAN_POL (Lewis), RAN_KNU (Knuth), RAN_MT (Mersenne Twister)
/// \param gType  RAN_KNU: PRECISE (floating point), FAST (long -> double)
///                        FAST is default because PRECISE != WITH_ZERO
///               RAN_MT: OPEN (0,1), WITH_ZERO [0,1), ZERO_ONE [0,1]
///
lot::lot(const int type, const int gType) : ix0(1L), ix(1L), mti(MT_N+1),
                                            rngType_(type),
                                            e(0.0), prev_alpha(0.0),
                                            c1(0.0), c2(0.0), c3(0.0),
                                            c4(0.0), c5(0.0)
{
  mt.resize(MT_N);
  set_generator(type, gType);
}

/// Destructor
///
/// Currently empty. Nothing to clean up
///
lot::~lot(void) {}

/// Set uniform RNG type
///
/// \param type RAN_POL, RAN_KNU, or RAN_MT
/// \param gType PRECISE or FAST (RAN_KNU)
///              DEFAULT, WITH_ZERO, or ZERO_ONE (RAN_MT)
///
void
lot::set_generator(const int type, const int gType) {
  e = exp(1.0);
  ready = false;
  time_t timer;

  switch(type) {
    case RAN_POL:
      do_uniform = &lot::POL_uniform;
      uniform_no_zero_generator = &lot::POL_uniform_open;
      randomize();
      break;
    case RAN_KNU:
      ran_arr_buf.resize(QUALITY);
      ranf_arr_buf.resize(QUALITY);
      ran_x.resize(KK);
      ran_u.resize(KK);
      if (gType == PRECISE) {
        do_uniform = &lot::KNU_uniform;
      } else {
        do_uniform = &lot::KNU_uniform_from_long;
      }
      uniform_no_zero_generator = &lot::KNU_uniform;
      ran_start(static_cast<long>(time(&timer) % (MM - 2)));
      ranf_start(static_cast<long>(time(&timer)));
      ran_arr_sentinel = -1;
      ranf_arr_sentinel = -1.0;
      ran_arr_ptr = &ran_arr_sentinel;
      ranf_arr_ptr = &ranf_arr_sentinel;
      break;
    case RAN_MT:
      Set_MT(gType);
      uniform_no_zero_generator = &lot::MT_genrand;
      MT_sgenrand(time(&timer));
      break;
    default:
      cerr << "Unrecognized generator type!" << endl;
      exit(1);
  }
}

/// Used with RAN_POL to "warm up" generator
///
/// \param spin number of preliminary calls to uniform()
///
void
lot::dememorize (int spin /* = 100 */ ) {
  for (int k = 0; k < spin; k++)
    uniform();
}

/// Initializes RAN_POL
///
/// \param spin (default 100) passed to dememorize()
///
/// Initializes seeds with current system time and calls dememorize to
/// "warm up" random number generator
///
void
lot::randomize (int spin /* = 100 */ ) {
  time_t timer;
  ix = ix0 = static_cast<long>(time(&timer));
  dememorize (spin);
}

/// Initialize RNG with known seed
///
/// \param s   Seed
///
void
lot::set_seed(const long s) {
  switch(rngType_) {
    case RAN_POL:
      ix = ix0 = s;
      break;
    case RAN_KNU:
      ran_start(s % (MM - 2));
      ranf_start(s);
      break;
    case RAN_MT:
      MT_sgenrand(s);
      break;
    default:
      cerr << "Unrecognized generator type!" << endl;
      exit(1);
  }
}

/// Set MT type
///
/// \param type  DEFAULT (0,1), ZERO [0,1), ZERO_ONE [0,1]
///
/// Leaves MT type unchanged if type is not one of DEFAULT, ZERO, or
/// ZERO_ONE. Changes from RAN_POL or RAN_KNU to RAN_MT.
///
bool
lot::Set_MT(const int type) {
  bool retval = true;
  switch (type) {
    case OPEN:
      do_uniform = &lot::MT_genrand;
      rngType_ = lot::RAN_MT;
      break;
    case ZERO:
      do_uniform = &lot::MT_genrand_with_zero;
      rngType_ = lot::RAN_MT;
      break;
    case ZERO_ONE:
      do_uniform = &lot::MT_genrand_with_zero_one;
      rngType_ = lot::RAN_MT;
      break;
    default:
      retval = false;
      break;
  }
  return retval;
}

/// Double implementation of Knuth generator
///
/// From the source to rng-double.c:
///
///   This program by D E Knuth is in the public domain and freely copyable
///   AS LONG AS YOU MAKE ABSOLUTELY NO CHANGES!
///   It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
///  (or in the errata to the 2nd edition --- see
///   http://www-cs-faculty.stanford.edu/~knuth/taocp.html
///  in the changes to Volume 2 on pages 171 and following).
///
///   N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
///   included here; there's no backwards compatibility with the original.
///
///   If you find any bugs, please report them immediately to
///   taocp@cs.stanford.edu
///   (and you will be rewarded if the bug is genuine). Thanks!
///
/// see the book for explanations and caveats!
/// particular, you need two's complement arithmetic
///
/// Note: This method is private.
///
double
lot::KNU_uniform(void) {
  return *ranf_arr_ptr >= 0 ? *ranf_arr_ptr++ : ranf_arr_cycle();
}

/// Updates floating point array of Knuth generator
///
/// \param aa  the vector of values in which to update
/// \param n   the number of values to update, note n == aa.size() assumed
///
void
lot::ranf_array(std::vector<double>& aa, int n) {
  int i, j;

  for (j = 0; j < KK; j++) {
    aa[j] = ran_u[j];
  }

  for (; j < n; j++) {
    aa[j] = mod_sum(aa[j - KK], aa[j - LL]);
  }

  for (i = 0; i < LL; i++, j++) {
    ran_u[i] = mod_sum(aa[j - KK], aa[j - LL]);
  }

  for (; i < KK; i++, j++) {
    ran_u[i] = mod_sum(aa[j - KK], ran_u[i - LL]);
  }
}

/// Cycle the Knuth double RNG
///
double
lot::ranf_arr_cycle(void) {
  ranf_array(ranf_arr_buf, QUALITY);
  ranf_arr_buf[100] = -1;
  ranf_arr_ptr = &ranf_arr_buf[1];
  return ranf_arr_buf[0];
}

/// Initialize the Knuth double RNG
///
void
lot::ranf_start(const long seed) {
  int t, s, j;
  vector<double> u(KK + KK - 1);
  double ulp = (1.0 / (1L << 30)) / (1L << 22);               /* 2 to the -52 */
  double ss = 2.0 * ulp * ((seed & 0x3fffffff) + 2);

  for (j = 0;j < KK;j++) {
    u[j] = ss;                                /* bootstrap the buffer */
    ss += ss;

    if (ss >= 1.0)
      ss -= 1.0 - 2 * ulp;  /* cyclic shift of 51 bits */
  }

  u[1] += ulp;                     /* make u[1] (and only u[1]) "odd" */

  for (s = seed & 0x3fffffff, t = TT - 1; t; ) {
    for (j = KK - 1;j > 0;j--)
      u[j + j] = u[j], u[j + j - 1] = 0.0;                         /* "square" */

    for (j = KK + KK - 2;j >= KK;j--) {
      u[j - (KK - LL)] = mod_sum(u[j - (KK - LL)], u[j]);
      u[j - KK] = mod_sum(u[j - KK], u[j]);
    }

    if (is_odd(s)) {                             /* "multiply by z" */

      for (j = KK;j > 0;j--)
        u[j] = u[j - 1];

      u[0] = u[KK];                    /* shift the buffer cyclically */

      u[LL] = mod_sum(u[LL], u[KK]);
    }

    if (s)
      s >>= 1;
    else
      t--;
  }

  for (j = 0;j < LL;j++)
    ran_u[j + KK - LL] = u[j];

  for (;j < KK;j++)
    ran_u[j - LL] = u[j];

  for (j = 0;j < 10;j++)
    ranf_array(u, KK + KK - 1);  /* warm things up */

  ranf_arr_ptr = &ranf_arr_sentinel;
}

/// long implementation of Knuth generator
///
/// From the source to rng.c
///
/// This program by D E Knuth is in the public domain and freely copyable
/// AS LONG AS YOU MAKE ABSOLUTELY NO CHANGES!
/// It is explained in Seminumerical Algorithms, 3rd edition, Section 3.6
/// (or in the errata to the 2nd edition --- see
/// http://www-cs-faculty.stanford.edu/~knuth/taocp.html
/// in the changes to Volume 2 on pages 171 and following).
///
/// N.B. The MODIFICATIONS introduced in the 9th printing (2002) are
/// included here; there's no backwards compatibility with the original.
///
/// If you find any bugs, please report them immediately to
/// taocp@cs.stanford.edu
/// and you will be rewarded if the bug is genuine). Thanks!
///
/// see the book for explanations and caveats!
/// in particular, you need two's complement arithmetic
///
/// NOTE: Knuth is not responsible for the implementation of
/// KNU_uniform_from_long(). If there's an error in that it's my fault.
///
/// Note: This method is private.
///
double
lot::KNU_uniform_from_long(void) {
  return ran_knu() / static_cast<double>(MM);
}


/// Updates long array of Knuth generator
///
/// \param aa  the vector of values in which to update
/// \param n   the number of values to update, note n == aa.size() assumed
///
void
lot::ran_array(std::vector<long>& aa, const int n) {
  int i, j;

  for (j = 0; j < KK; j++) {
    aa[j] = ran_x[j];
  }

  for (; j < n; j++) {
    aa[j] = mod_diff(aa[j - KK], aa[j - LL]);
  }

  for (i = 0; i < LL; i++, j++) {
    ran_x[i] = mod_diff(aa[j - KK], aa[j - LL]);
  }

  for (; i < KK; i++, j++) {
    ran_x[i] = mod_diff(aa[j - KK], ran_x[i - LL]);
  }
}

/// Cycle the Knuth long RNG
///
long
lot::ran_arr_cycle() {
  ran_array(ran_arr_buf, QUALITY);
  ran_arr_buf[100] = -1;
  ran_arr_ptr = &ran_arr_buf[1];
  return ran_arr_buf[0];
}

/// Initialize the Knuth long RNG
///
void
lot::ran_start(const long seed) {
  int t, j;
  vector<long> x(KK + KK - 1);  // the preparation buffer
  long ss = (seed + 2) & (MM - 2);

  for (j = 0;j < KK;j++) {
    x[j] = ss;                      /* bootstrap the buffer */
    ss <<= 1;

    if (ss >= MM)
      ss -= MM - 2; /* cyclic shift 29 bits */
  }

  x[1]++;              /* make x[1] (and only x[1]) odd */

  for (ss = seed & (MM - 1), t = TT - 1; t; ) {
    for (j = KK - 1;j > 0;j--)
      x[j + j] = x[j], x[j + j - 1] = 0; /* "square" */

    for (j = KK + KK - 2;j >= KK;j--)
      x[j - (KK - LL)] = mod_diff(x[j - (KK - LL)], x[j]),
                         x[j - KK] = mod_diff(x[j - KK], x[j]);

    if (is_odd(ss)) {              /* "multiply by z" */

      for (j = KK;j > 0;j--)
        x[j] = x[j - 1];

      x[0] = x[KK];            /* shift the buffer cyclically */

      x[LL] = mod_diff(x[LL], x[KK]);
    }

    if (ss)
      ss >>= 1;
    else
      t--;
  }

  for (j = 0;j < LL;j++)
    ran_x[j + KK - LL] = x[j];

  for (;j < KK;j++)
    ran_x[j - LL] = x[j];

  for (j = 0;j < 10;j++)
    ran_array(x, KK + KK - 1); /* warm things up */

  ran_arr_ptr = &ran_arr_sentinel;
}

/// Paul Lewis' RNG
///
/// Uniform pseudorandom number generator
/// Provided by J. Monahan, Statistics Dept., N.C. State University
/// From Schrage, ACM Trans. Math. Software 5:132-138 (1979)
/// Translated to C by Paul O. Lewis, Dec. 10, 1992
///
/// Note: this method is private
///
double
lot::POL_uniform() {
  long a, p, b15, b16, xhi, xalo, leftlo, fhi, k;

  a = 16807L;
  b15 = 32768L;
  b16 = 65536L;
  p = 2147483647L;
  xhi = ix / b16;
  xalo = (ix - xhi * b16) * a;
  leftlo = xalo / b16;
  fhi = xhi * a + leftlo;
  k = fhi / b15;
  ix = (((xalo - leftlo * b16) - p) + (fhi - k * b15) * b16) + k;

  if (ix < 0) {
    ix += p;
  }
  return ix * 4.6566128575e-10;
}

/// Paul's RNG modified to ensure non-zero return
/// Note: this method is private
///
double
lot::POL_uniform_open() {
  long a, p, b15, b16, xhi, xalo, leftlo, fhi, k;

  do {
    a = 16807L;
    b15 = 32768L;
    b16 = 65536L;
    p = 2147483647L;
    xhi = ix / b16;
    xalo = (ix - xhi * b16) * a;
    leftlo = xalo / b16;
    fhi = xhi * a + leftlo;
    k = fhi / b15;
    ix = (((xalo - leftlo * b16) - p) + (fhi - k * b15) * b16) + k;

    if (ix < 0) {
      ix += p;
    }
  } while (ix == 0);
  return ix * 4.6566128575e-10;
}

/// Initializes Mersenne twister RNG
///
/// \param seed
///
/// This code is translated directly from:
///
///   http://www.math.keio.ac.jp/matumoto/CODES/MT2002/mt19937ar.c
///
/// Here are the accompanying comments
///
/// A C-program for MT19937, with initialization improved 2002/1/26.
/// Coded by Takuji Nishimura and Makoto Matsumoto.
///
/// Before using, initialize the state by using init_genrand(seed)
/// or init_by_array(init_key, key_length).
///
/// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
/// All rights reserved.
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// 1. Redistributions of source code must retain the above copyright
///    notice, this list of conditions and the following disclaimer.
///
/// 2. Redistributions in binary form must reproduce the above copyright
///    notice, this list of conditions and the following disclaimer in the
///    documentation and/or other materials provided with the distribution.
///
/// 3. The names of its contributors may not be used to endorse or promote
///    products derived from this software without specific prior written
///    permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
/// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
/// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
/// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
/// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
/// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
/// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
/// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
/// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///
/// Any feedback is very welcome.
/// http://www.math.keio.ac.jp/matumoto/emt.html
/// email: matumoto@math.keio.ac.jp
///
void
lot::MT_sgenrand(long seed) {
  mt[0]= seed & 0xffffffffUL;
  for (mti = 1; mti < MT_N; mti++) {
    mt[mti] =
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
}

/// Initialize Mersenne twister with an array
///
/// \param init_key is the array for initializing keys
/// \param key_length is its length
///
void
lot::MT_init_by_array(unsigned long* init_key, int key_length) {
  int i, j, k;
  MT_sgenrand(19650218UL);
  i=1;
  j=0;
  k = (MT_N > key_length ? MT_N : key_length);
  for (; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
      + init_key[j] + j; /* non linear */
    mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
    i++;
    j++;
    if (i >= MT_N) {
      mt[0] = mt[MT_N-1]; i=1;
    }
    if (j >= key_length) {
      j=0;
    }
  }
  for (k = MT_N-1; k; k--) {
    mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
      - i; /* non linear */
    mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
    i++;
    if (i >= MT_N) {
      mt[0] = mt[MT_N-1];
      i=1;
    }
  }

  mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/// Initialize array directly with algorithm from R
///
/// \param seed
///
/// Probably useful only for testing purposes. I wrote it to allow me to
/// check norm(), binom(), beta(), etc. against R. It only produces the
/// same sequence as R when seed == 1.
///
void
lot::MT_R_initialize(int seed) {
  for (int j = 0; j < 50; ++j) {
    seed = (69069*seed + 1);
  }
  for (int j = 0; j < MT_N; ++j) {
    seed = (69069*seed + 1);
    mt[j] = seed;
  }
}

/// generate random integer on [0,0xffffffff]-interval
///
unsigned long
lot::MT_genrand_int(void) {
    unsigned long y;
    static unsigned long mag01[2]={0x0, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= MT_N) { /* generate MT_N words at one time */
      int kk;

      if (mti == MT_N+1)   /* if sgenrand() has not been called, */
        MT_sgenrand(5489UL); /* a default initial seed is used   */

      for (kk = 0; kk < MT_N - MT_M; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
        mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
      }
      for (; kk < MT_N - 1; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk+1] & LOWER_MASK);
        mt[kk] = mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
      }
      y = (mt[MT_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

      mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/// generate random double uniform on (0,1)
///
double
lot::MT_genrand(void) {
  return (static_cast<double>(MT_genrand_int()) + 0.5)*(1.0/4294967296.0);
}

/// generate random double uniform on [0,1)
///
double
lot::MT_genrand_with_zero(void) {
  return MT_genrand_int()*(1.0/4294967296.0);
}

/// generate random double uniform on [0,1]
///
double
lot::MT_genrand_with_zero_one(void) {
  return MT_genrand_int()*(1.0/4294967295.0);
}


/// Returns a random long in [0,maxval-1]
///
/// \param maxval
///
long
lot::random_long (long maxval) {
  long return_val = maxval;

  while (return_val == maxval) {
    return_val = static_cast<long>(floor((maxval * uniform())));
  }

  return return_val;
}

/// Returns a random int in [0,maxval-1]
///
/// \param maxval
///
int
lot::random_int (int maxval) {
  int return_val = maxval;

  while (return_val == maxval) {
    double r = uniform();
    return_val = static_cast<int>(floor(maxval * r));
  }

  return return_val;
}

#ifdef M_LN2
#define expmax (DBL_MAX_EXP * M_LN2)/* = log(DBL_MAX) */
#else
#define expmax  log(DBL_MAX)
#endif

/// Returns a random beta variate
///
/// \f[f(x) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1} \f]
///
/// \param aa   The first beta parameter (\f$a\f$)
/// \param bb   The second beta parameter (\f$b\f$)
///
/// Returns a random variable from a beta distribution with parameters
/// aa and bb.
///
/// NOTE: Checks for aa, bb > 0 not included
///
/// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
/// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
/// if you are using the Mersenne-Twister. ZERO is the default.
///
/// This implementation is derived from R v1.9.0
///
//  R : A Computer Language for Statistical Data Analysis
//  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
// Reference:
// R. C. H. Cheng (1978).
// Generating beta variates with nonintegral shape parameters.
// Communications of the ACM 21, 317-322.
// (Algorithms BB and BC)
//
double
lot::beta(const double aa, const double bb) {
  double a, b, alpha;
  double r, s, t, u1, u2, v, w, y, z;

  int qsame;
  /* FIXME:  Keep Globals (properly) for threading */
  /* Uses these GLOBALS to save time when many rv's are generated : */
  static double beta, gamma, delta, k1, k2;
  static double olda = -1.0;
  static double oldb = -1.0;
  /* Test if we need new "initializing" */
  qsame = (olda == aa) && (oldb == bb);
  if (!qsame) {
    olda = aa;
    oldb = bb;
  }
  a = fmin2(aa, bb);
  b = fmax2(aa, bb); /* a <= b */
  alpha = a + b;
#define v_w_from__u1_bet(AA)   \
     v = beta * log(u1 / (1.0 - u1)); \
     if (v <= expmax)   \
  w = AA * exp(v);  \
     else    \
  w = DBL_MAX
  if (a <= 1.0) { /* --- Algorithm BC --- */
    /* changed notation, now also a <= b (was reversed) */
    if (!qsame) { /* initialize */
      beta = 1.0 / a;
      delta = 1.0 + b - a;
      k1 = delta * (0.0138889 + 0.0416667 * a) / (b * beta - 0.777778);
      k2 = 0.25 + (0.5 + 0.25 / delta) * a;
    }
    /* FIXME: "do { } while()", but not trivially because of "continue"s:*/
    for (;;) {
      u1 = unif_rand();
      u2 = unif_rand();
      if (u1 < 0.5) {
        y = u1 * u2;
        z = u1 * y;
        if (0.25 * u2 + z - y >= k1)
          continue;
      } else {
        z = u1 * u1 * u2;
        if (z <= 0.25) {
          v_w_from__u1_bet(b);
          break;
        }
        if (z >= k2)
          continue;
      }
      v_w_from__u1_bet(b);
      if (alpha * (log(alpha / (a + w)) + v) - 1.3862944 >= log(z))
        break;
    }
    return (aa == a) ? a / (a + w) : w / (a + w);
  } else {  /* Algorithm BB */
    if (!qsame) { /* initialize */
      beta = sqrt((alpha - 2.0) / (2.0 * a * b - alpha));
      gamma = a + 1.0 / beta;
    }
    do {
      u1 = unif_rand();
      u2 = unif_rand();
      v_w_from__u1_bet(a);
      z = u1 * u1 * u2;
      r = gamma * v - 1.3862944;
      s = a + r - w;
      if (s + 2.609438 >= 5.0 * z)
        break;
      t = log(z);
      if (s > t)
        break;
    } while (r + alpha * log(alpha / (b + w)) < t);
    return (aa != a) ? b / (b + w) : w / (b + w);
  }
}

/// Returns a random binomial variate
///
/// \f[ f(x) = {n \choose k}p^k(1-p)^{n-k} \f]
///
/// \param nin  sample size (\f$n\f$)
/// \param pp   probability of success on each trial (\f$p\f$)
///
/// \f[ \mbox{E}(x) = np \f]
/// \f[ \mbox{Var}(x) = np(1-p) \f]
///
/// This implementation is derived from R v1.9.0.
///
/// NOTE: Checks for finite nin and pp not included.
///       Check for nin == floor(n+0.5) not included
///
// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
// if you are using the Mersenne-Twister. ZERO is the default.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
// #include "Rmath.h"
// double rbinom(double nin, double pp)
//
//  DESCRIPTION
//
// Random variates from the binomial distribution.
//
//  REFERENCE
//
// Kachitvichyanukul, V. and Schmeiser, B. W. (1988).
// Binomial random variate generation.
// Communications of the ACM 31, p216.
// (Algorithm BTPEC).
//
int
lot::binom(const double nin, const double pp) {
  static double c, fm, npq, p1, p2, p3, p4, qn;
  static double xl, xll, xlr, xm, xr;
  static double psave = -1.0;
  static int nsave = -1;
  static int m;
  double f, f1, f2, u, v, w, w2, x, x1, x2, z, z2;
  double p, q, np, g, r, al, alv, amaxp, ffm, ynorm;
  int i,ix,k, n;

  n = static_cast<int>(floor(nin + 0.5));
  if (n == 0 || pp == 0.) return 0;
  if (pp == 1.) return n;
  p = fmin2(pp, 1. - pp);
  q = 1. - p;
  np = n * p;
  r = p / q;
  g = r * (n + 1);

  /* Setup, perform only when parameters change [using static (globals): */
  /* FIXING: Want this thread safe
     -- use as little (thread globals) as possible
  */
  if (pp != psave || n != nsave) {
    psave = pp;
    nsave = n;
    if (np < 30.0) {
	    /* inverse cdf logic for mean less than 30 */
	    qn = pow(q, static_cast<double>(n));
	    goto L_np_small;
    } else {
	    ffm = np + p;
	    m = static_cast<int>(ffm);
	    fm = m;
	    npq = np * q;
	    p1 = static_cast<int>(2.195 * sqrt(npq) - 4.6 * q) + 0.5;
	    xm = fm + 0.5;
	    xl = xm - p1;
	    xr = xm + p1;
	    c = 0.134 + 20.5 / (15.3 + fm);
	    al = (ffm - xl) / (ffm - xl * p);
	    xll = al * (1.0 + 0.5 * al);
	    al = (xr - ffm) / (xr * q);
	    xlr = al * (1.0 + 0.5 * al);
	    p2 = p1 * (1.0 + c + c);
	    p3 = p2 + c / xll;
	    p4 = p3 + c / xlr;
    }
  } else if (n == nsave) {
    if (np < 30.0)
	    goto L_np_small;
  }
  /*-------------------------- np = n*p >= 30 : ------------------- */
  repeat {
    u = unif_rand() * p4;
    v = unif_rand();
    /* triangular region */
    if (u <= p1) {
      ix = static_cast<int>(xm - p1 * v + u);
      goto finis;
    }
    /* parallelogram region */
    if (u <= p2) {
      x = xl + (u - p1) / c;
      v = v * c + 1.0 - fabs(xm - x) / p1;
      if (v > 1.0 || v <= 0.)
	      continue;
      ix = static_cast<int>(x);
    } else {
      if (u > p3) {	/* right tail */
	      ix = static_cast<int>(xr - log(v) / xlr);
	      if (ix > n)
          continue;
	      v = v * (u - p3) * xlr;
      } else {/* left tail */
	      ix = static_cast<int>(xl + log(v) / xll);
	      if (ix < 0)
          continue;
	      v = v * (u - p2) * xll;
      }
    }
    /* determine appropriate way to perform accept/reject test */
    k = abs(ix - m);
    if (k <= 20 || k >= npq / 2 - 1) {
      /* explicit evaluation */
      f = 1.0;
      if (m < ix) {
	      for (i = m + 1; i <= ix; i++)
          f *= (g / i - r);
      } else if (m != ix) {
	      for (i = ix + 1; i <= m; i++)
          f /= (g / i - r);
      }
      if (v <= f)
	      goto finis;
    } else {
      /* squeezing using upper and lower bounds on log(f(x)) */
      amaxp = (k / npq) * ((k * (k / 3. + 0.625) + 0.1666666666666) / npq + 0.5);
      ynorm = -k * k / (2.0 * npq);
      alv = log(v);
      if (alv < ynorm - amaxp)
	      goto finis;
      if (alv <= ynorm + amaxp) {
	      /* stirling's formula to machine accuracy */
	      /* for the final acceptance/rejection test */
	      x1 = ix + 1;
	      f1 = fm + 1.0;
	      z = n + 1 - fm;
	      w = n - ix + 1.0;
	      z2 = z * z;
	      x2 = x1 * x1;
	      f2 = f1 * f1;
	      w2 = w * w;
	      if (alv <= xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) + (ix - m) * log(w * p / (x1 * q)) + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) / f1 / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) / z / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) / x1 / 166320.0 + (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) / w / 166320.)
          goto finis;
      }
    }
  }
 L_np_small:
  /*---------------------- np = n*p < 30 : ------------------------- */
  repeat {
    ix = 0;
    f = qn;
    u = unif_rand();
    repeat {
      if (u < f)
        goto finis;
      if (ix > 110)
        break;
      u -= f;
      ix++;
      f *= (g / ix - r);
    }
  }
 finis:
  if (psave > 0.5) {
    ix = n - ix;
  }
  return ix;
}

/// Returns a Cauchy variate
///
/// \f[ f(x) = \frac{1}{\pi\mbox{s} (1 + (\frac{x-\mbox{l}}{\mbox{s}})^2)} \f]
///
/// \param l         the location parameter
/// \param s     the scale parameter
///
/// The expectation and variance of the Cauchy distribution are infinite.
/// The mode is equal to the location parameter.
///
/// Modified from R v2.0. Does not check isnan() on arguments. Does not
/// check that arguments are finite
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rcauchy(double location, double scale);
//
//  DESCRIPTION
//
//    Random variates from the Cauchy distribution.
//
double
lot::cauchy(const double l, const double s) {
  return l + s * tan(M_PI * unif_rand());
}

/// Returns a chi-squared variate
///
/// \f[ f(x) = \frac{1}{2^{n/2}\Gamma(n/2)}x^{n/2-1}e^{-x/2} \f]
///
/// \param n         degrees of freedom for the chi-squared density
///
/// \f[ \mbox{E}(x) = \mbox{n} \f]
/// \f[ \mbox{Var}(x) = 2\mbox{n} \f]
///
/// Derived from R v2.0. Does not check isfinite() on argument.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rchisq(double df);
//
//  DESCRIPTION
//
//    Random variates from the chi-squared distribution.
//
//  NOTES
//
//    Calls rgamma to do the real work.
double
lot::chisq(const double n) {
    return gamma(n / 2.0, 2.0);
}


/// Returns a vector of Dirichlet variates
///
/// \f[ f({\bf x}) = \frac{\Gamma(\sum_i x_i)}{\prod_i\Gamma(x_i)}\prod_ix_i^{c_i} \f]
///
/// \param c [vector<double>] parameters of the Dirichlet
///
std::vector<double>
lot::dirichlet(std::vector<double> c) {
  int n = c.size();
  vector<double> p(n);
  for (;;) {
    DblVect g(n);
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      double gammadev = gamma(c[i]);
      g[i] = gammadev;
      sum += gammadev;
    }
    bool ok = true;
    for (int i = 0; i < n; i++) {
      p[i] = g[i] / sum;
      // if any of the p's are effectively zero,
      // we must abort and try again
      if (p[i] < DBL_MIN) {
        ok = false;
        break;
      }
    }
    if (ok) {
      break;
    }
  }
  return p;
}

/// Returns a random value from an exponential distribution with mean 1.
///
/// \f[ f(x) = e^{-1} \f]
///
/// \f[ \mbox{E}(x) = 1 \f]
/// \f[ \mbox{Var}(x) = 1 \f]
///
/// originally derived from R v1.8.1
///
/// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
/// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
/// if you are using the Mersenne-Twister. ZERO is the default.
///
// Mathlib : A C Library of Special Functions
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
// SYNOPSIS
//
//   #include <Rmath.h>
//   double exp_rand(void);
//
// DESCRIPTION
//
//   Random variates from the standard exponential distribution.
//
// REFERENCE
//
//   Ahrens, J.H. and Dieter, U. (1972).
//   Computer methods for sampling from the exponential and
//   normal distributions.
//   Comm. ACM, 15, 873-882.
//
double
lot::expon(void) {
  /* q[k-1] = sum(log(2)^k / k!)  k=1,..,n, */
  /* The highest n (here 8) is determined by q[n-1] = 1.0 */
  /* within standard precision */
  const double q[] =
    {
      0.6931471805599453,
      0.9333736875190459,
      0.9888777961838675,
      0.9984959252914960,
      0.9998292811061389,
      0.9999833164100727,
      0.9999985691438767,
      0.9999998906925558,
      0.9999999924734159,
      0.9999999995283275,
      0.9999999999728814,
      0.9999999999985598,
      0.9999999999999289,
      0.9999999999999968,
      0.9999999999999999,
      1.0000000000000000
    };
  double a, u, ustar, umin;
  int i;
  a = 0.;
  /* precaution if u = 0 is ever returned */
  u = uniform();
  while(u <= 0.0 || u >= 1.0) u = uniform();
  for (;;) {
    u += u;
    if (u > 1.0) {
	    break;
    }
    a += q[0];
  }
  u -= 1.;
  if (u <= q[0])
    return a + u;
  i = 0;
  ustar = uniform();
  umin = ustar;
  do {
    ustar = uniform();
    if (ustar < umin) {
	    umin = ustar;
    }
    i++;
  } while (u > q[i]);
  return a + umin * q[0];
}

/// Returns an F variate
///
/// \f[ f(x) = \frac{\Gamma((m+n)/2)}{\Gamma(m/2)\Gamma(n/2)}(m/n)^{m/2}x^{m/2-1}(1+(m/n)x)^{-(m+n)/2} \f]
///
/// \param m        ``numerator'' degrees of freedom
/// \param n        ``denominator'' degrees of freedom
///
/// \f[ \mbox{E}(x) = \frac{m}{m-2}, m > 2 \f]
/// \f[ \mbox{Var}(x) = \frac{2m^2(n-2)}{n(m+2)}, n > 2 \f]
///
/// Derived from R v2.0. Does not check arguments for isnan() or isfinite().
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include "mathlib.h"
//    double rf(double dfn, double dfd);
//
//  DESCRIPTION
//
//    Pseudo-random variates from an F distribution.
//
//  NOTES
//
//    This function calls rchisq to do the real work
//
double
lot::f(const double m, const double n) {
    double v1, v2;
    v1 = chisq(m)/m;
    v2 = chisq(n)/n;
    return v1/v2;
}

/// Gamma random deviate
///
/// \param a       Shape
/// \param scale   Scale (\f$\sigma\f$, defaults to 1.0)
///
/// \f[ f(x) = \frac{1}{\sigma^a \Gamma(a)}x^{a-1}e^{-x/\sigma} \f]
///
/// \f[ \mu = a\sigma \f]
/// \f[ \sigma^2 = a\sigma^2 \f]
///
/// NOTE: Checks for finite scale and shape parameters not included
///
/// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
/// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
/// if you are using the Mersenne-Twister. ZERO is the default.
///
/// Derived from R v1.9.0
///
// Mathlib : A C Library of Special Functions
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
// SYNOPSIS
//
//   #include <Rmath.h>
//   double rgamma(double a, double scale);
//
// DESCRIPTION
//
//   Random variates from the gamma distribution.
//
// REFERENCES
//
//  [1] Shape parameter a >= 1.  Algorithm GD in:
//
//  Ahrens, J.H. and Dieter, U. (1982).
//  Generating gamma variates by a modified
//  rejection technique.
//  Comm. ACM, 25, 47-54.
//
//
//  [2] Shape parameter 0 < a < 1. Algorithm GS in:
//
//  Ahrens, J.H. and Dieter, U. (1974).
//  Computer methods for sampling from gamma, beta,
//  poisson and binomial distributions.
//  Computing, 12, 223-246.
//
//   Input: a = parameter (mean) of the standard gamma distribution.
//   Output: a variate from the gamma(a)-distribution
//
double
lot::gamma(double a, double scale) {
  /* Constants : */
  const double sqrt32 = 5.656854;
  const double exp_m1 = 0.36787944117144232159;/* exp(-1) = 1/e */

  /* Coefficients q[k] - for q0 = sum(q[k]*a^(-k))
   * Coefficients a[k] - for q = q0+(t*t/2)*sum(a[k]*v^k)
   * Coefficients e[k] - for exp(q)-1 = sum(e[k]*q^k)
   */
  const double q1 = 0.04166669;
  const double q2 = 0.02083148;
  const double q3 = 0.00801191;
  const double q4 = 0.00144121;
  const double q5 = -7.388e-5;
  const double q6 = 2.4511e-4;
  const double q7 = 2.424e-4;

  const double a1 = 0.3333333;
  const double a2 = -0.250003;
  const double a3 = 0.2000062;
  const double a4 = -0.1662921;
  const double a5 = 0.1423657;
  const double a6 = -0.1367177;
  const double a7 = 0.1233795;

  /* State variables [FIXME for threading!] :*/
  static double aa = 0.;
  static double aaa = 0.;
  static double s, s2, d;    /* no. 1 (step 1) */
  static double q0, b, si, c;/* no. 2 (step 4) */

  double e, p, q, r, t, u, v, w, x, ret_val;

  if (a < 1.) { /* GS algorithm for parameters a < 1 */
    e = 1.0 + exp_m1 * a;
    repeat {
	    p = e * unif_rand();
	    if (p >= 1.0) {
        x = -log((e - p) / a);
        if (exp_rand() >= (1.0 - a) * log(x))
          break;
	    } else {
        x = exp(log(p) / a);
        if (exp_rand() >= x)
          break;
	    }
    }
    return scale * x;
  }

  /* --- a >= 1 : GD algorithm --- */
  /* Step 1: Recalculations of s2, s, d if a has changed */
  if (a != aa) {
    aa = a;
    s2 = a - 0.5;
    s = sqrt(s2);
    d = sqrt32 - s * 12.0;
  }
  /* Step 2: t = standard normal deviate,
     x = (s,1/2) -normal deviate. */
  /* immediate acceptance (i) */
  t = norm_rand();
  x = s + 0.5 * t;
  ret_val = x * x;
  if (t >= 0.0) {
    return scale * ret_val;
  }
  /* Step 3: u = 0,1 - uniform sample. squeeze acceptance (s) */
  u = unif_rand();
  if (d * u <= t * t * t) {
    return scale * ret_val;
  }
  /* Step 4: recalculations of q0, b, si, c if necessary */
  if (a != aaa) {
    aaa = a;
    r = 1.0 / a;
    q0 = ((((((q7 * r + q6) * r + q5) * r + q4) * r + q3) * r
           + q2) * r + q1) * r;
    /* Approximation depending on size of parameter a */
    /* The constants in the expressions for b, si and c */
    /* were established by numerical experiments */
    if (a <= 3.686) {
	    b = 0.463 + s + 0.178 * s2;
	    si = 1.235;
	    c = 0.195 / s - 0.079 + 0.16 * s;
    } else if (a <= 13.022) {
	    b = 1.654 + 0.0076 * s2;
	    si = 1.68 / s + 0.275;
	    c = 0.062 / s + 0.024;
    } else {
	    b = 1.77;
	    si = 0.75;
	    c = 0.1515 / s;
    }
  }
  /* Step 5: no quotient test if x not positive */
  if (x > 0.0) {
    /* Step 6: calculation of v and quotient q */
    v = t / (s + s);
    if (fabs(v) <= 0.25) {
	    q = q0 + 0.5 * t * t * ((((((a7 * v + a6) * v + a5) * v + a4) * v
                                + a3) * v + a2) * v + a1) * v;
    } else {
	    q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);

    }
    /* Step 7: quotient acceptance (q) */
    if (log(1.0 - u) <= q) {
	    return scale * ret_val;
    }
  }
  repeat {
    /* Step 8: e = standard exponential deviate
     *	u =  0,1 -uniform deviate
     *	t = (b,si)-double exponential (laplace) sample */
    e = exp_rand();
    u = unif_rand();
    u = u + u - 1.0;
    if (u < 0.0) {
	    t = b - si * e;
    } else {
	    t = b + si * e;
    }
    /* Step	 9:  rejection if t < tau(1) = -0.71874483771719 */
    if (t >= -0.71874483771719) {
	    /* Step 10:	 calculation of v and quotient q */
	    v = t / (s + s);
	    if (fabs(v) <= 0.25) {
        q = q0 + 0.5 * t * t *
          ((((((a7 * v + a6) * v + a5) * v + a4) * v + a3) * v
            + a2) * v + a1) * v;
	    } else {
        q = q0 - s * t + 0.25 * t * t + (s2 + s2) * log(1.0 + v);
      }
	    /* Step 11:	 hat acceptance (h) */
	    /* (if q not positive go to step 8) */
	    if (q > 0.0) {
        w = expm1(q);
        /*  ^^^^^ original code had approximation with rel.err < 2e-7 */
        /* if t is rejected sample again at step 8 */
        if (c * fabs(u) <= w * exp(e - 0.5 * t * t)) {
          break;
        }
	    }
    }
  } /* repeat .. until  `t' is accepted */
  x = s + 0.5 * t;
  return scale * x * x;
}

/// Returns a geometric deviate
///
/// \f[ f(x) = p(1-p)^x \f]
///
/// \param p         the parameter of the geometric distribution
///
/// \f[ \mbox{E}(x) = \frac{1-p}{p} \f]
/// \f[ \mbox{Var}(x) = \frac{1-p}{p^2} \f]
///
/// Derived from R v2.0. Does not check isnan() on x and p. Does not
/// check for 0 < p < 1.
///
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka and the R Development Core Team.
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rgeom(double p);
//
//  DESCRIPTION
//
//    Random variates from the geometric distribution.
//
//  NOTES
//
//    We generate lambda as exponential with scale parameter
//    p / (1 - p).  Return a Poisson deviate with mean lambda.
//
//  REFERENCE
//
//    Devroye, L. (1986).
//    Non-Uniform Random Variate Generation.
//    New York: Springer-Verlag.
//    Page 480.
//
double
lot::geom(const double p) {
  return poisson(exp_rand() * ((1 - p) / p));
}

/// Returns a hypergeometric variate
///
///
/// \f[ f(x) = \frac{{w \choose x}{b \choose n-x}}{{w+b \choose n}} \f]
///
/// \param nn1        The number of white balls in the urn (\f$w\f$)
/// \param nn2        The number of black balls in the urn (\f$b\f$)
/// \param kk         The sample size (\f$n\f$)
///
/// \f[ \mbox{E}(x) = n(\frac{w}{w+b}) \f]
/// \f[ \mbox{Var}(x) = \frac{n(\frac{w}{w+b})(1-\frac{w}{w+b})((w+b)-n)}{w+b-1} \f]
///
/// The code is modified from R v2.0 to take unsigned integer arguments
/// rather than doubles. isfinite() checks are no longer needed. Check for
/// n < r + b not done.
///
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000-2001 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rhyper(double NR, double NB, double n);
//
//  DESCRIPTION
//
//    Random variates from the hypergeometric distribution.
//    Returns the number of white balls drawn when kk balls
//    are drawn at random from an urn containing nn1 white
//    and nn2 black balls.
//
//  REFERENCE
//
//    V. Kachitvichyanukul and B. Schmeiser (1985).
//    ``Computer generation of hypergeometric random variates,''
//    Journal of Statistical Computation and Simulation 22, 127-145.
//
int
lot::hypergeom(const int nn1, const int nn2, const int kk) {
  const double con = 57.56462733;
  const double deltal = 0.0078;
  const double deltau = 0.0034;
  const double scale = 1e25;

  int i, ix;
  bool reject, setup1, setup2;

  double e, f, g, p, r, t, u, v, y;
  double de, dg, dr, ds, dt, gl, gu, nk, nm, ub;
  double xk, xm, xn, y1, ym, yn, yk, alv;

  /* These should become `thread_local globals' : */
  static int ks = -1;
  static int n1s = -1, n2s = -1;

  static int k, m;
  static int minjx, maxjx, n1, n2;

  static double a, d, s, w;
  static double tn, xl, xr, kl, kr, lamdl, lamdr, p1, p2, p3;

  /* if new parameter values, initialize */
  reject = true;
  if (nn1 != n1s || nn2 != n2s) {
    setup1 = true;	setup2 = true;
  } else if (kk != ks) {
    setup1 = false;	setup2 = true;
  } else {
    setup1 = false;	setup2 = false;
  }
  if (setup1) {
    n1s = nn1;
    n2s = nn2;
    tn = nn1 + nn2;
    if (nn1 <= nn2) {
	    n1 = nn1;
	    n2 = nn2;
    } else {
	    n1 = nn2;
	    n2 = nn1;
    }
  }
  if (setup2) {
    ks = kk;
    if (kk + kk >= tn) {
	    k = static_cast<int>(tn - kk);
    } else {
	    k = kk;
    }
  }
  if (setup1 || setup2) {
    m = static_cast<int>((k + 1.0) * (n1 + 1.0) / (tn + 2.0));
    minjx = imax2(0, k - n2);
    maxjx = imin2(n1, k);
  }
  /* generate random variate --- Three basic cases */

  if (minjx == maxjx) { /* I: degenerate distribution ---------------- */
    ix = maxjx;
    /* return ix;
       No, need to unmangle <TSL>*/
    /* return appropriate variate */

    if (kk + kk >= tn) {
      if (nn1 > nn2) {
        ix = kk - nn2 + ix;
      } else {
        ix = nn1 - ix;
      }
    } else {
      if (nn1 > nn2)
        ix = kk - ix;
    }
    return ix;

  } else if (m - minjx < 10) { /* II: inverse transformation ---------- */
    if (setup1 || setup2) {
	    if (k < n2) {
        w = exp(con + afc(n2) + afc(n1 + n2 - k)
                - afc(n2 - k) - afc(n1 + n2));
	    } else {
        w = exp(con + afc(n1) + afc(k)
                - afc(k - n2) - afc(n1 + n2));
	    }
    }
  L10:
    p = w;
    ix = minjx;
    u = unif_rand() * scale;
  L20:
    if (u > p) {
	    u -= p;
	    p *= (n1 - ix) * (k - ix);
	    ix++;
	    p = p / ix / (n2 - k + ix);
	    if (ix > maxjx)
        goto L10;
	    goto L20;
    }
  } else { /* III : h2pe --------------------------------------------- */

    if (setup1 || setup2) {
	    s = sqrt((tn - k) * k * n1 * n2 / (tn - 1) / tn / tn);

	    /* remark: d is defined in reference without int. */
	    /* the truncation centers the cell boundaries at 0.5 */

	    d = static_cast<int>((1.5 * s) + .5);
	    xl = m - d + .5;
	    xr = m + d + .5;
	    a = afc(m) + afc(n1 - m) + afc(k - m) + afc(n2 - k + m);
	    kl = exp(a - afc(static_cast<int>(xl)) - afc(static_cast<int>(n1 - xl))
               - afc(static_cast<int>(k - xl))
               - afc(static_cast<int>(n2 - k + xl)));
	    kr = exp(a - afc(static_cast<int>(xr - 1))
               - afc(static_cast<int>(n1 - xr + 1))
               - afc(static_cast<int>(k - xr + 1))
               - afc(static_cast<int>(n2 - k + xr - 1)));
	    lamdl = -log(xl * (n2 - k + xl) / (n1 - xl + 1) / (k - xl + 1));
	    lamdr = -log((n1 - xr + 1) * (k - xr + 1) / xr / (n2 - k + xr));
	    p1 = d + d;
	    p2 = p1 + kl / lamdl;
	    p3 = p2 + kr / lamdr;
    }
  L30:
    u = unif_rand() * p3;
    v = unif_rand();
    if (u < p1) {		/* rectangular region */
	    ix = static_cast<int>(xl + u);
    } else if (u <= p2) {	/* left tail */
	    ix = static_cast<int>(xl + log(v) / lamdl);
	    if (ix < minjx)
        goto L30;
	    v = v * (u - p1) * lamdl;
    } else {		/* right tail */
	    ix = static_cast<int>(xr - log(v) / lamdr);
	    if (ix > maxjx)
        goto L30;
	    v = v * (u - p2) * lamdr;
    }

    /* acceptance/rejection test */

    if (m < 100 || ix <= 50) {
	    /* explicit evaluation */
	    f = 1.0;
	    if (m < ix) {
        for (i = m + 1; i <= ix; i++)
          f = f * (n1 - i + 1) * (k - i + 1) / (n2 - k + i) / i;
	    } else if (m > ix) {
        for (i = ix + 1; i <= m; i++)
          f = f * i * (n2 - k + i) / (n1 - i) / (k - i);
	    }
	    if (v <= f) {
        reject = false;
	    }
    } else {
	    /* squeeze using upper and lower bounds */
	    y = ix;
	    y1 = y + 1.0;
	    ym = y - m;
	    yn = n1 - y + 1.0;
	    yk = k - y + 1.0;
	    nk = n2 - k + y1;
	    r = -ym / y1;
	    s = ym / yn;
	    t = ym / yk;
	    e = -ym / nk;
	    g = yn * yk / (y1 * nk) - 1.0;
	    dg = 1.0;
	    if (g < 0.0)
        dg = 1.0 + g;
	    gu = g * (1.0 + g * (-0.5 + g / 3.0));
	    gl = gu - .25 * (g * g * g * g) / dg;
	    xm = m + 0.5;
	    xn = n1 - m + 0.5;
	    xk = k - m + 0.5;
	    nm = n2 - k + xm;
	    ub = y * gu - m * gl + deltau
        + xm * r * (1. + r * (-0.5 + r / 3.0))
        + xn * s * (1. + s * (-0.5 + s / 3.0))
        + xk * t * (1. + t * (-0.5 + t / 3.0))
        + nm * e * (1. + e * (-0.5 + e / 3.0));
	    /* test against upper bound */
	    alv = log(v);
	    if (alv > ub) {
        reject = true;
	    } else {
				/* test against lower bound */
        dr = xm * (r * r * r * r);
        if (r < 0.0)
          dr /= (1.0 + r);
        ds = xn * (s * s * s * s);
        if (s < 0.0)
          ds /= (1.0 + s);
        dt = xk * (t * t * t * t);
        if (t < 0.0)
          dt /= (1.0 + t);
        de = nm * (e * e * e * e);
        if (e < 0.0)
          de /= (1.0 + e);
        if (alv < ub - 0.25 * (dr + ds + dt + de)
            + (y + m) * (gl - gu) - deltal) {
          reject = false;
        }
        else {
          /* * Stirling's formula to machine accuracy
           */
          if (alv <= (a - afc(ix) - afc(n1 - ix)
                      - afc(k - ix) - afc(n2 - k + ix))) {
            reject = false;
          } else {
            reject = true;
          }
        }
	    }
    }
    if (reject)
	    goto L30;
  }

  /* return appropriate variate */

  if (kk + kk >= tn) {
    if (nn1 > nn2) {
	    ix = kk - nn2 + ix;
    } else {
	    ix = nn1 - ix;
    }
  } else {
    if (nn1 > nn2)
	    ix = kk - ix;
  }
  return ix;
}

/// Returns a log-normal deviate
///
/// \f[ f(x) = \frac{1}{\sigma x\sqrt{2\pi}}e^{-\frac{(\log(x) - \mu)^2}{2\sigma^2}} \f]
///
/// \param logmean   logarithm of the mean of the corresponding normal (\f$\mu\f$)
/// \param logsd     logarithm of the sd of the corresponding normal (\f$\sigma\f$)
///
/// \f[ \mbox{E}(x) = e^{\mu + \sigma^2/2} \f]
/// \f[ \mbox{Var}(x) = e^{2\mu + \sigma^2}(e^{\sigma^2}-1) \f]
/// \f[ \mbox{mode} = \frac{e^\mu}{e^{\sigma^2}} \f]
/// \f[ \mbox{median} = e^\mu \f]
///
/// Derived from R v2.0. Does not do isnan() checks on arguments. Does
/// not check for sigma > 0.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000--2001  The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rlnorm(double logmean, double logsd);
//
//  DESCRIPTION
//
//    Random variates from the lognormal distribution.
//
double
lot::lnorm(const double logmean, const double logsd) {
    return exp(norm(logmean, logsd));
}

/// Returns a logistic variate
///
/// \f[ f(x) = \frac{1}{s}\frac{e^{\frac{x-m}{s}}}{(1 + e^{\frac{x-m}{s}})^2} \f]
///
/// or equivalently (dividing numerator and denominator by \f$e^{2\frac{x-m}{s}}\f$)
///
/// \f[ f(x) = \frac{1}{s}\frac{e^{\frac{-(x-m)}{s}}}{(1 + e^{\frac{-(x-m)}{s}})^2} \f]
///
/// \param location         the location parameter (\f$m\f$)
/// \param scale            the scale parameter (\f$s\f$)
///
/// \f[ \mbox{E}(x) = m \f]
/// \f[ \mbox{Var}(x) = \frac{\pi^2s^2}{3} \f]
///
/// Derived from R v2.0. Does not do isnan() checks on parameters. Does
/// not check for scale > 0.
//
//  R : A Computer Language for Statistical Data Analysis
//  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
double
lot::logis(double location, double scale) {
  double u = (this->*uniform_no_zero_generator)();
  return location + scale * log(u / (1. - u));
}

/// \f[ f({\bf n}) = {\sum_i n_i \choose n_1 \dots n_I}\prod_i p_i^{n_i} \f]
///
/// \param n                  Sample size (\f$\sum_i n_i\f$)
/// \param p                  Vector of probabilities (\f$p_i\f$)
///
/// Derived from R v2.0. Safety checks eliminated. User must ensure that
/// \f$\sum_i p_i = 1\f$ and \f$p_i >= 0\f$. \f$n > 0\f$ guaranteed, because
/// \f$n\f$ is unsigned.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 2003	      The R Foundation
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2, or (at your option)
//  any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available via WWW at
//  http://www.gnu.org/copyleft/gpl.html.  You can also obtain it by
//  writing to the Free Software Foundation, Inc., 59 Temple Place,
//  Suite 330, Boston, MA  02111-1307  USA.
//
//
//  SYNOPSIS
//
//	#include <Rmath.h>
//	void rmultinom(int n, double* prob, int K, int* rN);
//
//  DESCRIPTION
//
//	Random Vector from the multinomial distribution.
//             ~~~~~~
//  NOTE
//	Because we generate random _vectors_ this doesn't fit easily
//	into the do_random[1-4](.) framework setup in ../main/random.c
//	as that is used only for the univariate random generators.
//      Multivariate distributions typically have too complex parameter spaces
//	to be treated uniformly.
//	=> Hence also can have  int arguments.
std::vector<int>
lot::multinom(unsigned n, const std::vector<double>& p) {
    int K = p.size();
    vector<int> rN(K);
    double p_tot = 0.0;
    for(int k = 0; k < K; k++) {
      p_tot += p[k];
      rN[k] = 0;
    }
    if (n == 0) {
      return rN;
    }
    /* Generate the first K-1 obs. via binomials */
    for(int k = 0; k < K-1; k++) { /* (p_tot, n) are for "remaining binomial" */
      if(p[k] > 0.0) {
        double pp = p[k] / p_tot;
        rN[k] = ((pp < 1.) ? static_cast<int>(binom(n,  pp)):
                 /*>= 1; > 1 happens because of rounding */
                 n);
        n -= rN[k];
      } else {
        rN[k] = 0;
      }
      if(n <= 0) { /* we have all*/
        return rN;
      }
      p_tot -= p[k]; /* i.e. = sum(p[(k+1):K]) */
    }
    rN[K-1] = n;
    return rN;
}

/// Returns a negative binomial variate
///
/// \f[ f(x) = \frac{\Gamma(x+n)}{\Gamma(n)x!}p^n(1-p)^x \f]
///
/// \param n         the ``size'' parameter
/// \param p         the ``probability'' parameter
///
/// \f[ \mbox{E}(x) = \frac{x(1-p)}{p} \f]
/// \f[ \mbox{Var}(x) = \frac{x(1-p)}{p^2} \f]
///
/// Derived from R v2.0. Does not check isfinite() on arguments or
/// ensure that p is in [0, 1).
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000--2001  The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  SYNOPSIS
//
//    #include <Rmath.h>
//    double rnbinom(double n, double p)
//
//  DESCRIPTION
//
//    Random variates from the negative binomial distribution.
//
//  NOTES
//
//    x = the number of failures before the n-th success
//
//  REFERENCE
//
//    Devroye, L. (1986).
//    Non-Uniform Random Variate Generation.
//    New York:Springer-Verlag. Page 480.
//
//  METHOD
//
//    Generate lambda as gamma with shape parameter n and scale
//    parameter p/(1-p).  Return a Poisson deviate with mean lambda.
//
double
lot::nbinom(const double n /* size */, const double p /* prob */) {
  return poisson(gamma(n, (1 - p) / p));
}

/// Returns Poisson deviate
///
/// \f[ f(x) = \frac{\lambda^x e^{-\lambda}}{x!} \f]
///
/// \param mu  mean of the Poisson distribution (\f$\lambda\f$)
///
/// \f[ \mbox{E}(x) = \mu \f]
/// \f[ \mbox{Var}(x) = \mu \f]
///
/// derived from R v1.9.0
///
/// NOTE: Check for finite mu not included
///
/// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
/// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
/// if you are using the Mersenne-Twister. ZERO is the default.
///
// Mathlib : A C Library of Special Functions
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
// SYNOPSIS
//
//   #include <Rmath.h>
//   double rpois(double lambda)
//
// DESCRIPTION
//
//   Random variates from the Poisson distribution.
//
// REFERENCE
//
//   Ahrens, J.H. and Dieter, U. (1982).
//   Computer generation of Poisson deviates
//   from modified normal distributions.
//   ACM Trans. Math. Software 8, 163-179.
//
int
lot::poisson(const double mu) {
  /* Factorial Table (0:9)! */
  const double fact[10] =
    {
      1., 1., 2., 6., 24., 120., 720., 5040., 40320., 362880.
    };
  /* These are static --- persistent between calls for same mu : */
  static int l, m;
  static double b1, b2, c, c0, c1, c2, c3;
  static double pp[36], p0, p, q, s, d, omega;
  static double big_l;/* integer "w/o overflow" */
  static double muprev = 0., muprev2 = 0.;/*, muold	 = 0.*/

  /* Local Vars  [initialize some for -Wall]: */
  double del, difmuk= 0., E= 0., fk= 0., fx, fy, g, px, py, t, u= 0., v, x;
  double pois = -1.;
  int k, kflag, big_mu, new_big_mu = false;

  if (mu <= 0.) {
    return 0;
  }
  big_mu = mu >= 10.;
  if(big_mu) {
    new_big_mu = false;
  }
  if (!(big_mu && mu == muprev)) {/* maybe compute new persistent par.s */
    if (big_mu) {
	    new_big_mu = true;
	    /* Case A. (recalculation of s,d,l	because mu has changed):
	     * The poisson probabilities pk exceed the discrete normal
	     * probabilities fk whenever k >= m(mu).
	     */
	    muprev = mu;
	    s = sqrt(mu);
	    d = 6. * mu * mu;
	    big_l = floor(mu - 1.1484);
	    /* = an upper bound to m(mu) for all mu >= 10.*/
    }
    else { /* Small mu ( < 10) -- not using normal approx. */
	    /* Case B. (start new table and calculate p0 if necessary) */
	    /*muprev = 0.;-* such that next time, mu != muprev ..*/
	    if (mu != muprev) {
        muprev = mu;
        m = imax2(1, static_cast<int>(mu));
        l = 0; /* pp[] is already ok up to pp[l] */
        q = p0 = p = exp(-mu);
	    }
	    repeat {
        /* Step U. uniform sample for inversion method */
        u = unif_rand();
        if (u <= p0) {
          return 0;
        }
        /* Step T. table comparison until the end pp[l] of the
           pp-table of cumulative poisson probabilities
           (0.458 > ~= pp[9](= 0.45792971447) for mu=10 ) */
        if (l != 0) {
          for (k = (u <= 0.458) ? 1 : imin2(l, m);  k <= l; k++)
            if (u <= pp[k])
              return k;
          if (l == 35) /* u > pp[35] */
            continue;
        }
        /* Step C. creation of new poisson
           probabilities p[l..] and their cumulatives q =: pp[k] */
        l++;
        for (k = l; k <= 35; k++) {
          p *= mu / k;
          q += p;
          pp[k] = q;
          if (u <= q) {
            l = k;
            return k;
          }
        }
        l = 35;
	    } /* end(repeat) */
    }/* mu < 10 */
  } /* end {initialize persistent vars} */
  /* Only if mu >= 10 : ----------------------- */
  /* Step N. normal sample */
  g = mu + s * norm_rand();/* norm_rand() ~ N(0,1), standard normal */
  if (g >= 0.) {
    pois = floor(g);
    /* Step I. immediate acceptance if pois is large enough */
    if (pois >= big_l) {
	    return static_cast<int>(pois);
    }
    /* Step S. squeeze acceptance */
    fk = pois;
    difmuk = mu - fk;
    u = unif_rand(); /* ~ U(0,1) - sample */
    if (d * u >= difmuk * difmuk * difmuk) {
	    return static_cast<int>(pois);
    }
  }
  /* Step P. preparations for steps Q and H.
     (recalculations of parameters if necessary) */
  if (new_big_mu || mu != muprev2) {
    /* Careful! muprev2 is not always == muprev
       because one might have exited in step I or S
    */
    muprev2 = mu;
    omega = M_1_SQRT_2PI / s;
    /* The quantities b1, b2, c3, c2, c1, c0 are for the Hermite
     * approximations to the discrete normal probabilities fk. */
    b1 = one_24 / mu;
    b2 = 0.3 * b1 * b1;
    c3 = one_7 * b1 * b2;
    c2 = b2 - 15. * c3;
    c1 = b1 - 6. * b2 + 45. * c3;
    c0 = 1. - b1 + 3. * b2 - 15. * c3;
    c = 0.1069 / mu; /* guarantees majorization by the 'hat'-function. */
  }
  if (g >= 0.) {
    /* 'Subroutine' F is called (kflag=0 for correct return) */
    kflag = 0;
    goto Step_F;
  }
  repeat {
    /* Step E. Exponential Sample */
    E = exp_rand();	/* ~ Exp(1) (standard exponential) */
    /*  sample t from the laplace 'hat'
        (if t <= -0.6744 then pk < fk for all mu >= 10.) */
    u = 2 * unif_rand() - 1.;
    t = 1.8 + fsign(E, u);
    if (t > -0.6744) {
	    pois = floor(mu + s * t);
	    fk = pois;
	    difmuk = mu - fk;
	    /* 'subroutine' F is called (kflag=1 for correct return) */
	    kflag = 1;
	  Step_F: /* 'subroutine' F : calculation of px,py,fx,fy. */
	    if (pois < 10) { /* use factorials from table fact[] */
        px = -mu;
        py = pow(mu, pois) / fact[static_cast<int>(pois)];
	    }
	    else {
        /* Case pois >= 10 uses polynomial approximation
           a0-a7 for accuracy when advisable */
        del = one_12 / fk;
        del = del * (1. - 4.8 * del * del);
        v = difmuk / fk;
        if (fabs(v) <= 0.25)
          px = fk * v * v * (((((((a7 * v + a6) * v + a5) * v + a4) *
                                v + a3) * v + a2) * v + a1) * v + a0)
            - del;
        else /* |v| > 1/4 */
          px = fk * log(1. + v) - difmuk - del;
        py = M_1_SQRT_2PI / sqrt(fk);
	    }
	    x = (0.5 - difmuk) / s;
	    x *= x;/* x^2 */
	    fx = -0.5 * x;
	    fy = omega * (((c3 * x + c2) * x + c1) * x + c0);
	    if (kflag > 0) {
        /* Step H. Hat acceptance (E is repeated on rejection) */
        if (c * fabs(u) <= py * exp(px + E) - fy * exp(fx + E))
          break;
	    } else
        /* Step Q. Quotient acceptance (rare case) */
        if (fy - u * fy <= py * exp(px - fx))
          break;
    }/* t > -.67.. */
  }
  return static_cast<int>(pois);
}

/// Returns standard normal deviate
///
/// \f[ f(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}} \f]
///
/// Kinderman-Ramage standard normal generator from R v1.9.0
///
/// \f[ \mbox{E}(x) = 0 \f]
/// \f[ \mbox{Var}(x) = 1 \f]
///
/// NOTE: R uses RNGs uniform on [0,1). To ensure consistency with that
/// well tested code make sure that you have Set_MT(ZERO), or the equivalent,
/// if you are using the Mersenne-Twister. ZERO is the default.
///
double
lot::snorm(void) {
	double u1 = unif_rand();
	if(u1 < 0.884070402298758) {
    double u2 = unif_rand();
    return A*(1.131131635444180*u1+u2-1);
	}
	if(u1 >= 0.973310954173898) { /* tail: */
    repeat {
      double u2 = unif_rand();
      double u3 = unif_rand();
      double tt = (A*A-2*log(u3));
      if( u2*u2<(A*A)/tt )
		    return (u1 < 0.986655477086949) ? sqrt(tt) : -sqrt(tt);
    }
	}
	if(u1 >= 0.958720824790463) { /* region3: */
    repeat {
      double u2 = unif_rand();
      double u3 = unif_rand();
      double tt = A - 0.630834801921960* fmin2(u2,u3);
      if(fmax2(u2,u3) <= 0.755591531667601)
		    return (u2<u3) ? tt : -tt;
      if(0.034240503750111*fabs(u2-u3) <= g(tt))
		    return (u2<u3) ? tt : -tt;
    }
	}
	if(u1 >= 0.911312780288703) { /* region2: */
    repeat {
      double u2 = unif_rand();
      double u3 = unif_rand();
      double tt = 0.479727404222441+1.105473661022070*fmin2(u2,u3);
      if( fmax2(u2,u3)<=0.872834976671790 )
		    return (u2<u3) ? tt : -tt;
      if( 0.049264496373128*fabs(u2-u3)<=g(tt) )
		    return (u2<u3) ? tt : -tt;
    }
	}
	/* ELSE	 region1: */
	repeat {
    double u2 = unif_rand();
    double u3 = unif_rand();
    double tt = 0.479727404222441-0.595507138015940*fmin2(u2,u3);
    if (tt < 0.) continue;
    if(fmax2(u2,u3) <= 0.805577924423817)
      return (u2<u3) ? tt : -tt;
    if(0.053377549506886*fabs(u2-u3) <= g(tt))
      return (u2<u3) ? tt : -tt;
	}
}

/// Returns a t variate
///
/// \f[ f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi\nu}\Gamma(\frac{\nu}{2})(1 + \frac{x^2}{\nu})^{(\nu + 1)/2}} \f]
///
/// \param df        the degrees of freedom (\f$\nu\f$)
///
/// \f[ \mbox{E}(x) = 0 \quad , \quad \nu > 1 \f]
/// \f[ \mbox{Var}(x) = \frac{\nu}{\nu - 2} \quad , \quad \nu > 2 \f]
///
/// Derived from R v2.0. Does not check isnan() or isfinite() on argument.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000-2001 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  DESCRIPTION
//
//    Pseudo-random variates from a t distribution.
//
//  NOTES
//
//    This function calls rchisq and rnorm to do the real work.
//
double
lot::t(double df) {
  double num;
  /* Some compilers (including MW6) evaluated this from right to left
     return norm_rand() / sqrt(rchisq(df) / df); */
	num = norm_rand();
	return num / sqrt(chisq(df) / df);
}

/// Returns a Weibull variate
///
/// \f[ f(x) = (\frac{a}{b})(\frac{x}{b})^{a-1}e^{-\frac{x}{b}^a} \f]
///
/// \param shape        the ``shape'' parameter (\f$a\f$)
/// \param scale        the ``scale'' parameter (\f$b\f$)
///
/// \f[ \mbox{E}(x) = b\Gamma(1 + \frac{1}{a}) \f]
/// \f[ \mbox{Var}(x) = b^2(\Gamma(1+\frac{2}{a}) - \Gamma(1+\frac{1}{a})^2) \f]
///
/// Derived from R v2.0. Does not check isnan() on arguments. Does not
/// check for \f$a > 0\f$ and \f$b > 0\f$.
//
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
//
//  DESCRIPTION
//
//    Random variates from the Weibull distribution.
//
double
lot::weibull(double shape, double scale) {
  return scale * pow(-log(unif_rand()), 1.0 / shape);
}
