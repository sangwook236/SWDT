///
/// \file   Density.cpp
/// \brief  Collects a variety of constants and function in namespace Util.
///
/// Provides definitions for a variety of functions related to numerical
/// evaluation of probability densities.
///
/// All are declared in namespace Density with an eye towards avoiding
/// naming conflicts.
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

// standard includes
#include <cmath>
#include <limits>
// local includes
#include "mcmc++/Density.h"
#include "mcmc++/util.h"

using Util::dbl_max;

// chosen for agreement with R v2.2.0
const double Density::MinGammaPar = pow(2., -128);
const double Density::MaxGammaPar = pow(2., 128);

namespace {

  inline double rd0(const bool give_log) {
    return give_log ? Util::log_dbl_min : 0.0;
  }

  inline double rd1(const bool give_log) {
    return give_log ? 0.0 : 1.0;
  }

  inline double rdexp(const double x, const bool give_log) {
    return give_log ? x : exp(x);
  }

  inline double rdfexp(const double f, const double x, const bool give_log) {
    return give_log ? -0.5*log(f) + (x) : exp(x) / sqrt(f);
  }

  // From Rmath.h 
#if defined(M_PI)
#undef M_PI
#endif
  const double M_PI = 3.141592653589793238462643383280;
  const double M_2PI = 6.283185307179586476925286766559;
  const double M_LN_SQRT_2PI = 0.918938533204672741780329736406;
  const double M_1_SQRT_2PI = 0.398942280401432677939946059934;
  // for stirlerr()
  const double S0 = 0.083333333333333333333;       /* 1/12 */
  const double S1 = 0.00277777777777777777778;     /* 1/360 */
  const double S2 = 0.00079365079365079365079365;  /* 1/1260 */
  const double S3 = 0.000595238095238095238095238; /* 1/1680 */
  const double S4 = 0.0008417508417508417508417508; /* 1/1188 */

  /*
   *  AUTHOR
   *    Catherine Loader, catherine\research.bell-labs.com.
   *    October 23, 2000.
   *
   *  Merge in to R:
   * Copyright (C) 2000, The R Core Development Team
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
   *
   *  DESCRIPTION
   *
   *    Computes the log of the error term in Stirling's formula.
   *      For n > 15, uses the series 1/12n - 1/360n^3 + ...
   *      For n <=15, integers or half-integers, uses stored values.
   *      For other n < 15, uses lgamma directly (don't use this to
   *        write lgamma!)
   *
   * Merge in to R:
   * Copyright (C) 2000, The R Core Development Team
   * R has lgammafn, and lgamma is not part of ISO C
   *
   */
  /* stirlerr(n) = log(n!) - log( sqrt(2*pi*n)*(n/e)^n ) */
  double stirlerr(double n) {
    /*
      error for 0, 0.5, 1.0, 1.5, ..., 14.5, 15.0.
    */
    const double sferr_halves[31] = {
      0.0,       /* n=0 - wrong, place holder only */
      0.1534264097200273452913848,        /* 0.5 */
      0.0810614667953272582196702,        /* 1.0 */
      0.0548141210519176538961390,        /* 1.5 */
      0.0413406959554092940938221,        /* 2.0 */
      0.03316287351993628748511048,       /* 2.5 */
      0.02767792568499833914878929,       /* 3.0 */
      0.02374616365629749597132920,       /* 3.5 */
      0.02079067210376509311152277,       /* 4.0 */
      0.01848845053267318523077934,       /* 4.5 */
      0.01664469118982119216319487,       /* 5.0 */
      0.01513497322191737887351255,       /* 5.5 */
      0.01387612882307074799874573,       /* 6.0 */
      0.01281046524292022692424986,       /* 6.5 */
      0.01189670994589177009505572,       /* 7.0 */
      0.01110455975820691732662991,       /* 7.5 */
      0.010411265261972096497478567,       /* 8.0 */
      0.009799416126158803298389475,       /* 8.5 */
      0.009255462182712732917728637,       /* 9.0 */
      0.008768700134139385462952823,       /* 9.5 */
      0.008330563433362871256469318,       /* 10.0 */
      0.007934114564314020547248100,       /* 10.5 */
      0.007573675487951840794972024,       /* 11.0 */
      0.007244554301320383179543912,       /* 11.5 */
      0.006942840107209529865664152,       /* 12.0 */
      0.006665247032707682442354394,       /* 12.5 */
      0.006408994188004207068439631,       /* 13.0 */
      0.006171712263039457647532867,       /* 13.5 */
      0.005951370112758847735624416,       /* 14.0 */
      0.005746216513010115682023589,       /* 14.5 */
      0.005554733551962801371038690  /* 15.0 */
    };

    double nn;
    if (n <= 15.0) {
      nn = n + n;
      if (nn == (int)nn)
        return (sferr_halves[(int)nn]);
      return (Density::gamln(n + 1.0) - (n + 0.5)*log(n) + n - M_LN_SQRT_2PI);
    }
    nn = n * n;
    if (n > 500)
      return ((S0 -S1 / nn) / n);
    if (n > 80)
      return ((S0 -(S1 - S2 / nn) / nn) / n);
    if (n > 35)
      return ((S0 -(S1 - (S2 - S3 / nn) / nn) / nn) / n);
    /* 15 < n <= 35 : */
    return ((S0 -(S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n);
  }

  /*
   *  AUTHOR
   * Catherine Loader, catherine\research.bell-labs.com.
   * October 23, 2000.
   *
   *  Merge in to R:
   * Copyright (C) 2000, The R Core Development Team
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
   *
   *  DESCRIPTION
   * Evaluates the "deviance part"
   * bd0(x,M) :=  M * D0(x/M) = M*[ x/M * log(x/M) + 1 - (x/M) ] =
   *    =  x * log(x/M) + M - x
   * where M = E[X] = n*p (or = lambda), for   x, M > 0
   *
   * in a manner that should be stable (with small relative error)
   * for all x and np. In particular for x/np close to 1, direct
   * evaluation fails, and evaluation is based on the Taylor series
   * of log((1+v)/(1-v)) with v = (x-np)/(x+np).
   */
  double bd0(double x, double np) {
    double ej, s, s1, v;
    int j;
    if (fabs(x - np) < 0.1*(x + np)) {
      v = (x - np) / (x + np);
      s = (x - np) * v; /* s using v -- change by MM */
      ej = 2 * x * v;
      v = v * v;
      for (j = 1; ; j++) { /* Taylor series */
        ej *= v;
        s1 = s + ej / ((j << 1) + 1);

        if (s1 == s) /* last term was effectively 0 */
          return (s1);

        s = s1;
      }
    }
    /* else:  | x - np |  is not too small */
    return (x*log(x / np) + np - x);
  }

  /**
   * AUTHOR
   *   Catherine Loader, catherine\research.bell-labs.com.
   *   October 23, 2000.
   *
   *  Merge in to R:
   * Copyright (C) 2000, The R Core Development Team
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
   *
   * DESCRIPTION
   *
   *   To compute the binomial probability, call dbinom(x,n,p).
   *   This checks for argument validity, and calls dbinom_raw().
   *
   *   dbinom_raw() does the actual computation; note this is called by
   *   other functions in addition to dbinom()).
   *     (1) dbinom_raw() has both p and q arguments, when one may be represented
   *         more accurately than the other (in particular, in df()).
   *     (2) dbinom_raw() does NOT check that inputs x and n are integers. This
   *         should be done in the calling function, where necessary.
   *     (3) Also does not check for 0 <= p <= 1 and 0 <= q <= 1 or NaN's.
   *         Do this in the calling function.
   */
  double dbinom_raw(const double x, const double n, const double p,
                    const double q, const bool give_log) {
    double f, lc;
    if (p == 0)
      return ((x == 0) ? rd1(give_log) : rd0(give_log));
    if (q == 0)
      return ((x == n) ? rd1(give_log) : rd0(give_log));
    if (x == 0) {
      if (n == 0)
        return rd1(give_log);
      lc = (p < 0.1) ? -bd0(n, n * q) - n * p : n * log(q);
      return rdexp(lc, give_log);
    }
    if (x == n) {
      lc = (q < 0.1) ? -bd0(n, n * p) - n * q : n * log(p);
      return rdexp(lc, give_log);
    }
    if (x < 0 || x > n)
      return rd0(give_log);
    lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    f = (M_2PI * x * (n - x)) / n;
    return rdfexp(f, lc, give_log);
  }

  /*
   *  AUTHOR
   *    Catherine Loader, catherine\research.bell-labs.com.
   *    October 23, 2000.
   *
   *  Merge in to R:
   *	Copyright (C) 2000, The R Core Development Team
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
   *
   * DESCRIPTION
   *
   *    dpois_raw() computes the Poisson probability  lb^x exp(-lb) / x!.
   *      This does not check that x is an integer, since dgamma() may
   *      call this with a fractional x argument. Any necessary argument
   *      checks should be done in the calling function.
   *
   */
  double dpois_raw(double x, double lambda, int give_log) {
    if (lambda == 0) {
      return( (x == 0) ? rd1(give_log) : rd0(give_log) );
    }
    if (x == 0) {
      return( rdexp(-lambda, give_log) );
    }
    if (x < 0)  {
      return( rd0(give_log) );
    }
    return(rdfexp( M_2PI*x, -stirlerr(x)-bd0(x,lambda), give_log ));
  }

  const int dblMinExp = std::numeric_limits<double>::min_exponent;
  const int dblMaxExp = std::numeric_limits<double>::max_exponent;
  const int fltRadix = std::numeric_limits<double>::radix;
  const int dblMantDig = std::numeric_limits<double>::digits;

  /*
   *  Mathlib : A C Library of Special Functions
   *  Copyright (C) 1998 Ross Ihaka
   *  Copyright (C) 2000-2001 the R Development Core Team
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
   *    #include "Rnorm.h"
   *    void dpsifn(double x, int n, int kode, int m,
   *    double *ans, int *nz, int *ierr)
   *    double digamma(double x);
   *    double trigamma(double x)
   *    double tetragamma(double x)
   *    double pentagamma(double x)
   *
   *  DESCRIPTION
   *
   *    Compute the derivatives of the psi function
   *    and polygamma functions.
   *
   *    The following definitions are used in dpsifn:
   *
   *    Definition 1
   *
   *  psi(x) = d/dx (ln(gamma(x)),  the first derivative of
   *           the log gamma function.
   *
   *    Definition 2
   *       k  k
   *  psi(k,x) = d /dx (psi(x)),    the k-th derivative
   *           of psi(x).
   *
   *
   *    "dpsifn" computes a sequence of scaled derivatives of
   *    the psi function; i.e. for fixed x and m it computes
   *    the m-member sequence
   *
   *    (-1)^(k+1) / gamma(k+1) * psi(k,x)
   *       for k = n,...,n+m-1
   *
   *    where psi(k,x) is as defined above.   For kode=1, dpsifn
   *    returns the scaled derivatives as described.  kode=2 is
   *    operative only when k=0 and in that case dpsifn returns
   *    -psi(x) + ln(x). That is, the logarithmic behavior for
   *    large x is removed when kode=2 and k=0.  When sums or
   *    differences of psi functions are computed the logarithmic
   *    terms can be combined analytically and computed separately
   *    to help retain significant digits.
   *
   *    Note that dpsifn(x, 0, 1, 1, ans) results in ans = -psi(x).
   *
   *  INPUT
   *
   * x     - argument, x > 0.
   *
   * n     - first member of the sequence, 0 <= n <= 100
   *  n == 0 gives ans(1) = -psi(x)     for kode=1
   *          -psi(x)+ln(x) for kode=2
   *
   * kode  - selection parameter
   *  kode == 1 returns scaled derivatives of the
   *  psi function.
   *  kode == 2 returns scaled derivatives of the
   *  psi function except when n=0. In this case,
   *  ans(1) = -psi(x) + ln(x) is returned.
   *
   * m     - number of members of the sequence, m >= 1
   *
   *  OUTPUT
   *
   * ans   - a vector of length at least m whose first m
   *  components contain the sequence of derivatives
   *  scaled according to kode.
   *
   * nz    - underflow flag
   *  nz == 0, a normal return
   *  nz != 0, underflow, last nz components of ans are
   *    set to zero, ans(m-k+1)=0.0, k=1,...,nz
   *
   * ierr  - error flag
   *  ierr=0, a normal return, computation completed
   *  ierr=1, input error,  no computation
   *  ierr=2, overflow,  x too small or n+m-1 too
   *   large or both
   *  ierr=3, error,   n too large. dimensioned
   *   array trmr(nmax) is not large enough for n
   *
   *    The nominal computational accuracy is the maximum of unit
   *    roundoff (d1mach(4)) and 1e-18 since critical constants
   *    are given to only 18 digits.
   *
   *    The basic method of evaluation is the asymptotic expansion
   *    for large x >= xmin followed by backward recursion on a two
   *    term recursion relation
   *
   *      w(x+1) + x^(-n-1) = w(x).
   *
   *    this is supplemented by a series
   *
   *      sum( (x+k)^(-n-1) , k=0,1,2,... )
   *
   *    which converges rapidly for large n. both xmin and the
   *    number of terms of the series are calculated from the unit
   *    roundoff of the machine environment.
   *
   *  AUTHOR
   *
   *    Amos, D. E.  (Fortran)
   *    Ross Ihaka   (C Translation)
   *
   *  REFERENCES
   *
   *    Handbook of Mathematical Functions,
   *    National Bureau of Standards Applied Mathematics Series 55,
   *    Edited by M. Abramowitz and I. A. Stegun, equations 6.3.5,
   *    6.3.18, 6.4.6, 6.4.9 and 6.4.10, pp.258-260, 1964.
   *
   *    D. E. Amos, (1983). "A Portable Fortran Subroutine for
   *    Derivatives of the Psi Function", Algorithm 610,
   *    TOMS 9(4), pp. 494-502.
   *
   *    Routines called: d1mach, i1mach.
   */
  void dpsifn(double x, int n, int kode, int m, double *ans, int *nz,
              int *ierr) {
    const double bvalues[] = { /* Bernoulli Numbers */
      1.00000000000000000e+00,
      -5.00000000000000000e-01,
      1.66666666666666667e-01,
      -3.33333333333333333e-02,
      2.38095238095238095e-02,
      -3.33333333333333333e-02,
      7.57575757575757576e-02,
      -2.53113553113553114e-01,
      1.16666666666666667e+00,
      -7.09215686274509804e+00,
      5.49711779448621554e+01,
      -5.29124242424242424e+02,
      6.19212318840579710e+03,
      -8.65802531135531136e+04,
      1.42551716666666667e+06,
      -2.72982310678160920e+07,
      6.01580873900642368e+08,
      -1.51163157670921569e+10,
      4.29614643061166667e+11,
      -1.37116552050883328e+13,
      4.88332318973593167e+14,
      -1.92965793419400681e+16
    };
    const double *b = (double *) & bvalues - 1; /* ==> b[1] = bvalues[0], etc */
    const int nmax = 100;

    int i, j, k, mm, mx, nn, np, nx, fn;
    double arg, den, elim, eps, fln, fx, rln, rxsq,
      r1m4, r1m5, s, slope, t, ta, tk, tol, tols, tss, tst,
      tt, t1, t2, wdtol, xdmln, xdmy, xinc, xln, xm, xmin,
      xq, yint;
    double trm[23], trmr[101];

    *ierr = 0;
    if (x <= 0.0 || n < 0 || kode < 1 || kode > 2 || m < 1) {
      *ierr = 1;
      return ;
    }

    /* fortran adjustment */
    ans--;

    *nz = 0;
    mm = m;
    nx = std::min( -dblMinExp, dblMaxExp);
    r1m5 = log10(2.0);
    r1m4 = pow((double)fltRadix, 1 - dblMantDig) * 0.5;
    wdtol = std::max(r1m4, 0.5e-18);

    /* elim = approximate exponential over and underflow limit */

    elim = 2.302 * (nx * r1m5 - 3.0);
    xln = log(x);
    for (;;) {
      nn = n + mm - 1;
      fn = nn;
      t = (fn + 1) * xln;

      /* overflow and underflow test for small and large x */

      if (fabs(t) > elim) {
        if (t <= 0.0) {
          *nz = 0;
          *ierr = 2;
          return ;
        }
      } else {
        if (x < wdtol) {
          ans[1] = pow(x, -n - 1.0);
          if (mm != 1) {
            for (i = 2, k = 1; i <= mm ; i++, k++)
              ans[k + 1] = ans[k] / x;
          }
          if (n == 0 && kode == 2)
            ans[1] += xln;
          return ;
        }

        /* compute xmin and the number of terms of the series,  fln+1 */

        rln = r1m5 * dblMantDig;
        rln = std::min(rln, 18.06);
        fln = std::max(rln, 3.0) - 3.0;
        yint = 3.50 + 0.40 * fln;
        slope = 0.21 + fln * (0.0006038 * fln + 0.008677);
        xm = yint + slope * fn;
        mx = (int)xm + 1;
        xmin = mx;
        if (n != 0) {
          xm = -2.302 * rln - std::min(0.0, xln);
          arg = xm / n;
          arg = std::min(0.0, arg);
          eps = exp(arg);
          xm = 1.0 - eps;
          if (fabs(arg) < 1.0e-3)
            xm = -arg;
          fln = x * xm / eps;
          xm = xmin - x;
          if (xm > 7.0 && fln < 15.0)
            break;
        }
        xdmy = x;
        xdmln = xln;
        xinc = 0.0;
        if (x < xmin) {
          nx = (int)x;
          xinc = xmin - nx;
          xdmy = x + xinc;
          xdmln = log(xdmy);
        }

        /* generate w(n+mm-1, x) by the asymptotic expansion */

        t = fn * xdmln;
        t1 = xdmln + xdmln;
        t2 = t + xdmln;
        tk = std::max(fabs(t), std::max(fabs(t1), fabs(t2)));
        if (tk <= elim)
          goto L10;
      }
      nz++;
      ans[mm] = 0.0;
      mm--;
      if (mm == 0)
        return ;
    }
    nn = (int)fln + 1;
    np = n + 1;
    t1 = (n + 1) * xln;
    t = exp( -t1);
    s = t;
    den = x;
    for (i = 1 ; i <= nn ; i++) {
      den += 1.;
      trm[i] = pow(den, (double) - np);
      s += trm[i];
    }
    ans[1] = s;
    if (n == 0 && kode == 2)
      ans[1] = s + xln;

    if (mm != 1) { /* generate higher derivatives, j > n */

      tol = wdtol / 5.0;
      for (j = 2; j <= mm; j++) {
        t = t / x;
        s = t;
        tols = t * tol;
        den = x;
        for (i = 1 ; i <= nn ; i++) {
          den += 1.;
          trm[i] /= den;
          s += trm[i];
          if (trm[i] < tols)
            break;
        }
        ans[j] = s;
      }
    }
    return ;

  L10:
    tss = exp( -t);
    tt = 0.5 / xdmy;
    t1 = tt;
    tst = wdtol * tt;
    if (nn != 0)
      t1 = tt + 1.0 / fn;
    rxsq = 1.0 / (xdmy * xdmy);
    ta = 0.5 * rxsq;
    t = (fn + 1) * ta;
    s = t * b[3];
    if (fabs(s) >= tst) {
      tk = 2.0;
      for (k = 4; k <= 22; k++) {
        t = t * ((tk + fn + 1) / (tk + 1.0)) * ((tk + fn) / (tk + 2.0)) * rxsq;
        trm[k] = t * b[k];
        if (fabs(trm[k]) < tst)
          break;
        s += trm[k];
        tk += 2.;
      }
    }
    s = (s + t1) * tss;
    if (xinc != 0.0) {

      /* backward recur from xdmy to x */

      nx = (int)xinc;
      np = nn + 1;
      if (nx > nmax) {
        *nz = 0;
        *ierr = 3;
        return ;
      } else {
        if (nn == 0)
          goto L20;
        xm = xinc - 1.0;
        fx = x + xm;

        /* this loop should not be changed. fx is accurate when x is small */
        for (i = 1; i <= nx; i++) {
          trmr[i] = pow(fx, (double) - np);
          s += trmr[i];
          xm -= 1.;
          fx = x + xm;
        }
      }
    }
    ans[mm] = s;
    if (fn == 0)
      goto L30;

    /* generate lower derivatives,  j < n+mm-1 */

    for (j = 2; j <= mm; j++) {
      fn--;
      tss *= xdmy;
      t1 = tt;
      if (fn != 0)
        t1 = tt + 1.0 / fn;
      t = (fn + 1) * ta;
      s = t * b[3];
      if (fabs(s) >= tst) {
        tk = 4 + fn;
        for (k = 4 ; k <= 22 ; k++) {
          trm[k] = trm[k] * (fn + 1) / tk;
          if (fabs(trm[k]) < tst)
            break;
          s += trm[k];
          tk += 2.;
        }
      }
      s = (s + t1) * tss;
      if (xinc != 0.0) {
        if (fn == 0)
          goto L20;
        xm = xinc - 1.0;
        fx = x + xm;
        for (i = 1 ; i <= nx ; i++) {
          trmr[i] = trmr[i] * fx;
          s += trmr[i];
          xm -= 1.;
          fx = x + xm;
        }
      }
      mx = mm - j + 1;
      ans[mx] = s;
      if (fn == 0)
        goto L30;
    }
    return ;

  L20:
    for (i = 1; i <= nx; i++)
      s += 1. / (x + nx - i);

  L30:
    if (kode != 2)
      ans[1] = s - xdmln;
    else if (xdmy != x) {
      xq = xdmy / x;
      ans[1] = s - log(xq);
    }
    return ;
  }

  /// Digamma function from R v1.7.0 (error checking removed)
  ///
  double digamma(const double x) {
    double ans;
    int nz, ierr;
    dpsifn(x, 0, 1, 1, &ans, &nz, &ierr);
    return -ans;
  }

  /// Trigamma function from R v1.7.0 (error checking removed)
  ///
  double trigamma(double x) {
    double ans;
    int nz, ierr;
    dpsifn(x, 1, 1, 1, &ans, &nz, &ierr);
    return ans;
  }

  /// Tetragamma function from R v1.7.0 (error checking removed)
  ///
  double tetragamma(double x) {
    double ans;
    int nz, ierr;
    dpsifn(x, 2, 1, 1, &ans, &nz, &ierr);
    return -2.0*ans;
  }

  /// Pentagamma function from R v1.7.0 (error checking removed)
  ///
  double pentagamma(double x) {
    double ans;
    int nz, ierr;
    dpsifn(x, 3, 1, 1, &ans, &nz, &ierr);
    return 6.0*ans;
  }

}

/// Density of the beta distribution
///
/// Returns the probability density associated with a beta variate:
///
/// \f[f(x) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1} \f]
///
/// \param x 
/// \param a
/// \param b          
/// \param give_log   Return log density?
///
/// \f[ \mbox{E}(x) = \frac{a}{a+b} \f]
/// \f[ \mbox{Var}(x) = \frac{ab}{(a+b)^2(a+b+1)} \f]
///
/// This implementation is derived from R. It assumes that the caller has
/// ensured that a and b are positive.
//
//
// MODIFIED BY
//   Kent Holsinger
//   23 July 2002
//
// AUTHOR
//   Catherine Loader, catherine@research.bell-labs.com.
//   October 23, 2000.
//
// Merge in to R:
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
//
// DESCRIPTION
//
//   Beta density,
//                  (a+b-1)!     a-1       b-1
//     p(x;a,b) = ------------ x     (1-x)
//                (a-1)!(b-1)!
// 
//              = (a+b-1) dbinom(a-1; a+b-2,x)
//
//   We must modify this when a<1 or b<1, to avoid passing negative
//   arguments to dbinom_raw. Note that the modifications require
//   division by x and/or 1-x, so cannot be used except where necessary.
// 
double
Density::dbeta(const double x, const double a, const double b,
               const bool give_log) {
  double f, p;
  volatile double am1, bm1; /* prevent roundoff trouble on some
                               platforms */

  if (a < 1) {
    if (b < 1) {  /* a,b < 1 */
      f = a * b / ((a + b) * x * (1 - x));
      p = dbinom_raw(a, a + b, x, 1 - x, give_log);
    } else {   /* a < 1 <= b */
      f = a / x;
      bm1 = b - 1;
      p = dbinom_raw(a, a + bm1, x, 1 - x, give_log);
    }
  } else {
    if (b < 1) {  /* a >= 1 > b */
      f = b / (1 - x);
      am1 = a - 1;
      p = dbinom_raw(am1, am1 + b, x, 1 - x, give_log);
    } else {   /* a,b >= 1 */
      f = a + b - 1;
      am1 = a - 1;
      bm1 = b - 1;
      p = dbinom_raw(am1, am1 + bm1, x, 1 - x, give_log);
    }
  }
  return ((give_log) ? p + log(f) : p*f);
}

/// Density of the binomial distribution.
///
/// \f[ f(k) = {n \choose k}p^k (1-p)^{n-k} \f]
///
/// Returns probability of getting k successes in n binomial trials
/// with a probability p of success on each trial, if give_log == false. 
/// If give_log == true, returns the natural logarithm of the probability.
///
/// \param k          Number of successes
/// \param n          Number of trials
/// \param p          Probability of success on each trial
/// \param give_log   Return log probability?
///
/// \f[ \mbox{E}(x) = np \f]
/// \f[ \mbox{Var}(x) = np(1-p) \f]
///
/// The implementation uses dbinom_raw from R
///
double
Density::dbinom(const int k, const int n, double p, bool give_log) {
  return dbinom_raw(k, n, p, 1.0 - p, give_log);
}

/// Density of the Cauchy distribution
///
/// \f[ f(x) = \frac{1}{\pi\mbox{s} (1 + (\frac{x-\mbox{l}}{\mbox{s}})^2)} \f]
///
/// \param x     the x value at which the density is to be calculated
/// \param l     the location parameter
/// \param s     the scale parameter
/// \param give_log  return natural log of density if true
///
/// The expectation and variance of the Cauchy distribution are infinite.
/// The mode is equal to the location parameter.
///
/// This implementation is adapted from R v2.0. The sanity checks for
/// s > 0 and ISNAN(x) have been removed.
///
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
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  DESCRIPTION
//
//    The density of the Cauchy distribution.
//
double 
Density::dcauchy(const double x, const double l, const double s, 
                 const bool give_log) 
{
  double y;
  y = (x - l) / s;
  return give_log ? - log(M_PI*s*(1. + y*y)) : 1./(M_PI*s*(1. + y*y));
}

/// Density of the chi-squared distribution
///
/// \f[ f(x) = \frac{1}{2^{n/2}\Gamma(n/2)}x^{n/2-1}e^{-x/2} \f]
///
/// \param x         the chi-squared variate whose density is desired
/// \param n         degrees of freedom for the chi-squared density
/// \param give_log  true if natural logarithm of density is desired
///
/// \f[ \mbox{E}(x) = \mbox{n} \f]
/// \f[ \mbox{Var}(x) = 2\mbox{n} \f]
///
/// From R v2.0. The implementation simply calls dgamma with shape = n/2 
/// and scale = 2.
///
//  Mathlib : A C Library of Special Functions
//  Copyright (C) 1998 Ross Ihaka
//  Copyright (C) 2000 The R Development Core Team
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful, but
//  WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
//  USA
//
//  DESCRIPTION
//
//    The density of the chi-squared distribution.
//
double 
Density::dchisq(const double x, const double n, const bool give_log) {
  return dgamma(x, n / 2., 2., give_log);
}

/// Density of the Dirichlet distribution
///
/// A brute-force implementation of the Dirichlet density:
///
/// \f[f({\bf p}) 
/// = \Gamma(\sum_k a_k)\prod_k \frac{p_k^{a_k-1}}{\Gamma(a_k)}\f]
///
/// \param p             Vector of probabilities
/// \param a             Vector of Dirichlet parameters
/// \param give_log      Return log density? 
/// \param include_const Include normalizing constant
///
double
Density::ddirch(const std::vector<double>& p, const std::vector<double>& a,
                const bool give_log, const bool include_const) 
{
  double l = 0.0;
  int nElem = p.size();
  if (include_const) {
    double sum = 0.0;
    for (unsigned k = 0; k < nElem; k++) {
      sum += a[k];
      l -= gamln(a[k]);
    }
    l += gamln(sum);
  }
  for (unsigned k = 0; (k < nElem) && (l > Util::log_dbl_min); k++) {
    l += (a[k] - 1.0) * log(p[k]);
  }
  return ((give_log) ? l : exp(l));
}

/// Exponential density
///
/// \f[ f(x) = \frac{1}{b}e^{-x/b} \f]
///
/// \param x         the exponential variate
/// \param b         the parameter of the exponential density
/// \param give_log  return log density if true
/// \return          0 or Util::log_dbl_min if x < 0
///
/// \f[ \mbox{E}(x) = \mbox{b} \f]
/// \f[ \mbox{Var}(x) = \mbox{b} \f]
///
/// Derived from R v2.0. Does not propagate NaNs. Does not check to ensure
/// b > 0. 
///
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
//  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
//  DESCRIPTION
//
//	The density of the exponential distribution.
//
double 
Density::dexp(const double x, const double b, const bool give_log) {
  if (x < 0.0) {
    return rd0(give_log);
  }
  return (give_log ? (-x/b) - log(b) : exp(-x/b)/b);
}

/// Density of the F distribution
///
/// \f[ f(x) = \frac{\Gamma((m+n)/2)}{\Gamma(m/2)\Gamma(n/2)}(m/n)^{m/2}x^{m/2-1}(1+(m/n)x)^{-(m+n)/2} \f]
///
/// \param x        the F variate
/// \param m        ``numerator'' degrees of freedom
/// \param n        ``denominator'' degrees of freedom
/// \param give_log return log density if true
///
/// \f[ \mbox{E}(x) = \frac{m}{m-2}, m > 2 \f]
/// \f[ \mbox{Var}(x) = \frac{2m^2(n-2)}{n(m+2)}, n > 2 \f]
///
/// Derived from R v2.0. Does not do isnan() check on arguments. Does
/// not check m > 0 and n > 0. Callers must ensure that these conditions
/// are met.
///
//
//  AUTHOR
//    Catherine Loader, catherine@research.bell-labs.com.
//    October 23, 2000.
//
//  Merge in to R:
//	Copyright (C) 2000, The R Core Development Team
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
//    The density function of the F distribution.
//    To evaluate it, write it as a Binomial probability with p = x*m/(n+x*m). 
//    For m >= 2, we use the simplest conversion.
//    For m < 2, (m-2)/2 < 0 so the conversion will not work, and we must use
//               a second conversion. 
//    Note the division by p; this seems unavoidable
//    for m < 2, since the F density has a singularity as x (or p) -> 0.
//
double 
Density::df(const double x, const double m, const double n, 
            const bool give_log) 
{
  double p, q, f, dens;
  if (x <= 0.) return rd0(give_log);
  
  f = 1./(n+x*m);
  q = n*f;
  p = x*m*f;
  
  if (m >= 2) { 
    f = m*q/2;
    dens = dbinom_raw((m-2)/2, (m+n-2)/2, p, q, give_log);
  } else { 
    f = m*m*q / (2*p*(m+n));
    dens = dbinom_raw(m/2, (m+n)/2, p, q, give_log);
  }
  return(give_log ? log(f)+dens : f*dens);
}

/// Density of the gamma distribution.
///
/// Returns the probability density associated with a gamma variate:
///
/// \f[f(x) = \frac{1}{s^{a} \Gamma(a)} x^{a-1} e^{-x/s}\f]
///
/// \param x          A gamma variate (x)
/// \param shape      Shape of the distribution (a)
/// \param scale      Scale of the distribution (s)
/// \param give_log   Return log density?
///
/// \f[ \mbox{E}(x) = sa \f]
/// \f[ \mbox{Var}(x) = s^2a \f]
///
/// The implementation is derived from R.
///
double 
Density::dgamma(const double x, const double shape, const double scale,
                const bool give_log)
{ 
  // should return error if shape <= 0 || scale <= 0
  if (x < 0) {
    return rd0(give_log);
  }
  if (x == 0) {
    if (shape < 1) {
      return dbl_max; 
    }
    if (shape > 1) {
      return rd0(give_log);
    }
    // shape == 1
    return give_log ? -log(scale) : 1/scale;
  }
  double pr;
  if (shape < 1) { 
    pr = dpois_raw(shape, x/scale, give_log);
    return give_log ?  pr + log(shape/x) : pr*shape/x;
  }
  /* else  shape >= 1 */
  pr = dpois_raw(shape-1, x/scale, give_log);
  return give_log ? pr - log(scale) : pr/scale;
}

/// Density of the geometric distribution
/// 
/// \f[ f(x) = p(1-p)^x \f]
///
/// \param x         the (integer) geometric variate
/// \param p         the parameter of the geometric distribution
/// \param give_log  return log density if true
///
/// \f[ \mbox{E}(x) = \frac{1-p}{p} \f]
/// \f[ \mbox{Var}(x) = \frac{1-p}{p^2} \f]
///
/// Derived from R v2.0. Does not check isnan() on x and p. Does not
/// check for 0 < p < 1. Changed x to unsigned int, so check on it is 
/// no longer required.
///
//
//  AUTHOR
//    Catherine Loader, catherine@research.bell-labs.com.
//    October 23, 2000.
//
//  Merge in to R:
//	Copyright (C) 2000, 2001 The R Core Development Team
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
//    Computes the geometric probabilities, Pr(X=x) = p(1-p)^x.
//
double 
Density::dgeom(const unsigned x, const double p, const bool give_log) { 
  double prob;
  if (x < 0 || p == 0) return rd0(give_log);
  /* prob = (1-p)^x, stable for small p */
  prob = dbinom_raw(0.,x, p,1-p, give_log);
  return((give_log) ? log(p) + prob : p*prob);
}


/// Density of the hypergeometric distribution.
///
/// \f[ f(x) = \frac{{r \choose x}{b \choose n-x}}{{r+b \choose n}} \f]
///
/// Returns the probability of choosing x white balls in a sample of size n
/// from an urn with r white balls and b black balls (sampling without
/// replacement.
///
/// \param x        The number of white balls in the sample
/// \param r        The number of white balls in the urn
/// \param b        The number of black balls in the urn
/// \param n        The sample size
/// \param giveLog  Return log probability?
///
/// \f[ \mbox{E}(x) = n(\frac{r}{r+b}) \f]
/// \f[ \mbox{Var}(x) = \frac{n(\frac{r}{r+b})(1-\frac{r}{r+b})((r+b)-n)}{r+b-1} \f]
///
/// The code is modified from R v1.8.1 to take unsigned integer arguments 
/// rather than doubles.
///
// AUTHOR
//   Catherine Loader, catherine@research.bell-labs.com.
//   October 23, 2000.
//
// Merge in to R:
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
// DESCRIPTION
//
//   Given a sequence of r successes and b failures, we sample n (<= b+r)
//   items without replacement. The hypergeometric probability is the
//   probability of x successes:
//
//                  dbinom(x,r,p) * dbinom(n-x,b,p)
//     p(x;r,b,n) = ---------------------------------
//                            dbinom(n,r+b,p)
//
//   for any p. For numerical stability, we take p=n/(r+b); with this choice,
//   the denominator is not exponentially small.
//
double 
Density::dhyper(const unsigned x, const unsigned r, const unsigned b, 
                const unsigned n, bool giveLog) 
{
  if (n < x || r < x || n - x > b) {
    return rd0(giveLog);
  }
  if (n == 0) {
    return (x == 0) ? rd1(giveLog) : rd0(giveLog);
  }

  double p = static_cast<double>(n)/static_cast<double>(r+b);
  double q = static_cast<double>(r+b-n)/static_cast<double>(r+b);

  double p1 = dbinom_raw(x,  r, p,q,giveLog);
  double p2 = dbinom_raw(n-x,b, p,q,giveLog);
  double p3 = dbinom_raw(n,r+b, p,q,giveLog);

  return (giveLog) ? p1 + p2 - p3 : p1*p2/p3;
}

/// Density of the inverse gamma distribution.
///
/// Returns the probability density associated with an inverse gamma variate:
///
/// \f[f(y) = f(1/x) \f]
/// \f[f(y) = \frac{s^a}{\Gamma(a)} y^{-(a+1)} e^{-s/y}\f]
///
/// \param y          An inverse gamma variate (y)
/// \param shape      Shape of the distribution (a)
/// \param scale      Scale of the distribution (s)
/// \param give_log   Return log density?
///
/// \f[ \mbox{E}(x) = \frac{s}{a-1} \mbox{ for } a > 1 \f]
/// \f[ \mbox{Var}(x) = \frac{s^2}{(a-1)^2(a-2)} \mbox{ for } a > 2 \f]
///
/// The implementation is based on the R implementation of dgamma().
/// It first calculates
///
/// \f[ p(y) = \frac{(s/y)^a e^{-s/y}}{\Gamma(a)} \f]
///
/// \f$f(y)\f$ is then given by
///
/// \f[ f(y) = \frac{p(y)}{y} \f]
///
double 
Density::dinvgamma(const double y, const double shape, const double scale,
                   const bool give_log)
{
  // should return error if shape <= 0 || scale <= 0
  if (y < 0) {
    return rd0(give_log);
  }
  double l  = scale/y;
  if (true) { // using dpois_raw from R
    double pr = dpois_raw(shape, l, give_log);
    return give_log ? pr + log(shape) - log(y) : pr*shape/y;
  } else { // my brute force approach
    double logP = shape*log(l) - l - gamln(shape);
    return give_log ? logP - log(y) : exp(logP)/y;
  }
}

/// Density of the lognormal distribution
///
/// \f[ f(x) = \frac{1}{\sigma x\sqrt{2\pi}}e^{-\frac{(\log(x) - \mu)^2}{2\sigma^2}} \f]
///
/// \param x         the lognormal variate
/// \param mu        logarithm of the mean of the corresponding normal (\f$\mu\f$)
/// \param sigma     logarithm of the sd of the corresponding normal (\f$\sigma\f$)
/// \param give_log  return log density if true
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
//    The density of the lognormal distribution.
//
double 
Density::dlnorm(const double x, const double mu, const double sigma, 
                const bool give_log)
{
  double y;
  if(x <= 0) return rd0(give_log);
  y = (log(x) - mu) / sigma;
  return (give_log ?
          -(M_LN_SQRT_2PI   + 0.5 * y * y + log(x * sigma)) :
          M_1_SQRT_2PI * exp(-0.5 * y * y)  /	 (x * sigma));
}

/// Density of the logistic distribution
///
/// \f[ f(x) = \frac{1}{s}\frac{e^{\frac{x-m}{s}}}{(1 + e^{\frac{x-m}{s}})^2} \f]
///
/// or equivalently (dividing numerator and denominator by \f$e^{2\frac{x-m}{s}}\f$)
///
/// \f[ f(x) = \frac{1}{s}\frac{e^{\frac{-(x-m)}{s}}}{(1 + e^{\frac{-(x-m)}{s}})^2} \f]
///
/// \param x         the logistic variate
/// \param m         the location parameter
/// \param s         the scale parameter
/// \param give_log  return log density if true
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
Density::dlogis(const double x, const double m, const double s, 
                const bool give_log)
{
  double e, f;
  double y = (x - m) / s;
  e = exp(-y);
  f = 1.0 + e;
  return give_log ? -(x + log(s * f * f)) : e / (s * f * f);
}


/// Density of the multinomial distribution
///
/// A brute-force implementation of the multinomial density
///
/// \f[ f({\bf n}) = {\sum_i n_i \choose n_1 \dots n_I}\prod_i p_i^{n_i} \f]
///
/// \param n                  Vector of observations
/// \param p                  Vector of probabilities
/// \param give_log           Return log probability
/// \param include_factorial  Leave out combinatorial coefficient?
///
double
Density::dmulti(const std::vector<int>& n, const std::vector<double>& p,
                const bool give_log, const bool include_factorial) 
{
  unsigned nElem = n.size();
  double l = 0.0;
  if (include_factorial) {
    int sum = 0;
    for (unsigned k = 0; k < nElem; k++) {
      sum += n[k];
    }
    l = gamln(sum + 1.0);
  }
  for (unsigned k = 0; k < nElem; k++) {
    if (include_factorial) {
      l -= gamln(n[k] + 1.0);
    }
    if (p[k] >= Util::dbl_min) {
      l += n[k] * log(p[k]);
    } else {
      l += n[k] * Util::log_dbl_min;
    }
  }
  return ((give_log) ? l : exp(l));
}

/// Density of the negative binomial distribution
///
/// \f[ f(x) = \frac{\Gamma(x+n)}{\Gamma(n)x!}p^n(1-p)^x \f]
///
/// \param x         the negative binomial variate
/// \param n         the ``size'' parameter
/// \param p         the ``probability'' parameter
/// \param give_log  return log density if true
///
/// \f[ \mbox{E}(x) = \frac{x(1-p)}{p} \f]
/// \f[ \mbox{Var}(x) = \frac{x(1-p)}{p^2} \f]
///
/// Derived from R v2.0. Does not check isnan() on arguments. Does not
/// check 0 < p < 1 or n > 0. Allows non-integer n (as in R). Integer
/// checks for x not needed, since it is passed as unsigned.
//
//  AUTHOR
//    Catherine Loader, catherine@research.bell-labs.com.
//    October 23, 2000 and Feb, 2001.
//
//  Merge in to R:
//	Copyright (C) 2000--2001, The R Core Development Team
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
//
// DESCRIPTION
//
//   Computes the negative binomial distribution. For integer n,
//   this is probability of x failures before the nth success in a
//   sequence of Bernoulli trials. We do not enforce integer n, since
//   the distribution is well defined for non-integers,
//   and this can be useful for e.g. overdispersed discrete survival times.
//
double 
Density::dnbinom(const unsigned x, const double n, const double p, 
                 const bool give_log)
{ 
  double prob;
  if (x < 0) return rd0(give_log);

  prob = dbinom_raw(n, x+n, p, 1-p, give_log);
  double q = (static_cast<double>(n))/(n+x);
  return((give_log) ? log(q) + prob : q * prob);
}

/// Density of the normal distribution.
///
/// Returns the probability density associated with a normal variate:
///
/// \f[f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2} \f]
///
/// \param x_in       A normal variate (x)
/// \param mu         Mean (\f$\mu\f$)
/// \param sigma      Standard deviation (\f$\sigma\f$)
/// \param give_log   Return log density?
///
/// \f[ \mbox{E}(x) = \mu \f]
/// \f[ \mbox{Var}(x) = \sigma^2 \f]
///
/// This implementation is derived from Mathlib via R.
///
// Mathlib : A C Library of Special Functions
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
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
// double dnorm4(double x, double mu, double sigma, int give_log)
//      {dnorm (..) is synonymous and preferred inside R}
//
// DESCRIPTION
//
// ompute the density of the normal distribution.
// 
double
Density::dnorm(const double x_in, const double mu, const double sigma,
               const bool give_log) {
  double x = (x_in - mu) / sigma;
  return (give_log ? -(M_LN_SQRT_2PI + 0.5*x*x + log(sigma))
          : M_1_SQRT_2PI*exp( -0.5*x*x) / sigma);
}

/// Density of the Poisson distribution
///
/// \f[ f(x) = \frac{\lambda^x e^{-\lambda}}{x!} \f]
///
/// \param x        the Poisson variate
/// \param lambda   the Poisson parameter (\f$\lambda\f$)
/// \param give_log return log density if true
///
/// \f[ \mbox{E}(x) = \lambda \f]
/// \f[ \mbox{Var}(x) = \lambda \f]
///
/// Derived from R v2.0. No isnan() checks. Integer check on x discarded
/// since it is unsigned.
//
//  AUTHOR
//    Catherine Loader, catherine@research.bell-labs.com.
//    October 23, 2000.
//
//  Merge in to R:
//	Copyright (C) 2000, The R Core Development Team
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
//
// DESCRIPTION
//
//    dpois() checks argument validity and calls dpois_raw().
//
// N.B: Above description no longer true. Argument checking is not done.
//
//
double 
Density::dpois(const unsigned x, const double lambda, const bool give_log) {
  if (x < 0) return rd0(give_log);
  return( dpois_raw(x,lambda,give_log) );
}

/// Density of Student's t distribution
///
/// \f[ f(x) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\pi\nu}\Gamma(\frac{\nu}{2})(1 + \frac{x^2}{\nu})^{(\nu + 1)/2}} \f]
///
/// \param x         the t deviate
/// \param n         the degrees of freedom (\f$\nu\f$)
/// \param give_log  return log density if true
///
/// \f[ \mbox{E}(x) = 0 \quad , \quad \nu > 1 \f]
/// \f[ \mbox{Var}(x) = \frac{\nu}{\nu - 2} \quad , \quad \nu > 2 \f]
///
/// Derived from R v2.0. Does not check isnan() or isfinite() on arguments.
/// Does not verify n >= 0.
//
//  AUTHOR
//    Catherine Loader, catherine@research.bell-labs.com.
//    October 23, 2000.
//
//  Merge in to R:
//	Copyright (C) 2000, The R Core Development Team
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
//
// DESCRIPTION
//
//    The t density is evaluated as
//         sqrt(n/2) / ((n+1)/2) * Gamma((n+3)/2) / Gamma((n+2)/2).
//             * (1+x^2/n)^(-n/2)
//             / sqrt( 2 pi (1+x^2/n) )
//
//    This form leads to a stable computation for all
//    values of n, including n -> 0 and n -> infinity.
//
double 
Density::dt(const double x, const double n, const bool give_log) {
  double t, u;
  t = -bd0(n/2.,(n+1)/2.) + stirlerr((n+1)/2.) - stirlerr(n/2.);
  if ( x*x > 0.2*n ) {
    u = log( 1+ x*x/n ) * n/2;
  } else {
    u = -bd0(n/2.,(n+x*x)/2.) + x*x/2.;
  }
  return rdfexp(M_2PI*(1+x*x/n), t-u, give_log);
}

/// Density of the Weibull distribution
///
/// \f[ f(x) = (\frac{a}{b})(\frac{x}{b})^{a-1}e^{-\frac{x}{b}^a} \f]
///
/// \param x        the Weibull variate
/// \param a        the ``shape'' parameter
/// \param b        the ``scale'' parameter
/// \param give_log return log density if true
///
/// \f[ \mbox{E}(x) = b\Gamma(1 + \frac{1}{a}) \f]
/// \f[ \mbox{Var}(x) = b^2(\Gamma(1+\frac{2}{a}) - \Gamma(1+\frac{1}{a})^2) \f]
///
/// Derived from R v2.0. Does not check isnan() on arguments. Does not
/// check for finite x.
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
//    The density function of the Weibull distribution.
//
#include <iostream>
double 
Density::dweibull(const double x, const double a, const double b, 
                  const bool give_log)
{
  double tmp1, tmp2;
  if (x < 0) return rd0(give_log);
  tmp1 = pow(x / b, a - 1);
  tmp2 = tmp1 * (x / b);
  return  give_log ? -tmp2 + log(a * tmp1 / b) : a * tmp1 * exp(-tmp2) / b;
}



/// Natural log of the gamma function
///
/// \param x 
///
/// from
///
/// NIST Guide to Available Math Software. 
/// Source for module GAMLN from package CMLIB. 
/// Retrieved from TIBER on Wed Apr 29 17:30:20 1998. 
// ===================================================================== 
//    WRITTEN BY D. E. AMOS, SEPTEMBER, 1977. 
// 
//    REFERENCES 
//        SAND-77-1518 
//
//        COMPUTER APPROXIMATIONS BY J.F.HART, ET.AL., SIAM SERIES IN 
//        APPLIED MATHEMATICS, WILEY, 1968, P.135-136. 
//
//        NBS HANDBOOK OF MATHEMATICAL FUNCTIONS, AMS 55, BY 
//        M. ABRAMOWITZ AND I.A. STEGUN, DECEMBER. 1955, P.257. 
//
//    ABSTRACT 
//        GAMLN COMPUTES THE NATURAL LOG OF THE GAMMA FUNCTION FOR 
//        X.GT.0. A RATIONAL CHEBYSHEV APPROXIMATION IS USED ON 
//        8.LT.X.LT.1000., THE ASYMPTOTIC EXPANSION FOR X.GE.1000. AND 
//        A RATIONAL CHEBYSHEV APPROXIMATION ON 2.LT.X.LT.3. FOR 
//        0.LT.X.LT.8. AND X NON-INTEGRAL, FORWARD OR BACKWARD 
//        RECURSION FILLS IN THE INTERVALS  0.LT.X.LT.2 AND 
//        3.LT.X.LT.8. FOR X=1.,2.,...,100., GAMLN IS SET TO 
//        NATURAL LOGS OF FACTORIALS. 
//
//    DESCRIPTION OF ARGUMENTS 
//
//        INPUT 
//          X      - X.GT.0 
//
//        OUTPUT 
//          GAMLN  - NATURAL LOG OF THE GAMMA FUNCTION AT X 
//
//    ERROR CONDITIONS 
//        IMPROPER INPUT ARGUMENT - A FATAL ERROR 
// 
double
Density::gamln(const double x) {
  static double xlim1 = 8.0;
  static double xlim2 = 1e3;
  static double rtwpil = .918938533204673;
  static double p[5] =
    { 7.66345188e-4, -5.9409561052e-4,
      7.936431104845e-4, -.00277777775657725,
      .0833333333333169
    };
  static double q[2] =
    { -.00277777777777778, .0833333333333333 }
  ;
  static double pcoe[9] =
    { .00297378664481017,
      .0092381945590276, .109311595671044, .398067131020357,
      2.15994312846059, 6.33806799938727,
      20.7824725317921, 36.0367725300248, 62.0038380071273
    }
  ;
  static double qcoe[4] =
    { 1.0, -8.90601665949746,
      9.82252110471399, 62.003838007127
    };
  static double gln[100] =
    { 0., 0., .693147180559945,
      1.79175946922806, 3.17805383034795,
      4.78749174278205, 6.5792512120101,
      8.52516136106541, 10.6046029027453,
      12.8018274800815, 15.1044125730755,
      17.5023078458739, 19.9872144956619,
      22.5521638531234, 25.1912211827387,
      27.8992713838409, 30.6718601060807,
      33.5050734501369, 36.3954452080331,
      39.3398841871995, 42.3356164607535,
      45.3801388984769, 48.4711813518352,
      51.6066755677644, 54.7847293981123,
      58.0036052229805, 61.261701761002,
      64.5575386270063, 67.8897431371815,
      71.257038967168, 4.6582363488302,
      78.0922235533153, 81.557959456115,
      85.0544670175815, 88.5808275421977,
      92.1361756036871, 95.7196945421432,
      99.3306124547874, 102.968198614514,
      106.631760260643, 110.320639714757,
      114.034211781462, 117.771881399745,
      121.533081515439, 125.317271149357,
      129.123933639127, 132.952575035616,
      136.802722637326, 140.673923648234,
      144.565743946345, 148.477766951773,
      152.409592584497, 156.360836303079,
      160.331128216631, 164.320112263195,
      168.327445448428, 172.352797139163,
      176.395848406997, 180.456291417544,
      184.533828861449, 188.628173423672,
      192.739047287845, 196.86618167289,
      201.009316399282, 205.168199482641,
      209.342586752537, 213.532241494563,
      217.736934113954, 221.95644181913,
      226.190548323728, 230.439043565777,
      234.701723442818, 238.978389561834,
      243.268849002983, 247.572914096187,
      251.890402209723, 256.22113555001,
      260.564940971863, 264.921649798553,
      269.29109765102, 273.673124285694,
      278.067573440366, 282.47429268763,
      286.893133295427, 291.32395009427,
      295.766601350761, 300.220948647014,
      304.686856765669, 309.164193580147,
      313.652829949879, 318.152639620209,
      322.663499126726, 327.185287703775,
      331.717887196928, 336.261181979198,
      340.815058870799, 345.379407062267,
      349.95411804077, 354.539085519441,
      359.134205369575
    };

  /* System generated locals */
  long int i__1;
  double ret_val = 0.0;

  /* Local variables */
  static double dgam;
  static long int i__;
  static double t, dx, px, qx, rx, xx;
  static long int ndx, nxm;
  static double sum, rxx;

  if (x <= 0.0) {
    goto L90;
  } else {
    goto L5;
  }
 L5:
  ndx = static_cast<long>(x);
  t = x - static_cast<double>(ndx);
  if (t == 0.0) {
    goto L51;
  }
  dx = xlim1 - x;
  if (dx < 0.0) {
    goto L40;
  }
  /*     RATIONAL CHEBYSHEV APPROXIMATION ON 2.LT.X.LT.3 FOR GAMMA(X) */
  nxm = ndx - 2;
  px = pcoe[0];
  for (i__ = 2; i__ <= 9; ++i__) {
    /* L10: */
    px = t * px + pcoe[i__ - 1];
  }
  qx = qcoe[0];
  for (i__ = 2; i__ <= 4; ++i__) {
    /* L15: */
    qx = t * qx + qcoe[i__ - 1];
  }
  dgam = px / qx;
  if (nxm > 0) {
    goto L22;
  }
  if (nxm == 0) {
    goto L25;
  }
  /*     BACKWARD RECURSION FOR 0.LT.X.LT.2 */
  dgam /= t + 1.0;
  if (nxm == -1) {
    goto L25;
  }
  dgam /= t;
  ret_val = log (dgam);
  return ret_val;
  /*     FORWARD RECURSION FOR 3.LT.X.LT.8 */
 L22:
  xx = t + 2.0;
  i__1 = nxm;
  for (i__ = 1; i__ <= i__1; ++i__) {
    dgam *= xx;
    /* L24: */
    xx += 1.0;
  }
 L25:
  ret_val = log (dgam);
  return ret_val;
  /*     X.GT.XLIM1 */
 L40:
  rx = 1.0 / x;
  rxx = rx * rx;
  if (x - xlim2 < 0.0) {
    goto L41;
  }
  px = q[0] * rxx + q[1];
  ret_val = px * rx + (x - 0.5) * log (x) - x + rtwpil;
  return ret_val;
  /*     X.LT.XLIM2 */
 L41:
  px = p[0];
  sum = (x - 0.5) * log (x) - x;
  for (i__ = 2; i__ <= 5; ++i__) {
    px = px * rxx + p[i__ - 1];
    /* L42: */
  }
  ret_val = px * rx + sum + rtwpil;
  return ret_val;
  /*     TABLE LOOK UP FOR INTEGER ARGUMENTS LESS THAN OR EQUAL 100. */
 L51:
  if (ndx > 100) {
    goto L40;
  }
  ret_val = gln[ndx - 1];
  return ret_val;
 L90:
  return ret_val;
}

/// Logarithm of n choose k.
///
/// Formula:
///
/// \f[\log(\gamma(n+1)) - \log(\gamma(k+1)) - log(\gamma(n-k+1))\f]
///
/// \param n   Sample size
/// \param k   Number of successes
///
double 
Density::logChoose(const double n, const double k) {
  return gamln(n+1) - gamln(k+1) - gamln(n-k+1);
}

/// Logarithm of \f$\beta(a,b)\f$.
///
/// Formula:
///
/// \f[\log(\Gamma(a)) + \log(\Gamma(b)) - \log(\Gamma(a+b))\f]
///
/// \param a
/// \param b
///
double 
Density::lbeta(const double a, const double b) {
  return gamln(a) + gamln(b) - gamln(a + b);
}

/// Entropy of a beta distribution with parameters a and b.
///
/// Formula:
///
/// \f[(a-1)(\Psi(a) - \Psi(a+b)) + (b-1)(\Psi(b) - \Psi(a+b))
///   - \log(\beta(a, b))\f]
///
/// \param a  first parameter of the beta distribution
/// \param b  second parameter of the beta distribution
///
/// \f$\Psi(x)\f$ is Euler's psi function (also known as the digamma
/// function).
///
double 
Density::BetaEntropy(const double a, const double b) {
  return (a - 1)*(digamma(a) - digamma(a + b))
    + (b - 1)*(digamma(b) - digamma(a + b)) - lbeta(a, b);
}

