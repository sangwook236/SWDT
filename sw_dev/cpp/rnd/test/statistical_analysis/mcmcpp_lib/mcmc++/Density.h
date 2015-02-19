///
/// \file   Density.h
/// \brief  Collects a variety of constants and function in namespace Util.
///
/// Provides definitions for a variety of functions related to numerical
/// evaluation of probability densities.
///
/// All are declared in namespace Density with an eye towards avoiding
/// naming conflicts.
///
/// \author Kent Holsinger
/// \date   2004-07-03
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

#if !defined(__DENSITY_H)
#define __DENSITY_H

// standard includes
#include <vector>

/// \namespace Density
///
/// Used to isolate functions for density evaluation.
///
/// Provides a variety of numerical density routines. All of them are
/// derived from R. Details are provided in the comments accompanying each
/// one.
///
namespace Density {

  /// Minimum value of parameters to be used in gamln()
  ///
  extern const double MinGammaPar;
  /// Maximum value of parameters to be used in gamln()
  ///
  extern const double MaxGammaPar;

  double dbeta(double x, double a, double b,
               bool give_log = false);
  // probability of choosing k from n, with probability p on each choice
  // (sampling with replacement)
  double dbinom(int k, int n, double p, bool give_log);
  double dcauchy(double x, double l, double s, bool give_log); 
  double dchisq(double x, double n, bool give_log);
  double ddirch(const std::vector<double>& p, const std::vector<double>& a,
                const bool give_log = false, const bool include_const = true);
  double dexp(double x, double b, bool give_log);
  double df(double x, double m, double n, bool give_log);
  double dgeom(unsigned x, double p, bool give_log);
  double dgamma(double x, double shape, double scale, bool give_log);
  double dhyper(unsigned x, unsigned r, unsigned b, unsigned n, bool giveLog);
  double dinvgamma(double y, double shape, double scale, bool give_log);
  double dlnorm(double x, double mu, double sigma, bool give_log);
  double dlogis(double x, double m, double s, bool give_log);
  double dmulti(const std::vector<int>& n, const std::vector<double>& p,
                bool give_log = false, bool include_factorial = false);
  double dnbinom(unsigned x, double n, double p, bool give_log);
  double dnorm(double x_in, double mu, double sigma, bool give_log);
  double dpois(unsigned x, double lambda, bool give_log);
  double dt(double x, double n, bool give_log);
  double dweibull(double x, double a, double b, bool give_log);
  double BetaEntropy(double a, double b);
  double logChoose(double n, double k);
  double gamln(double x);
  double lbeta(double a, double b);
 
}

#endif

// Local Variables: //
// mode: c++ //
// End: //

