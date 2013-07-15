#include <math.h>
#include <stdlib.h>

/* The following are some utility routines. */

/* This is the worst random number routine in the known universe, but
   I use it for portability. Feel free to replace it. */
double uniform_random(void)
{
  return (double) rand() / (double) RAND_MAX;
}

/* This Gaussian routine is stolen from Numerical Recipes and is their
   copyright. */

double gaussian_random(void)
{
  static int next_gaussian = 0;
  static double saved_gaussian_value;

  double fac, rsq, v1, v2;

  if (next_gaussian == 0) {
    do {
      v1 = 2.0*uniform_random()-1.0;
      v2 = 2.0*uniform_random()-1.0;
      rsq = v1*v1+v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0*log(rsq)/rsq);
    saved_gaussian_value=v1*fac;
    next_gaussian=1;
    return v2*fac;
  } else {
    next_gaussian=0;
    return saved_gaussian_value;
  }
}

double evaluate_gaussian(double val, double sigma)
{
  /* This private definition is used for portability */
  static const double Condense_PI = 3.14159265358979323846;

  return 1.0/(sqrt(2.0*Condense_PI) * sigma) *
    exp(-0.5 * (val*val / (sigma*sigma)));
}

/* End of utility routines */
