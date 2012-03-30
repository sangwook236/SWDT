//#include "stdafx.h"
#include <gsl/gsl_siman.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

// set up parameters for this simulated annealing run how many points do we try before stepping
#define N_TRIES 200
// how many iterations for each T?
#define ITERS_FIXED_T 10
// max step size in random walk
#define STEP_SIZE 10
// Boltzmann constant
#define K 1.0
// initial temperature
#define T_INITIAL 0.002
// damping factor for temperature
#define MU_T 1.005
#define T_MIN 2.0e-6

gsl_siman_params_t params = { N_TRIES, ITERS_FIXED_T, STEP_SIZE, K, T_INITIAL, MU_T, T_MIN };

// now some functions to test in one dimension
double E1(void *xp)
{
	const double x = *(double *)xp;
	return std::exp(-std::pow(x-1.0, 2.0)) * std::sin(8*x);
}

double M1(void *xp, void *yp)
{
	const double x = *(double *)xp;
	const double y = *(double *)yp;
	return std::fabs(x - y);
}

void S1(const gsl_rng *r, void *xp, double step_size)
{
	const double old_x = *(double *)xp;
	const double u = gsl_rng_uniform(r);
	const double new_x = u * 2 * step_size - step_size + old_x;
	memcpy(xp, &new_x, sizeof(new_x));
}

void P1(void *xp)
{
	std::cout << *(double *)xp;
}

}  // namespace local
}  // unnamed namespace

void simulated_annealing()
{
	const double x_initial = 15.5;

	gsl_rng_env_setup();

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	gsl_siman_solve(r, (void *)&x_initial, local::E1, local::S1, local::M1, local::P1, NULL, NULL, NULL, sizeof(double), local::params);

	gsl_rng_free(r);
}
