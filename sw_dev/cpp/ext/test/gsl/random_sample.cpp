#include "stdafx.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <cmath>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


void random_sample_uniform();
void random_sample_gaussian();
void random_sample_poisson();
void random_sample_spherical();

void random_sample()
{
	random_sample_uniform();
	//random_sample_gaussian();
	//random_sample_poisson();
	//random_sample_spherical();
}

void random_sample_uniform()
{
	// The seed for the default generator type gsl_rng_default can be changed with the GSL_RNG_SEED environment variable to produce a different stream of variates
	// GSL_RNG_SEED=123 ./a.out

	// create a generator chosen by the environment variable GSL_RNG_TYPE
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	// print n random variates chosen from the poisson distribution with mean parameter mu
	const int m = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		const int n = 10;
		for (int i = 0; i < n; ++i)
		{
			const double k = gsl_ran_flat(r, 0.0, 1.0);
			std::cout << k << "  ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	gsl_rng_free(r);
}

void random_sample_gaussian()
{
	// The seed for the default generator type gsl_rng_default can be changed with the GSL_RNG_SEED environment variable to produce a different stream of variates
	// GSL_RNG_SEED=123 ./a.out

	// create a generator chosen by the environment variable GSL_RNG_TYPE
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	// print n random variates chosen from the poisson distribution with mean parameter mu
	const int m = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		const int n = 10;
		const double sigma = 2.0;
		for (int i = 0; i < n; ++i)
		{
			const double k = gsl_ran_gaussian(r, sigma);
			std::cout << k << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	gsl_rng_free(r);
}

void random_sample_poisson()
{
	// create a generator chosen by the environment variable GSL_RNG_TYPE
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	// print n random variates chosen from the poisson distribution with mean parameter mu
	const int m = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		const int n = 10;
		const double mu = 3.0;
		for (int i = 0; i < n; ++i)
		{
			unsigned int k = gsl_ran_poisson(r, mu);
			std::cout << k << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	gsl_rng_free(r);
}

void random_sample_spherical()
{
	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	double x = 0.0, y = 0.0, dx, dy;
	std::cout << x << ", " << y << std::endl;

	// The spherical distributions generate random vectors, located on a spherical surface
	// They can be used as random directions
	const int m = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < 10; ++i)
		{
			gsl_ran_dir_2d(r, &dx, &dy);
			x += dx;  y += dy;
			std::cout << x << ", " << y << std::endl;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	gsl_rng_free(r);
}
