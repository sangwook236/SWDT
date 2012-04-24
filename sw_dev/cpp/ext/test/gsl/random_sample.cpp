//#include "stdafx.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <stdexcept>


namespace {
namespace local {

void random_sample_poisson()
{
	// print n random variates chosen from the poisson distribution with mean parameter mu
	const int m = 10, n = 10;
	const double mu = 3.0;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_ranlxs0;
	//gsl_rng_default = gsl_rng_ranlxs1;
	//gsl_rng_default = gsl_rng_ranlxs2;
	//gsl_rng_default = gsl_rng_ranlxd1;
	//gsl_rng_default = gsl_rng_ranlxd2;
	//gsl_rng_default = gsl_rng_ranlux;
	//gsl_rng_default = gsl_rng_ranlux389;
	//gsl_rng_default = gsl_rng_cmrg;
	//gsl_rng_default = gsl_rng_mrg;
	//gsl_rng_default = gsl_rng_taus;
	//gsl_rng_default = gsl_rng_taus2;
	//gsl_rng_default = gsl_rng_gfsr4;
	// Unix number generator algorithms
	//gsl_rng_default = gsl_rng_rand;
	//gsl_rng_default = gsl_rng_random_bsd;
	//gsl_rng_default = gsl_rng_random_libc5;
	//gsl_rng_default = gsl_rng_random_glibc2;
	//gsl_rng_default = gsl_rng_rand48;
	// other random number generator algorithms
	//gsl_rng_default = gsl_rng_ranf;
	//gsl_rng_default = gsl_rng_ranmar;
	//gsl_rng_default = gsl_rng_r250;
	//gsl_rng_default = gsl_rng_tt800;
	//gsl_rng_default = gsl_rng_vax;
	//gsl_rng_default = gsl_rng_transputer;
	//gsl_rng_default = gsl_rng_randu;
	//gsl_rng_default = gsl_rng_minstd;
	//gsl_rng_default = gsl_rng_uni;
	//gsl_rng_default = gsl_rng_uni32;
	//gsl_rng_default = gsl_rng_slatec;
	//gsl_rng_default = gsl_rng_zuf;
	//gsl_rng_default = gsl_rng_knuthran2;
	//gsl_rng_default = gsl_rng_knuthran2002;
	//gsl_rng_default = gsl_rng_knuthran;
	//gsl_rng_default = gsl_rng_borosh13;
	//gsl_rng_default = gsl_rng_fishman18;
	//gsl_rng_default = gsl_rng_fishman20;
	//gsl_rng_default = gsl_rng_lecuyer21;
	//gsl_rng_default = gsl_rng_waterman14;
	//gsl_rng_default = gsl_rng_fishman2x;
	//gsl_rng_default = gsl_rng_coveyou;

	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			unsigned int k = gsl_ran_poisson(r, mu);

			std::cout << k << "  ";
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_uniform()
{
	// The seed for the default generator type gsl_rng_default can be changed with the GSL_RNG_SEED environment variable to produce a different stream of variates
	//	GSL_RNG_TYPE="taus" GSL_RNG_SEED=123 ./a.out

	// print n random variates chosen from the uniform distribution
	const int m = 10, n = 10;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			const double k = gsl_ran_flat(r, 0.0, 1.0);

			std::cout << k << "  ";
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_univariate_normal()
{
	// The seed for the default generator type gsl_rng_default can be changed with the GSL_RNG_SEED environment variable to produce a different stream of variates
	//	GSL_RNG_TYPE="taus" GSL_RNG_SEED=123 ./a.out

	// print n random variates chosen from the univariate normal distribution
	const int m = 10, n = 10;
	const double sigma = 2.0;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			const double k = gsl_ran_gaussian(r, sigma);

			std::cout << k << "  ";
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_bivariate_normal()
{
	// The seed for the default generator type gsl_rng_default can be changed with the GSL_RNG_SEED environment variable to produce a different stream of variates.
	//	GSL_RNG_TYPE="taus" GSL_RNG_SEED=123 ./a.out

	// print n random variates chosen from the bivariate normal distribution
	const int m = 10, n = 10;
	const double sigma_x = 1.0, sigma_y = 1.0, rho = 0.9;
	double x = 0.0, y = 0.0;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			gsl_ran_bivariate_gaussian(r, sigma_x, sigma_y, rho, &x, &y);

			std::cout << '{' << x << ", " << y << '}' << std::endl;
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_multivariate_normal()
{
	throw std::runtime_error("not yet implemented");
}

void random_sample_spherical_2d()
{
	const double x = 0.0, y = 0.0;
	double dx = 0.0, dy = 0.0;

	std::cout << '{' << x << ", " << y << '}' << std::endl;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	// the spherical distributions generate random vectors, located on a circle.
	// they can be used as random directions.
	const int m = 10, n = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			// the obvious way to do this is to take a uniform random number between 0 and 2 * pi and let x and y be the sine and cosine respectively.

			// dx^2 + dy^2 = 1
#if 1
			gsl_ran_dir_2d(r, &dx, &dy);
#else
			gsl_ran_dir_2d_trig_method(r, &dx, &dy);
#endif

			std::cout << '{' << (x + dx) << ", " << (y + dy) << '}' << std::endl;
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_spherical_3d()
{
	const double x = 0.0, y = 0.0, z = 0.0;
	double dx = 0.0, dy = 0.0, dz = 0.0;

	std::cout << '{' << x << ", " << y << '}' << std::endl;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	// the spherical distributions generate random vectors, located on a sphere.
	// they can be used as random directions.
	const int m = 10, n = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			// dx^2 + dy^2 + dz^2 = 1
			gsl_ran_dir_3d(r, &dx, &dy, &dz);

			std::cout << '{' << (x + dx) << ", " << (y + dy) << ", " << (z + dz) << '}' << std::endl;
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

void random_sample_spherical_nd()
{
	const std::size_t dim = 5;
	const double x[dim] = { 0.0, };
	double dx[dim] = { 0.0, };

	std::cout << '{';
	for (std::size_t i = 0; i < dim; ++i)
	{
		if (i) std::cout << ", ";
		std::cout << x[i];
	}
	std::cout << '}' << std::endl;

#if 0
	// this function reads the environment variables GSL_RNG_TYPE and GSL_RNG_SEED and uses their values to set the corresponding library variables gsl_rng_default and gsl_rng_default_seed.
	// if you don't specify a generator for GSL_RNG_TYPE then gsl_rng_mt19937 is used as the default. the initial value of gsl_rng_default_seed is zero.
	gsl_rng_env_setup();
#else
	// random number generator algorithms
	gsl_rng_default = gsl_rng_mt19937;
	//gsl_rng_default = gsl_rng_taus;
	gsl_rng_default_seed = (unsigned long)std::time(NULL);
#endif

	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	std::cout << "generator type: " << gsl_rng_name(r) << std::endl;
	std::cout << "seed = " << gsl_rng_default_seed << std::endl;

	// the spherical distributions generate random vectors, located on a hyper-spherical surface.
	// they can be used as random directions.
	const int m = 10, n = 10;
	for (int j = 0; j < m; ++j)
	{
		//r = gsl_rng_alloc(T);  // caution: generate repetitive sample

		for (int i = 0; i < n; ++i)
		{
			// | dx | = 1
			gsl_ran_dir_nd(r, dim, dx);

			std::cout << '{';
			for (std::size_t i = 0; i < dim; ++i)
			{
				if (i) std::cout << ", ";
				std::cout << (x[i] + dx[i]);
			}
			std::cout << '}' << std::endl;
		}
		std::cout << std::endl;
	}

	gsl_rng_free(r);
}

}  // namespace local
}  // unnamed namespace

void random_sample()
{
	//local::random_sample_poisson();

	//local::random_sample_uniform();
	//local::random_sample_univariate_normal();
	//local::random_sample_bivariate_normal();
	//local::random_sample_multivariate_normal();

	local::random_sample_spherical_2d();
	local::random_sample_spherical_3d();
	//local::random_sample_spherical_nd();
}
