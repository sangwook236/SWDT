//#include "stdafx.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>


namespace {
namespace local {

#define _USE_SIGMA 1
#define N 40


struct data
{
	size_t n;
	double *y;
	double *sigma;
};

struct data2
{
	size_t n;
	double *y;
};

static void print_state(size_t iter, gsl_multifit_fsolver *s)
{
	printf(
		"iter: %3u x = % 15.8f % 15.8f % 15.8f |f(x)| = %g\n",
		iter,
		gsl_vector_get(s->x, 0),
		gsl_vector_get(s->x, 1),
		gsl_vector_get(s->x, 2),
		gsl_blas_dnrm2(s->f)
	);
}

static void print_state(size_t iter, gsl_multifit_fdfsolver *s)
{
	printf(
		"iter: %3u x = % 15.8f % 15.8f % 15.8f |f(x)| = %g\n",
		iter,
		gsl_vector_get(s->x, 0),
		gsl_vector_get(s->x, 1),
		gsl_vector_get(s->x, 2),
		gsl_blas_dnrm2(s->f)
	);
}

static int expb_f(const gsl_vector *x, void *dataset, gsl_vector *f)
{
	const size_t n = ((struct data *)dataset)->n;
	const double *y = ((struct data *)dataset)->y;
	const double *sigma = ((struct data *)dataset)->sigma;

	const double A = gsl_vector_get(x, 0);
	const double lambda = gsl_vector_get(x, 1);
	const double b = gsl_vector_get(x, 2);

	for (size_t i = 0; i < n; ++i)
	{
		// unknowns: A, lambda, b
		// model: fi = (Yi - yi) / sigma[i]
		//        Yi = A * std::exp(-lambda * i) + b
		const double t = i;
		const double Yi = A * std::exp(-lambda * t) + b;
#if defined(_USE_SIGMA)
		gsl_vector_set(f, i, (Yi - y[i]) / sigma[i]);
#else
		gsl_vector_set(f, i, Yi - y[i]);
#endif
	}

	return GSL_SUCCESS;
}

static int expb_df(const gsl_vector *x, void *dataset, gsl_matrix *J)
{
	const size_t n = ((struct data *)dataset)->n;
	const double *sigma = ((struct data *)dataset)->sigma;

	const double A = gsl_vector_get(x, 0);
	const double lambda = gsl_vector_get(x, 1);

	for (size_t i = 0; i < n; ++i)
	{
		// unknowns: A, lambda, b
		// Jacobian matrix J(i,j) = dfi / dxj
		//   where fi = (Yi - yi) / sigma[i]
		//         Yi = A * std::exp(-lambda * i) + b
		//         xj = the parameters (A, lambda, b)
		const double t = i;
		const double s = sigma[i];
		const double e = std::exp(-lambda * t);
#if defined(_USE_SIGMA)
		gsl_matrix_set(J, i, 0, e / s);
		gsl_matrix_set(J, i, 1, -t * A * e / s);
		gsl_matrix_set(J, i, 2, 1 / s);
#else
		gsl_matrix_set(J, i, 0, e);
		gsl_matrix_set(J, i, 1, -t * A * e);
		gsl_matrix_set(J, i, 2, 1);
#endif
	}

	return GSL_SUCCESS;
}

static int expb_fdf(const gsl_vector *x, void* dataset, gsl_vector *f, gsl_matrix *J)
{
	expb_f(x, dataset, f);
	expb_df(x, dataset, J);

	return GSL_SUCCESS;
}

void levenberg_marquardt_f_1()
{
/*
	//
	gsl_rng_env_setup();

	const gsl_rng_type *type = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(type);

	// This is the data to be fitted
	double y[N], sigma[N];
	for (size_t i = 0; i < N; ++i)
	{
		double t = i;
		y[i] = 1.0 + 5 * std::exp(-0.1 * t) + gsl_ran_gaussian(r, 0.1);
		sigma[i] = 0.1;
		printf("data: %d %g %g\n", i, y[i], sigma[i]);
	};
	gsl_rng_free(r);

	//
	const size_t n = N;
	const size_t p = 3;  // the number of independent variables, i.e. the number of components of the vector x.

	const gsl_multifit_fsolver_type *T = NULL; // gsl_multifit_fsolver_lmsder;
	gsl_multifit_fsolver *s = gsl_multifit_fsolver_alloc(T, n, p);

	gsl_multifit_function f;
	f.f = &expb_f;
	f.n = n;
	f.p = p;
	struct data d = { n, y, sigma };
	f.params = (void *)&d;

	double x_init[3] = { 1.0, 0.0, 0.0 };
	gsl_vector_view x = gsl_vector_view_array(x_init, p);

	gsl_multifit_fsolver_set(s, &f, &x.vector);

	print_state(0, s);

	//
	size_t iter = 0;
	int status;
	do
	{
		++iter;
		status = gsl_multifit_fsolver_iterate(s);
		printf("status = %s\n", gsl_strerror(status));
		print_state(iter, s);
		if (status)
			break;
		status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
	} while (status == GSL_CONTINUE && iter < 500);

	//
	gsl_matrix *covar = gsl_matrix_alloc(p, p);
	gsl_multifit_covar(s->J, 0.0, covar);

#define FIT(i) gsl_vector_get(s->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

	{
		const double chi = gsl_blas_dnrm2(s->f);
		const double dof = n - p;
		const double c = GSL_MAX_DBL(1, chi / std::sqrt(dof));
		printf("chisq/dof = %g\n", std::pow(chi, 2.0) / dof);
		printf("A = %.5f +/- %.5f\n", FIT(0), c*ERR(0));
		printf("lambda = %.5f +/- %.5f\n", FIT(1), c*ERR(1));
		printf("b = %.5f +/- %.5f\n", FIT(2), c*ERR(2));
	}

	printf("status = %s\n", gsl_strerror(status));

	//
	gsl_multifit_fsolver_free(s);
*/
}

void levenberg_marquardt_fdf_1()
{
	//
	gsl_rng_env_setup();

	const gsl_rng_type *type = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(type);

	// This is the data to be fitted
	double y[N], sigma[N];
	for (size_t i = 0; i < N; ++i)
	{
		const double t = i;
		y[i] = 1.0 + 5 * std::exp(-0.1 * t) + gsl_ran_gaussian(r, 0.1);
		sigma[i] = 0.1;
		printf("data: %d %g %g\n", i, y[i], sigma[i]);
	};
	gsl_rng_free(r);

	//
	const size_t n = N;
	const size_t p = 3;  // the number of independent variables, i.e. the number of components of the vector x.

	//const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmder;
	const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmsder;
	gsl_multifit_fdfsolver *s = gsl_multifit_fdfsolver_alloc(T, n, p);

	gsl_multifit_function_fdf f;
	f.f = &expb_f;
	f.df = &expb_df;
	//f.df = NULL;
	f.fdf = &expb_fdf;
	//f.fdf = NULL;
	f.n = n;
	f.p = p;
	struct data d = { n, y, sigma };
	f.params = (void *)&d;

	double x_init[3] = { 1.0, 0.0, 0.0 };
	gsl_vector_view x = gsl_vector_view_array(x_init, p);

	gsl_multifit_fdfsolver_set(s, &f, &x.vector);

	print_state(0, s);

	//
	size_t iter = 0;
	int status;
	do
	{
		++iter;
		status = gsl_multifit_fdfsolver_iterate(s);
		printf("status = %s\n", gsl_strerror(status));
		print_state(iter, s);
		if (status)
			break;
		status = gsl_multifit_test_delta(s->dx, s->x, 1e-4, 1e-4);
	} while (status == GSL_CONTINUE && iter < 500);

	//
	gsl_matrix *covar = gsl_matrix_alloc(p, p);
	gsl_multifit_covar(s->J, 0.0, covar);

#define FIT(i) gsl_vector_get(s->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

	{
		const double chi = gsl_blas_dnrm2(s->f);
		const double dof = n - p;
		const double c = GSL_MAX_DBL(1, chi / sqrt(dof));
		printf("chisq/dof = %g\n", pow(chi, 2.0) / dof);
		printf("A = %.5f +/- %.5f\n", FIT(0), c*ERR(0));
		printf("lambda = %.5f +/- %.5f\n", FIT(1), c*ERR(1));
		printf("b = %.5f +/- %.5f\n", FIT(2), c*ERR(2));
	}

	printf("status = %s\n", gsl_strerror(status));

	//
	gsl_multifit_fdfsolver_free(s);
}

static int objective_f(const gsl_vector *x, void *dataset, gsl_vector *f)
{
	const size_t n = ((struct data2 *)dataset)->n;
	const double *y = ((struct data2 *)dataset)->y;

	const double a0 = gsl_vector_get(x, 0);
	const double a1 = gsl_vector_get(x, 1);
	const double a2 = gsl_vector_get(x, 2);
	const double a3 = gsl_vector_get(x, 3);
	const double a4 = gsl_vector_get(x, 4);
	const double a5 = gsl_vector_get(x, 5);

	for (size_t i = 0; i < n; ++i)
	{
		// unknowns: a0, a1, a2, a3, a4, a5
		// model: fi = Yi - yi
		//        Yi = a0 + a1 * i + a2 * i^2 + a3 * i^3 + a4 * i^4 + a5 * i^5
		const double t = i;
		const double Yi = a0 + a1 * t + a2 * t*t + a3 * t*t*t + a4 * t*t*t*t + a5 * t*t*t*t*t;
		gsl_vector_set(f, i, Yi - y[i]);
	}

	return GSL_SUCCESS;
}

static int objective_df(const gsl_vector *x, void* dataset, gsl_matrix* J)
{
	const size_t n = ((struct data2 *)dataset)->n;
/*
	const double a0 = gsl_vector_get(x, 0);
	const double a1 = gsl_vector_get(x, 1);
	const double a2 = gsl_vector_get(x, 2);
	const double a3 = gsl_vector_get(x, 3);
	const double a4 = gsl_vector_get(x, 4);
	const double a5 = gsl_vector_get(x, 5);
*/
	for (size_t i = 0; i < n; ++i)
	{
		// unknowns: a0, a1, a2, a3, a4, a5
		// Jacobian matrix J(i,j) = dfi / dxj
		//   where fi = Yi - yi
		//         Yi = a0 + a1 * i + a2 * i^2 + a3 * i^3 + a4 * i^4 + a5 * i^5
		//         xj = the parameters (a0, a1, a2, a3, a4, a5)
		const double t = i;
		gsl_matrix_set(J, i, 0, 1.0);
		gsl_matrix_set(J, i, 1, t);
		gsl_matrix_set(J, i, 2, t*t);
		gsl_matrix_set(J, i, 3, t*t*t);
		gsl_matrix_set(J, i, 4, t*t*t*t);
		gsl_matrix_set(J, i, 5, t*t*t*t*t);
	}

	return GSL_SUCCESS;
}

static int objective_fdf(const gsl_vector *x, void *dataset, gsl_vector *f, gsl_matrix *J)
{
	objective_f(x, dataset, f);
	objective_df(x, dataset, J);

	return GSL_SUCCESS;
}

void levenberg_marquardt_fdf_2()
{
	//
	gsl_rng_env_setup();

	const gsl_rng_type *type = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(type);

	// This is the data to be fitted
	double y[N];
	for (size_t i = 0; i < N; ++i)
	{
		const double t = i;
		y[i] = 2.0 + 4.5 * t - 2.3 * t*t - 11.7 * t*t*t + 0.3 * t*t*t*t - 8.4 * t*t*t*t*t + gsl_ran_gaussian(r, 1.0);
		printf("data: %d %g\n", i, y[i]);
	};
	gsl_rng_free(r);

	//
	const size_t n = N;
	const size_t p = 6;  // the number of independent variables, i.e. the number of components of the vector x.

	//const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmder;
	const gsl_multifit_fdfsolver_type *T = gsl_multifit_fdfsolver_lmsder;
	gsl_multifit_fdfsolver *s = gsl_multifit_fdfsolver_alloc(T, n, p);

	gsl_multifit_function_fdf f;
	f.f = &objective_f;
	f.df = &objective_df;
	//f.df = NULL;
	f.fdf = &objective_fdf;
	//f.fdf = NULL;
	f.n = n;
	f.p = p;
	struct data2 d = { n, y };
	f.params = (void *)&d;

	double x_init[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	gsl_vector_view x = gsl_vector_view_array(x_init, p);

	gsl_multifit_fdfsolver_set(s, &f, &x.vector);

	print_state(0, s);

	//
	size_t iter = 0;
	int status;
	do
	{
		++iter;
		status = gsl_multifit_fdfsolver_iterate(s);
		printf("status = %s\n", gsl_strerror(status));
		print_state(iter, s);
		if (status)
			break;
		status = gsl_multifit_test_delta(s->dx, s->x, 1.0e-7, 1.0e-7);
	} while (status == GSL_CONTINUE && iter < 500);

	//
	gsl_matrix *covar = gsl_matrix_alloc(p, p);
	gsl_multifit_covar(s->J, 0.0, covar);

#define FIT(i) gsl_vector_get(s->x, i)
#define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

	{
/*
		for (size_t i = 0; i < n; ++i)
			printf("%.5f ", gsl_vector_get(s->f, i));
		printf("\n");
		for (size_t i = 0; i < n; ++i)
		{
			const double t = i;
			const double Yi = gsl_vector_get(s->x, 0) + gsl_vector_get(s->x, 1) * t +
				gsl_vector_get(s->x, 2) * t*t + gsl_vector_get(s->x, 3) * t*t*t +
				gsl_vector_get(s->x, 4) * t*t*t*t + gsl_vector_get(s->x, 5) * t*t*t*t*t;
			//printf("%.5f ", Yi);
			printf("%.5f ", Yi - y[i]);
		}
		printf("\n");
*/
		printf("covariance of the fit =\n");
		for (size_t i = 0; i < p; ++i)
		{
			for (size_t j = 0; j < p; ++j)
				printf("%.5f ", gsl_matrix_get(covar, i, j));
			printf("\n");
		}
		printf("\n");

		const double chi = gsl_blas_dnrm2(s->f);
		const double dof = n - p;
		const double c = GSL_MAX_DBL(1, chi / std::sqrt(dof));
		printf("chisq/dof = %g\n", std::pow(chi, 2.0) / dof);

		printf("\nsolution =\n");
		printf("a0 = %.5f +/- %.5f\n", FIT(0), c*ERR(0));
		printf("a1 = %.5f +/- %.5f\n", FIT(1), c*ERR(1));
		printf("a2 = %.5f +/- %.5f\n", FIT(2), c*ERR(2));
		printf("a3 = %.5f +/- %.5f\n", FIT(3), c*ERR(3));
		printf("a4 = %.5f +/- %.5f\n", FIT(4), c*ERR(4));
		printf("a5 = %.5f +/- %.5f\n", FIT(5), c*ERR(5));
	}

	printf("status = %s\n", gsl_strerror(status));

	//
	gsl_multifit_fdfsolver_free(s);
}

}  // namespace local
}  // unnamed namespace

namespace gsl {

void levenberg_marquardt()
{
	//local::levenberg_marquardt_f_1();  // not implemented
	//local::levenberg_marquardt_fdf_1();
	local::levenberg_marquardt_fdf_2();
}

}  // namespace gsl
