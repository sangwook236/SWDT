//#include "stdafx.h"
#include <gsl/gsl_multimin.h>
#include <cmath>


namespace {
namespace local {

// paraboloid centered on (dp[0], dp[1])
double my_f(const gsl_vector *v, void *params)
{
	const double *dp = (double *)params;

	const double x = gsl_vector_get(v, 0);
	const double y = gsl_vector_get(v, 1);
	return 10.0 * (x - dp[0]) * (x - dp[0]) + 20.0 * (y - dp[1]) * (y - dp[1]) + 30.0;
}

// The gradient of f, df = (df/dx, df/dy)
void my_df(const gsl_vector *v, void *params, gsl_vector *df)
{
	const double *dp = (double *)params;

	const double x = gsl_vector_get(v, 0);
	const double y = gsl_vector_get(v, 1);

	gsl_vector_set(df, 0, 20.0 * (x - dp[0]));
	gsl_vector_set(df, 1, 40.0 * (y - dp[1]));
}

// compute both f and df together
void my_fdf(const gsl_vector *x, void *params, double *f, gsl_vector *df)
{
	*f = my_f(x, params);
	my_df(x, params, df);
}

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void conjugate_gradient()
{
	// position of the minimum (1, 2)
	double par[2] = { 1.0, 2.0 };

	gsl_multimin_function_fdf my_func;

	my_func.f = &local::my_f;
	my_func.df = &local::my_df;
	my_func.fdf = &local::my_fdf;
	my_func.n = 2;  // the dimension of the system, i.e. the number of components of the vectors x
	my_func.params = (void *)&par;

	// Starting point, x = (5, 7)
	gsl_vector *x = gsl_vector_alloc(2);
	gsl_vector_set(x, 0, 5.0);
	gsl_vector_set(x, 1, 7.0);

	const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_conjugate_fr;
	//const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_conjugate_pr;
	//const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs;
	gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc(T, 2);

	gsl_multimin_fdfminimizer_set(s, &my_func, x, 0.01, 1e-4);

	//
	size_t iter = 0;
	int status;
	do
	{
		++iter;
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient(s->gradient, 1e-7);

		if (status == GSL_SUCCESS)
			printf("Minimum found at:\n");

		printf("%5d %.5f %.5f %10.5f\n", iter,
			gsl_vector_get(s->x, 0),
			gsl_vector_get(s->x, 1),
			s->f
		);
	} while (status == GSL_CONTINUE && iter < 100);

	//
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(x);
}

}  // namespace my_gsl
