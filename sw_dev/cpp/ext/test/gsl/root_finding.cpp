//#include "stdafx.h"
#include <gsl/gsl_roots.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <iomanip>
#include <cmath>


namespace {
namespace local {

struct quadratic_params
{
	double a, b, c;
};

double quadratic(double x, void *params)
{
	struct quadratic_params *p = (struct quadratic_params *)params;
	const double a = p->a;
	const double b = p->b;
	const double c = p->c;
	return (a * x + b) * x + c;
}

double quadratic_df(double x, void *params)
{
	struct quadratic_params *p = (struct quadratic_params *)params;
	const double a = p->a;
	const double b = p->b;
	const double c = p->c;
	return 2.0 * a * x + b;
}

void quadratic_fdf(double x, void *params, double *y, double *dy)
{
	struct quadratic_params *p = (struct quadratic_params *)params;
	const double a = p->a;
	const double b = p->b;
	const double c = p->c;
	*y = (a * x + b) * x + c;
	*dy = 2.0 * a * x + b;
}

void one_dim_root_finding_using_f()
{
	struct quadratic_params params = { 1.0, 0.0, -5.0 };

	gsl_function func;
	func.function = &quadratic;
	func.params = (void *)&params;

	//const gsl_root_fsolver_type *T = gsl_root_fsolver_bisection;
	//const gsl_root_fsolver_type *T = gsl_root_fsolver_falsepos;
	const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
	gsl_root_fsolver *s = gsl_root_fsolver_alloc(T);

	double x_lo = 0.0, x_hi = 5.0;
	gsl_root_fsolver_set(s, &func, x_lo, x_hi);

	std::cout << "===== using " << gsl_root_fsolver_name(s) << " method =====" << std::endl;
	std::cout << std::setw(5) << "iter" << " [" << std::setw(9) << "lower" << ", " << std::setw(9) << "upper" << "] " << std::setw(9) << "root" << std::setw(11) << "err" << std::setw(10) << "err(est)" << std::endl;

	int status;
	int iter = 0, max_iter = 100;
	double r = 0, r_expected = std::sqrt(5.0);
	do
	{
		++iter;

		status = gsl_root_fsolver_iterate(s);
		r = gsl_root_fsolver_root(s);
		x_lo = gsl_root_fsolver_x_lower(s);
		x_hi = gsl_root_fsolver_x_upper(s);
		status = gsl_root_test_interval(x_lo, x_hi, 0, 0.001);

		if (GSL_SUCCESS == status)
			std::cout << "converged" << std::endl;

		std::cout << std::setw(5) << iter << " [" << std::setw(9) << x_lo << ", " << std::setw(9) << x_hi << "] " << std::setw(9) << r << std::setw(11) << (r - r_expected) << std::setw(10) << (x_hi - x_lo) << std::endl;
	} while (GSL_CONTINUE == status && iter < max_iter);

	if (GSL_SUCCESS != status)
		std::cout << "not converged" << std::endl;
}

void one_dim_root_finding_using_fdf()
{
	struct quadratic_params params = { 1.0, 0.0, -5.0 };

	gsl_function_fdf func;
	func.f = &quadratic;
	func.df = &quadratic_df;
	func.fdf = &quadratic_fdf;
	func.params = (void *)&params;

	const gsl_root_fdfsolver_type *T = gsl_root_fdfsolver_newton;
	//const gsl_root_fdfsolver_type *T = gsl_root_fdfsolver_secant;
	//const gsl_root_fdfsolver_type *T = gsl_root_fdfsolver_steffenson;
	gsl_root_fdfsolver *s = gsl_root_fdfsolver_alloc(T);

	double x = 5.0;
	gsl_root_fdfsolver_set(s, &func, x);

	std::cout << "===== using " << gsl_root_fdfsolver_name(s) << " method =====" << std::endl;
	std::cout << std::setw(5) << "iter" << std::setw(11) << "root" << std::setw(11) << "err" << std::setw(11) << "err(est)" << std::endl;

	int status;
	int iter = 0, max_iter = 100;
	double x0, x_expected = std::sqrt(5.0);
	do
	{
		++iter;

		x0 = x;

		status = gsl_root_fdfsolver_iterate(s);
		x = gsl_root_fdfsolver_root(s);
		status = gsl_root_test_delta(x, x0, 0, 1e-3);

		if (GSL_SUCCESS == status)
			std::cout << "converged" << std::endl;

		std::cout << std::setw(5) << iter << std::setw(11) << x << std::setw(11) << (x - x_expected) << std::setw(11) << (x - x0) << std::endl;
	} while (GSL_CONTINUE == status && iter < max_iter);

	if (GSL_SUCCESS != status)
		std::cout << "not converged" << std::endl;
}

struct rparams
{
	double a;
	double b;
};

int rosenbrock_f(const gsl_vector *x, void *params, gsl_vector *f)
{
	// f1(x, y) = a * (1 − x), f2(x, y) = b * (y − x^2) with a = 1 & b = 1
	// the solution of this system lies at (x, y) = (1, 1) in a narrow valley

	const double a = ((struct rparams *)params)->a;
	const double b = ((struct rparams *)params)->b;
	const double x0 = gsl_vector_get(x, 0);
	const double x1 = gsl_vector_get(x, 1);
	const double y0 = a * (1 - x0);
	const double y1 = b * (x1 - x0 * x0);

	gsl_vector_set(f, 0, y0);
	gsl_vector_set(f, 1, y1);

	return GSL_SUCCESS;
}

int rosenbrock_df(const gsl_vector *x, void *params, gsl_matrix *J)
{
	const double a = ((struct rparams *)params)->a;
	const double b = ((struct rparams *)params)->b;
	const double x0 = gsl_vector_get(x, 0);
	const double df00 = -a;
	const double df01 = 0;
	const double df10 = -2 * b * x0;
	const double df11 = b;

	gsl_matrix_set(J, 0, 0, df00);
	gsl_matrix_set(J, 0, 1, df01);
	gsl_matrix_set(J, 1, 0, df10);
	gsl_matrix_set(J, 1, 1, df11);

	return GSL_SUCCESS;
}

int rosenbrock_fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *J)
{
	rosenbrock_f(x, params, f);
	rosenbrock_df(x, params, J);

	return GSL_SUCCESS;
}

void multidim_root_finding_using_f()
{
	const size_t n = 2;
	struct rparams p = { 1.0, 10.0 };

#if 1
	gsl_multiroot_function func = { &rosenbrock_f, n, (void *)&p };
#else
	gsl_multiroot_function func;
	func.f = &rosenbrock_f;
	func.n = n;
	func.params = (void *)&p;
#endif

	const double x_init[2] = { -10.0, -5.0 };
	gsl_vector *x = gsl_vector_alloc(n);
	gsl_vector_set(x, 0, x_init[0]);
	gsl_vector_set(x, 1, x_init[1]);

	//const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrids;
	//const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_hybrid;
	//const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_dnewton;
	const gsl_multiroot_fsolver_type *T = gsl_multiroot_fsolver_broyden;
	gsl_multiroot_fsolver *s = gsl_multiroot_fsolver_alloc(T, 2);
	gsl_multiroot_fsolver_set(s, &func, x);

	std::cout << "===== using " << gsl_multiroot_fsolver_name(s) << " method =====" << std::endl;
	size_t iter = 0;
	std::cout << "iter = " << std::setw(3) << iter << " x = " << std::setw(10) << gsl_vector_get(s->x, 0) << std::setw(10) << gsl_vector_get(s->x, 1) << " f(x) = " << std::setw(10) << gsl_vector_get(s->f, 0) << std::setw(10) << gsl_vector_get(s->f, 1) << std::endl;

	int status;
	do
	{
		++iter;

		status = gsl_multiroot_fsolver_iterate(s);
		std::cout << "iter = " << std::setw(3) << iter << " x = " << std::setw(10) << gsl_vector_get(s->x, 0) << std::setw(10) << gsl_vector_get(s->x, 1) << " f(x) = " << std::setw(10) << gsl_vector_get(s->f, 0) << std::setw(10) << gsl_vector_get(s->f, 1) << std::endl;

		if (status)  // check if solver is stuck
			break;

		status = gsl_multiroot_test_residual(s->f, 1e-7);
	} while (GSL_CONTINUE == status && iter < 1000);

	std::cout << "status = " << gsl_strerror(status) << std::endl;

	gsl_multiroot_fsolver_free(s);
	gsl_vector_free(x);
}

void multidim_root_finding_using_fdf()
{
	const size_t n = 2;
	struct rparams p = {1.0, 10.0 };

#if 1
	gsl_multiroot_function_fdf func = { &rosenbrock_f, &rosenbrock_df, &rosenbrock_fdf, n, (void *)&p };
#else
	func.f = &rosenbrock_f;
	func.df = &rosenbrock_df;
	func.fdf = &rosenbrock_fdf;
	func.n = n;
	func.params = (void *)&p;
#endif

	double x_init[2] = { -10.0, -5.0 };
	gsl_vector *x = gsl_vector_alloc(n);
	gsl_vector_set(x, 0, x_init[0]);
	gsl_vector_set(x, 1, x_init[1]);

	//const gsl_multiroot_fdfsolver_type *T = gsl_multiroot_fdfsolver_hybridsj;
	//const gsl_multiroot_fdfsolver_type *T = gsl_multiroot_fdfsolver_hybridj;
	//const gsl_multiroot_fdfsolver_type *T = gsl_multiroot_fdfsolver_newton;
	const gsl_multiroot_fdfsolver_type *T = gsl_multiroot_fdfsolver_gnewton;
	gsl_multiroot_fdfsolver *s = gsl_multiroot_fdfsolver_alloc(T, n);
	gsl_multiroot_fdfsolver_set(s, &func, x);

	std::cout << "===== using " << gsl_multiroot_fdfsolver_name(s) << " method =====" << std::endl;
	size_t iter = 0;
	std::cout << "iter = " << std::setw(3) << iter << " x = " << std::setw(10) << gsl_vector_get(s->x, 0) << std::setw(10) << gsl_vector_get(s->x, 1) << " f(x) = " << std::setw(10) << gsl_vector_get(s->f, 0) << std::setw(10) << gsl_vector_get(s->f, 1) << std::endl;

	int status;
	do
	{
		++iter;

		status = gsl_multiroot_fdfsolver_iterate(s);
		std::cout << "iter = " << std::setw(3) << iter << " x = " << std::setw(10) << gsl_vector_get(s->x, 0) << std::setw(10) << gsl_vector_get(s->x, 1) << " f(x) = " << std::setw(10) << gsl_vector_get(s->f, 0) << std::setw(10) << gsl_vector_get(s->f, 1) << std::endl;

		if (status)
			break;

		status = gsl_multiroot_test_residual(s->f, 1e-7);
	} while (GSL_CONTINUE == status && iter < 1000);

	std::cout << "status = " << gsl_strerror(status) << std::endl;

	gsl_multiroot_fdfsolver_free(s);
	gsl_vector_free(x);
}

}  // namespace local
}  // unnamed namespace

void one_dim_root_finding()
{
	local::one_dim_root_finding_using_f();
	std::cout << std::endl;
	local::one_dim_root_finding_using_fdf();
}

void multidim_root_finding()
{
	local::multidim_root_finding_using_f();
	std::cout << std::endl;
	local::multidim_root_finding_using_fdf();
}
