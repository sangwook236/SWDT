//#include "stdafx.h"
//#define HAVE_INLINE 1
#include <gsl/gsl_blas.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void vector_operation()
{
	void print_gsl_vector(gsl_vector *vec);
	void print_gsl_vector(gsl_vector *vec, const int dim);
	void print_gsl_matrix(gsl_matrix *mat);
	void print_gsl_matrix(gsl_matrix *mat, const int rdim, const int cdim);

	// vector view
	std::cout << ">>> vector view\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };

		gsl_vector_view v = gsl_vector_view_array(a, dim);

		gsl_vector_set(&v.vector, 0, 11.0);
		gsl_vector_set(&v.vector, 3, 14.0);

		// use inline function
		double *p = gsl_vector_ptr(&v.vector, 0);

		//gsl_vector_set(v, i, 2.0);
		//double val = gsl_vector_get(v, i);
		//double* pval = gsl_vector_ptr(v, i);
		//const double* cpval = gsl_vector_const_ptr(v, i);

		for (int i = 0; i < dim; ++i)
			std::cout << a[i] << ' ';
		std::cout << std::endl;
	}

	// subvector
	std::cout << "\n>>> subvector\n";
	{
		const int dim1 = 4;
		const int dim2 = 2;
		double a[] = { 1., 2., 3., 4., 5., 6. };

		gsl_vector_view v1 = gsl_vector_view_array(a, dim1);
		gsl_vector_view v2 = gsl_vector_subvector(&v1.vector, 1, dim2);

		print_gsl_vector(&v2.vector);
	}

	// vector copy & element change
	std::cout << "\n>>> vector copy & element change\n";
	{
		const int dim1 = 9;
		const int dim2 = 4;
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		double b[] = { -1., -2., -3., -4. };

		gsl_vector_view v1 = gsl_vector_view_array(a, dim1);
		gsl_vector_view v2 = gsl_vector_view_array(b, dim2);

		//gsl_vector_memcpy(&v1.vector, &v2.vector);
#if defined(__GNUC__)
        gsl_vector_view v1_roi(gsl_vector_subvector(&v1.vector, 1, dim2));
		gsl_vector_memcpy(&v1_roi.vector, &v2.vector);
#else
		gsl_vector_memcpy(&gsl_vector_subvector(&v1.vector, 1, dim2).vector, &v2.vector);
#endif

		//gsl_vector_swap(a, b);
		//gsl_vector_swap_elements(a, i, j);
		//gsl_vector_reverse(a);

		print_gsl_vector(&v1.vector);
	}

	// set value, zero, basis
	std::cout << "\n>>> set value, zero, identity\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };

		gsl_vector_view v = gsl_vector_view_array(a, dim);

		gsl_vector_set_all(&v.vector, 1.0);
		gsl_vector_set_zero(&v.vector);
		gsl_vector_set_basis(&v.vector, 2);
	}

	// vector arithmetic
	std::cout << "\n>>> vector arithmetic\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };
		double b[] = { 2., 2., 2., 2. };

		gsl_vector_view v = gsl_vector_view_array(a, dim);
		gsl_vector_view v2 = gsl_vector_view_array(b, dim);

		gsl_vector_add(&v.vector, &v2.vector);
		gsl_vector_sub(&v.vector, &v2.vector);
		gsl_vector_mul(&v.vector, &v2.vector);
		gsl_vector_div(&v.vector, &v2.vector);

		gsl_vector_scale(&v.vector, -1.0);
		gsl_vector_add_constant(&v.vector, -10.0);

		print_gsl_vector(&v.vector);
	}

	// blas level 1
	{
		// y = a x + y
		//gsl_blas_daxpy(1.0, x, y);
	}

	// min, max of vector
	std::cout << "\n>>> min, max of vector\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };

		gsl_vector_view v = gsl_vector_view_array(a, dim);

		const double maxval = gsl_vector_max(&v.vector);
		const double minval = gsl_vector_min(&v.vector);
		std::cout << maxval << ", " << minval << std::endl;
		double maxval2, minval2;
		gsl_vector_minmax(&v.vector, &minval2, &maxval2);
		std::cout << maxval2 << ", " << minval2 << std::endl;

		const size_t maxidx = gsl_vector_max_index(&v.vector);
		const size_t minidx = gsl_vector_min_index(&v.vector);
		std::cout << maxidx << ", " << minidx << std::endl;
		size_t maxidx2, minidx2;
		gsl_vector_minmax_index(&v.vector, &minidx2, &maxidx2);
		std::cout << maxidx2 << ", " << minidx2 << std::endl;

		//const CBLAS_INDEX_t maxidx = gsl_blas_idamax(v);
	}

	// vector property
	std::cout << "\n>>> vector property\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };
		gsl_vector_view v = gsl_vector_view_array(a, dim);

		std::cout << gsl_vector_isnull(&v.vector) << std::endl;
	}

	// norm
	std::cout << "\n>>> vector norm\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };
		gsl_vector_view v = gsl_vector_view_array(a, dim);

		std::cout << gsl_blas_dnrm2(&v.vector) << std::endl;
	}

	// dot
	std::cout << "\n>>> dot product\n";
	{
		const int dim = 4;
		double a[] = { -1., -2., -3., -4. };
		gsl_vector_view v = gsl_vector_view_array(a, dim);

		double dot;
		gsl_blas_ddot(&v.vector, &v.vector, &dot);
		std::cout << dot << std::endl;
	}
}
