#define BOOST_NUMERIC_BINDINGS_USE_CLAPACK 1  // in order to use clapack.lib
//#define BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK 1
//#define BIND_FORTRAN_LOWERCASE_UNDERSCORE 1  // caution !!!
//#define BOOST_NUMERIC_BINDINGS_LAPACK_2 1  // caution !!!

#include "stdafx.h"
#include <cassert>  // necessary
//#include <boost/numeric/bindings/atlas/clapack.hpp>
#include <boost/numeric/bindings/lapack/geev.hpp>
#include <boost/numeric/bindings/lapack/gesvd.hpp>
#include <boost/numeric/bindings/traits/ublas_matrix.hpp>
#include <boost/numeric/bindings/traits/ublas_vector.hpp>
#include <boost/numeric/bindings/traits/std_vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>


void numeric_bindings_lapack_ev();
void numeric_bindings_lapack_svd();
void numeric_bindings_atlas_lu();

void numeric_bindings()
{
	numeric_bindings_lapack_ev();
	numeric_bindings_lapack_svd();

	numeric_bindings_atlas_lu();
}

namespace
{

template <typename T>
void Hessenberg(boost::numeric::ublas::matrix<T, boost::numeric::ublas::column_major>& H)
{
	T k = 1;
	for (unsigned int i = 0; i < H.size1(); ++i)
	{
		for (unsigned int j = i; j <= H.size2(); ++j)
		{
			if (j > 0)
			{
				H(i,j-1) = k;
				k += 1;
			}
		}
	}
}

}  // unnamed namespace

void numeric_bindings_lapack_ev()
{
	typedef double real_type;
	typedef std::complex<real_type> complex_type;
	typedef boost::numeric::ublas::matrix<real_type, boost::numeric::ublas::column_major> matrix_type;
	typedef boost::numeric::ublas::matrix<complex_type, boost::numeric::ublas::column_major> complex_matrix_type;
	typedef boost::numeric::ublas::vector<real_type> vector_type;
	typedef boost::numeric::ublas::vector<complex_type> complex_vector_type;
/*
	{
		const size_t m = 3, n = 2;
		const size_t minmn = m < n ? m : n;

		matrix_type A(m, n);
		A(0,0) = 1.; A(0,1) = 1.;
		A(1,0) = 0.; A(1,1) = 1.;
		A(2,0) = 1.; A(2,1) = 0.;
		std::cout << "A: " << A << std::endl;

		complex_vector_type eigvals(minmn, 0.0);
		matrix_type eigvecs_left(m, n, 0.0);
		matrix_type eigvecs_right(m, n, 0.0);

		boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, (matrix_type *)0L, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, (matrix_type *)0L, boost::numeric::bindings::lapack::optimal_workspace());

		std::cout << "eigvals = " << eigvals << std::endl;
		std::cout << "eigvecs_left = " << eigvecs_left << std::endl;
		std::cout << "eigvecs_right = " << eigvecs_right << std::endl;
	}

	std::cout << std::endl;
*/
	{
		const int n = 3;

		matrix_type A(n, n, 0.0);
		A(0,0) = 1.; A(0,1) = 1.; A(0,2) = 1.;
		A(1,0) = 1.; A(1,1) = 1.; A(1,2) = 0.;
		A(2,0) = 1.; A(2,1) = 0.; A(2,2) = -2.;
		std::cout << "A = " << A << std::endl; 

		complex_vector_type eigvals(n, 0.0);
		matrix_type eigvecs_left(n, n, 0.0);
		matrix_type eigvecs_right(n, n, 0.0);

		boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, (matrix_type *)0L, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, (matrix_type *)0L, boost::numeric::bindings::lapack::optimal_workspace());

		std::cout << "eigvals = " << eigvals << std::endl;
		std::cout << "eigvecs_left = " << eigvecs_left << std::endl;
		std::cout << "eigvecs_right = " << eigvecs_right << std::endl;
	}

	std::cout << std::endl;
	{
		const int n = 5;

		complex_matrix_type A(n, n, complex_type(0.0, 0.0));
		Hessenberg(A);
		std::cout << "A = " << A << std::endl; 

		complex_vector_type eigvals(n, complex_type(0.0, 0.0));
		complex_matrix_type eigvecs_left(n, n, complex_type(0.0, 0.0));
		complex_matrix_type eigvecs_right(n, n, complex_type(0.0, 0.0));

		boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, (complex_matrix_type *)0L, &eigvecs_right, boost::numeric::bindings::lapack::optimal_workspace());
		//boost::numeric::bindings::lapack::geev(A, eigvals, &eigvecs_left, (complex_matrix_type *)0L, boost::numeric::bindings::lapack::optimal_workspace());

		std::cout << "eigvals = " << eigvals << std::endl;
		std::cout << "eigvecs_left = " << eigvecs_left << std::endl;
		std::cout << "eigvecs_right = " << eigvecs_right << std::endl;
	}
}

void numeric_bindings_lapack_svd()
{
	typedef double real_type; 
	typedef boost::numeric::ublas::matrix<real_type, boost::numeric::ublas::column_major> matrix_type;
	typedef boost::numeric::ublas::vector<real_type> vector_type;

	std::cout << std::endl;
	{
		const size_t m = 3, n = 2;
		const size_t minmn = m < n ? m : n;

		matrix_type A(m, n);
		A(0,0) = 1.; A(0,1) = 1.;
		A(1,0) = 0.; A(1,1) = 1.;
		A(2,0) = 1.; A(2,1) = 0.;
		std::cout << "A: " << A << std::endl;

		vector_type S(minmn);
		matrix_type U(m, m);
		matrix_type Vt(n, n);

		size_t lw;

#if !defined(BOOST_NUMERIC_BINDINGS_LAPACK_2)
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'N', 'N', A);
		std::cout << "opt NN lw: " << lw << std::endl;
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'A', 'A', A);
		std::cout << "opt AA lw: " << lw << std::endl;
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'S', 'S', A);
		std::cout << "opt SS lw: " << lw << std::endl;
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'O', 'N', A);
		std::cout << "opt ON lw: " << lw << std::endl;
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'N', 'O', A);
		std::cout << "opt NO lw: " << lw << std::endl;
#endif
		lw = boost::numeric::bindings::lapack::gesvd_work('M', 'A', 'A', A);
		std::cout << "min lw: " << lw << std::endl << std::endl;

#if !defined(BOOST_NUMERIC_BINDINGS_LAPACK_2)
		lw = boost::numeric::bindings::lapack::gesvd_work('O', 'A', 'A', A);
#endif

		std::vector<real_type> w(lw);

		boost::numeric::bindings::lapack::gesvd('A', 'A', A, S, U, Vt, w);

		std::cout << "S = " << S << std::endl;
		std::cout << "U = " << U << std::endl;
		std::cout << "V^t = " << Vt << std::endl;
	}
}

void numeric_bindings_atlas_lu()
{
	const std::size_t n = 5;
	boost::numeric::ublas::matrix<double> A(n, n);
	for (unsigned i = 0; i < n; ++i)
	{
		for (unsigned j = 0; j < n; ++j)
			A(i, j) = n * i + j;
	}
	std::cout << "A = " << A << std::endl;

	//std::vector<int> ipiv(n);  // pivot vector
	//boost::numeric::bindings::atlas::lu_factor(A, ipiv);  // alias for getrf()
	//boost::numeric::bindings::atlas::lu_invert(A, ipiv);  // alias for getri()

	std::cout << "inverse using LU: inv(A) = " << A << std::endl;
}
