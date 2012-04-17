#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>


namespace {
namespace local {

/**
* Invert a matrix via gauss-jordan algorithm (PARTIAL PIVOT)
*
* @param m The matrix to invert. Must be square.
* @param singular If the matrix was found to be singular, then this is set to true, else set to false.
* @return If singular is false, then the inverted matrix is returned. Otherwise it contains random values.
*/
template<class T>
boost::numeric::ublas::matrix<T> ublas_inv_by_gj(const boost::numeric::ublas::matrix<T> &m, bool &singular)
{
	const T eps = (T)1.0e-15;

	const int size = m.size1();
	// Cannot invert if non-square matrix or 0x0 matrix.
	// Report it as singular in these cases, and return a 0x0 matrix.
	if (size != m.size2() || size == 0)
	{
		singular = true;
		boost::numeric::ublas::matrix<T> A(0,0);
		return A;
	}

	// Handle 1x1 matrix edge case as general purpose 
	// inverter below requires 2x2 to function properly.
	if (size == 1)
	{
		boost::numeric::ublas::matrix<T> A(1, 1);
		if (std::abs(m(0,0)) <= eps)
		{
			singular = true;
			return A;
		}

		singular = false;
		A(0,0) = 1 / m(0,0);
		return A;
	}

	// Create an augmented matrix A to invert. Assign the
	// matrix to be inverted to the left hand side and an
	// identity matrix to the right hand side.
	boost::numeric::ublas::matrix<T> A(size, 2*size);
	boost::numeric::ublas::matrix_range<boost::numeric::ublas::matrix<T> > Aleft(A, boost::numeric::ublas::range(0, size), boost::numeric::ublas::range(0, size));
	Aleft = m;
	boost::numeric::ublas::matrix_range<boost::numeric::ublas::matrix<T> > Aright(A, boost::numeric::ublas::range(0, size), boost::numeric::ublas::range(size, 2*size));
	Aright = boost::numeric::ublas::identity_matrix<T>(size);

	// Swap rows to eliminate zero diagonal elements.
	for (int k = 0; k < size; ++k)
	{
		if (A(k,k) == 0) // XXX: test for "small" instead
		{
			// Find a row(l) to swap with row(k)
			int l = -1;
			for (int i = k+1; i < size; i++) 
			{
				if ( A(i,k) != 0 )
				{
					l = i; 
					break;
				}
			}

			// Swap the rows if found
			if (l < 0) 
			{
				//std::cerr << "Error:" << __FUNCTION__ << ": Input matrix is singular, because cannot find a row to swap while eliminating zero-diagonal.";
				singular = true;
				return Aleft;
			}
			else 
			{
				boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<T> > rowk(A, k);
				boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<T> > rowl(A, l);
				rowk.swap(rowl);

//#if defined(DEBUG) || !defined(NDEBUG)
//					std::cerr << __FUNCTION__ << ":" << "Swapped row " << k << " with row " << l << ":" << A << std::endl;
//#endif
			}
		}
	}

	// Doing partial pivot
	for (int k = 0; k < size; ++k)
	{
		// normalize the current row
		for (int j = k+1; j < 2*size; ++j)
			A(k,j) /= A(k,k);
		A(k,k) = 1;

		// normalize other rows
		for (int i = 0; i < size; ++i)
		{
			if (i != k)  // other rows  // FIX: PROBLEM HERE
			{
				if (A(i,k) != 0)
				{
					for (int j = k+1; j < 2*size; ++j)
						A(i,j) -= A(k,j) * A(i,k);
					A(i,k) = 0;
				}
			}
		}

//#if defined(DEBUG) || !defined(NDEBUG)
//		std::cerr << __FUNCTION__ << ": GJ row " << k << " : " << A << std::endl;
//#endif
	}

	singular = false;
	return Aright;
}

// Matrix inversion routine. Uses lu_factorize and lu_substitute in uBLAS to invert a matrix
template<class T>
bool ublas_inv_by_lu(const boost::numeric::ublas::matrix<T> &input, boost::numeric::ublas::matrix<T> &inverse)
{
	typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix_t;

	// create a working copy of the input
	boost::numeric::ublas::matrix<T> A(input);
	// create a permutation matrix for the LU-factorization
	pmatrix_t pm(A.size1());

	// perform LU-factorization
	const int res = boost::numeric::ublas::lu_factorize(A,pm);
	if (0 != res) return false;

	// create identity matrix of "inverse"
	inverse.assign(boost::numeric::ublas::identity_matrix<T>(A.size1()));

	// backsubstitute to get the inverse
	boost::numeric::ublas::lu_substitute(A, pm, inverse);

	return true;
}


template<class T>
void TransposeMultiply(const boost::numeric::ublas::vector<T> &v, boost::numeric::ublas::matrix<T> &m)
{
	//m.resize(size, size);
	//m.clear();
	for (unsigned int row = 0; row < v.size(); ++row)
	{
		for(unsigned int col = 0; col < v.size(); ++col)
			m(row,col) = v(col) * v(row);
	}
}

template<class T>
bool HouseholderCornerSubstraction(boost::numeric::ublas::matrix<T> &LeftLarge, const boost::numeric::ublas::matrix<T> &RightSmall)
{
	if (!(LeftLarge.size1() >= RightSmall.size1() && LeftLarge.size2() >= RightSmall.size2()))
	{
		//std::cerr << "invalid matrix dimensions" << std::endl;
		return false;
	}

	size_t row_offset = LeftLarge.size2() - RightSmall.size2();
	size_t col_offset = LeftLarge.size1() - RightSmall.size1();

	for (unsigned int row = 0; row < RightSmall.size2(); ++row)
		for (unsigned int col = 0; col < RightSmall.size1(); ++col)
			LeftLarge(col_offset+col,row_offset+row) -= RightSmall(col,row);

	return true;
}

template<class T>
bool ublas_qr(const boost::numeric::ublas::matrix<T> &M, boost::numeric::ublas::matrix<T> &Q, boost::numeric::ublas::matrix<T> &R)
{
	if (M.size1() != M.size2())
	{
		//std::cerr << "invalid matrix dimensions" << std::endl;
		return false;
	}

	const size_t size = M.size1();

	// init Matrices
	boost::numeric::ublas::matrix<T> H, HTemp;
	HTemp = boost::numeric::ublas::identity_matrix<T>(size);
	Q = boost::numeric::ublas::identity_matrix<T>(size);
	R = M;

	// find Householder reflection matrices
	for (unsigned int col = 0; col < size-1; ++col)
	{
		// create X vector
		boost::numeric::ublas::vector<T> RRowView = boost::numeric::ublas::column(R, col);      
		boost::numeric::ublas::vector_range<boost::numeric::ublas::vector<T> > X2(RRowView, boost::numeric::ublas::range(col, size));
		boost::numeric::ublas::vector<T> X = X2;

		// X -> U~
		X(0) += X(0) >= 0 ? norm_2(X) : -norm_2(X);

		HTemp.resize(X.size(), X.size(), true);
		HTemp.clear();
		TransposeMultiply(X, HTemp);

		// HTemp = the 2UUt part of H 
		HTemp *= 2 / boost::numeric::ublas::inner_prod(X,X);

		// H = I - 2UUt
		H = boost::numeric::ublas::identity_matrix<T>(size);
		if (!HouseholderCornerSubstraction(H, HTemp))
			return false;

		// add H to Q and R
		Q = boost::numeric::ublas::prod(Q, H);
		R = boost::numeric::ublas::prod(H, R);
	}

	return true;
}

void ublas_basic()
{
	typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major, boost::numeric::ublas::unbounded_array<double> > matrix_type;

	{
		matrix_type m(3, 3);
		for (unsigned i = 0, k = 0; i < m.size1(); ++i)
			for (unsigned j = 0; j < m.size2(); ++j, ++k)
				m(i, j) = k;

		for (unsigned i = 0, k = 0; i < m.size1(); ++i)
		{
			for (unsigned j = 0; j < m.size2(); ++j, ++k)
				std::cout << (j ? " " : "") << m(i, j);
			std::cout << std::endl;
		}

		std::cout << m << std::endl;

		// element
		matrix_type::array_type arr = m.data();
		//for (unsigned i = 0; i < m.size1() * m.size2(); ++i)
		//	std::cout << arr[i] << ' ';
		for (matrix_type::array_type::iterator it = arr.begin(); it != arr.end(); ++it)
			std::cout << *it << ' ';
		std::cout << std::endl;

		// vector range
		boost::numeric::ublas::matrix_vector_range<matrix_type> mvr(m, boost::numeric::ublas::range(0, 3), boost::numeric::ublas::range(0, 3));
		std::cout << mvr << std::endl;

		// vector slice
		boost::numeric::ublas::matrix_vector_slice<matrix_type> mvs(m, boost::numeric::ublas::slice(0, 2, 2), boost::numeric::ublas::slice(0, 2, 2));
		std::cout << mvs << std::endl;

		// matrix range
		boost::numeric::ublas::matrix_range<matrix_type> mr(m, boost::numeric::ublas::range(0, 3), boost::numeric::ublas::range(0, 2));
		std::cout << mr << std::endl;

		// matrix slice
		boost::numeric::ublas::matrix_slice<matrix_type> ms(m, boost::numeric::ublas::slice(0, 2, 2), boost::numeric::ublas::slice(0, 1, 3));
	    std::cout << ms << std::endl;

		// symmetric matrix
		boost::numeric::ublas::symmetric_adaptor<matrix_type, boost::numeric::ublas::lower> sal(m);
		//for (unsigned i = 0, k = 0; i < sal.size1(); ++i)
		//	for (unsigned j = 0; j <= i; ++j, ++k)
		//		sal(i, j) = k;
		std::cout << sal << std::endl;

		boost::numeric::ublas::symmetric_adaptor<matrix_type, boost::numeric::ublas::upper> sau(m);
		//for (unsigned i = 0, k = 0; i < sau.size1(); ++i)
		//	for (unsigned j = i; j < sau.size2(); ++j, ++k)
		//		sau(i, j) = k;
		std::cout << sau << std::endl;
	}
}

void ublas_vector_operation()
{
	typedef boost::numeric::ublas::vector<double, boost::numeric::ublas::unbounded_array<double> > vector_type;

	{
		//
		{
			boost::numeric::ublas::vector<double> v1(3), v2(3);
			for (unsigned i = 0; i < std::min(v1.size(), v2.size()); ++i)
				v1(i) = v2(i) = i;

			std::cout << v1 + v2 << std::endl;
			std::cout << v1 - v2 << std::endl;

			std::cout << 2.0 * v1 << std::endl;
			std::cout << v1 * 2.0 << std::endl;
		}

		//
		{
			boost::numeric::ublas::vector<std::complex<double> > v(3);
			for (unsigned i = 0; i < v.size(); ++i)
				v(i) = std::complex<double>(i, i);

			std::cout << -v << std::endl;
			std::cout << boost::numeric::ublas::conj(v) << std::endl;
			std::cout << boost::numeric::ublas::real(v) << std::endl;
			std::cout << boost::numeric::ublas::imag(v) << std::endl;
			std::cout << boost::numeric::ublas::trans(v) << std::endl;
			std::cout << boost::numeric::ublas::herm(v) << std::endl;
		}

		//
		{
			boost::numeric::ublas::vector<double> v(3);
			for (unsigned i = 0; i < v.size(); ++i)
				v(i) = i;

			std::cout << boost::numeric::ublas::sum(v) << std::endl;
			std::cout << boost::numeric::ublas::norm_1(v) << std::endl;
			std::cout << boost::numeric::ublas::norm_2(v) << std::endl;
			std::cout << boost::numeric::ublas::norm_inf(v) << std::endl;
			std::cout << boost::numeric::ublas::index_norm_inf(v) << std::endl;
		}

		//
		{
			boost::numeric::ublas::vector<double> v1(3), v2(3);
			for (unsigned i = 0; i < std::min(v1.size(), v2.size()); ++i)
				v1(i) = v2(i) = i;

			std::cout << boost::numeric::ublas::inner_prod(v1, v2) << std::endl;
			std::cout << boost::numeric::ublas::outer_prod(v1, v2) << std::endl;  // caution: it's not vector product. (outer_prod(v1, v2))[i][j] = v1[i] * v2[j]
		}

		//
		{
			vector_type v1(4);
			for (unsigned i = 0; i < v1.size(); ++i)
				v1(i) = i + 1;

			vector_type v2(4);
			for (unsigned i = 0; i < v2.size(); ++i)
				v2(i) = 10 - i;

			std::cout << boost::numeric::ublas::blas_1::asum(v1) << std::endl;
			std::cout << boost::numeric::ublas::blas_1::nrm2(v1) << std::endl;
			std::cout << boost::numeric::ublas::blas_1::amax(v1) << std::endl;

			std::cout << boost::numeric::ublas::blas_1::dot(v1, v2) << std::endl;
			std::cout << boost::numeric::ublas::blas_1::axpy(v1, 2.0, v2) << std::endl;  // v1 = v1 + t * v2

			vector_type v3(4);
			boost::numeric::ublas::blas_1::copy(v3, v1);
			std::cout << v3 << std::endl;
			boost::numeric::ublas::blas_1::swap(v1, v2);
			std::cout << v1 << ", " << v2 << std::endl;
			boost::numeric::ublas::blas_1::scal(v1, 3);
			std::cout << v1 << std::endl;

			//boost::numeric::ublas::blas_1::rot(...);  // plane rotation
		}
	}
}

void ublas_matrix_operation()
{
	typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major, boost::numeric::ublas::unbounded_array<double> > matrix_type;

	{
		{
			matrix_type m1(3, 3);
			for (unsigned i = 0, k = 0; i < m1.size1(); ++i)
				for (unsigned j = 0; j < m1.size2(); ++j, ++k)
					m1(i, j) = k;

			matrix_type m2(3, 3);
			for (unsigned i = 0, k = 0; i < m2.size1(); ++i)
				for (unsigned j = 0; j < m2.size2(); ++j, ++k)
					m2(i, j) = k;

			std::cout << 2.0 * m1 << std::endl;
			std::cout << m1 * 2.0 << std::endl;

			std::cout << m1 + m2 << std::endl;
			std::cout << m1 - m2 << std::endl;
		}

		//
		{
			boost::numeric::ublas::matrix<std::complex<double> > m(3,3);
			for (unsigned i = 0; i < m.size1(); ++i)
				for (unsigned j = 0; j < m.size2(); ++j)
					m(i, j) = std::complex<double>(3 * i + j, 3 * i + j);

			std::cout << -m << std::endl;
			std::cout << boost::numeric::ublas::conj(m) << std::endl;
			std::cout << boost::numeric::ublas::real(m) << std::endl;
			std::cout << boost::numeric::ublas::imag(m) << std::endl;
			std::cout << boost::numeric::ublas::trans(m) << std::endl;
			std::cout << boost::numeric::ublas::herm(m) << std::endl;
		}

		//
		{
			//boost::numeric::ublas_2::tmv(v, m);  // m * v. m: triangular matrix
			//boost::numeric::ublas_2::tsv(v, m);  // m * x = v. m: triangular matrix
			//boost::numeric::ublas_2::gmv(v1, t1, t2, m, v2);  // v1 = t1 * v1 + t2 * (m * v2)
			//boost::numeric::ublas_2::gr(m, t, v1, v2);  // m = m + t * (v1 * v2^T)
			//boost::numeric::ublas_2::sr(m, t, v);  // m = m + t * (v * v^T)
			//boost::numeric::ublas_2::hr(m, t, v);  // m = m + t * (v * v^H)
			//boost::numeric::ublas_2::sr2(m, t, v1, v2);  // m = m + t * (v1 * v2^T + v2 * v1^T)
			//boost::numeric::ublas_2::hr2(m, t, v1, v2);  // m = m + t * (v1 * v2^H) + (v2 * (t * v1)^H)
		}
	}
}

void ublas_matrix_vector_operation()
{
	typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major, boost::numeric::ublas::unbounded_array<double> > matrix_type;
	typedef boost::numeric::ublas::vector<double, boost::numeric::ublas::unbounded_array<double> > vector_type;

	{
		{
			matrix_type m(3, 3);
			vector_type v(3);
			for (unsigned i = 0; i < std::min(m.size1(), v.size()); ++i)
			{
				for (unsigned j = 0; j < m.size2(); ++j)
					m(i, j) = 3 * i + j;
				v(i) = i;
			}

			std::cout << boost::numeric::ublas::prod(m, v) << std::endl;
			std::cout << boost::numeric::ublas::prod(v, m) << std::endl;
		}

		//
		{
			matrix_type m(3, 3);
			vector_type v(3);
			for (unsigned i = 0; i < std::min(m.size1(), v.size()); ++i)
			{
				for (unsigned j = 0; j <= i; ++j)
					m(i, j) = 3 * i + j + 1;
				v(i) = i;
			}

			// m: lower triangular matrix
			std::cout << boost::numeric::ublas::solve(m, v, boost::numeric::ublas::lower_tag()) << std::endl;
			//std::cout << boost::numeric::ublas::solve(v, m, boost::numeric::ublas::lower_tag()) << std::endl;
		}
	}
}

void ublas_matrix_matrix_operation()
{
	typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major, boost::numeric::ublas::unbounded_array<double> > matrix_type;

	{
		{
			matrix_type m1(3, 3), m2(3, 3);
			for (unsigned i = 0; i < std::min(m1.size1(), m2.size1()); ++i)
			{
				for (unsigned j = 0; j < std::min(m1.size2(), m2.size2()); ++j)
					m1(i, j) = m2(i, j) = 3 * i + j;
			}

			std::cout << boost::numeric::ublas::prod(m1, m2) << std::endl;
		}

		//
		{
			boost::numeric::ublas::matrix<double> m1(3, 3), m2(3, 3);
			for (unsigned i = 0; i < std::min(m1.size1(), m2.size1()); ++i)
				for (unsigned j = 0; j <= i; ++j)
					m1(i, j) = m2(i, j) = 3 * i + j + 1;

			// m1: lower triangular matrix
			std::cout << boost::numeric::ublas::solve(m1, m2, boost::numeric::ublas::lower_tag()) << std::endl;
		}

		//
		{
			matrix_type m1(3, 3), m2(3, 3);
			for (unsigned i = 0; i < std::min(m1.size1(), m2.size1()); ++i)
			{
				for (unsigned j = 0; j < std::min(m1.size2(), m2.size2()); ++j)
					m1(i, j) = m2(i, j) = 3 * i + j;
			}

			matrix_type m3(m1.size1(), m2.size2());
			boost::numeric::ublas::axpy_prod(m1, m2, m3, true);
			std::cout << m3 << std::endl;

			matrix_type m4(m1.size1(), m2.size2());
			m4.clear();
			boost::numeric::ublas::blas_3::gmm(m4, 0.0, 1.0, m1, m2);
			std::cout << m4 << std::endl;
		}

		//
		{
			//boost::numeric::ublas_3::tmm(m1, t, m2, m3);  // m1 = t * (m2 * m3). triangular matrix multiplication
			//boost::numeric::ublas_3::tsm(m1, t, m2);  // m2 * x = t * m1. m2: triangular matrix
			//boost::numeric::ublas_3::gmm(m1, t, m2);  // m1 = t1 * m1 + t2 * (m2 * m3)
			//boost::numeric::ublas_3::srk(m1, t1, t2, m2);  // m1 = t1 * m1 + t2 * (m2 * m2^T);
			//boost::numeric::ublas_3::hrk(m1, t1, t2, m2);  // m1 = t1 * m1 + t2 * (m2 * m2^H);
			//boost::numeric::ublas_3::sr2k(m1, t1, t2, m2, m3);  // m1 = t1 * m1 + t2 * (m2 * m3^T + m3 * m2^T);
			//boost::numeric::ublas_3::hr2k(m1, t1, t2, m2, m3);  // m1 = t1 * m1 + (t2 * (m2 * m3^H)) + (m3 * (t2 * m2)^H);
		}
	}
}

void ublas_lu()
{
	typedef boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major, boost::numeric::ublas::unbounded_array<double> > matrix_type;
	typedef boost::numeric::ublas::vector<double, boost::numeric::ublas::unbounded_array<double> > vector_type;
	typedef boost::numeric::ublas::permutation_matrix<std::size_t> pmatrix_type;

	{
		matrix_type m1(3, 3);
		m1(0, 0) = 4.0;  m1(0, 1) = 2.0;  m1(0, 2) = 3.0;
		m1(1, 0) = 0.0;  m1(1, 1) = -1.0;  m1(1, 2) = 2.0;
		m1(2, 0) = 5.0;  m1(2, 1) = -2.0;  m1(2, 2) = 3.0;
		matrix_type m2(m1);

		vector_type vb(3);
		vb(0) = 6;  vb(1) = 12;  vb(2) = 12;
		vector_type vs1(vb), vs2(vb);

		pmatrix_type pm1(m1.size1()), pm2(m2.size1());

		const matrix_type::size_type s1 = boost::numeric::ublas::lu_factorize(m1, pm1);
		boost::numeric::ublas::lu_substitute(m1, pm1, vs1);
		std::cout << "A * x = b: sol = " << vs1 << std::endl;
	
		//const matrix_type::size_type s2 = boost::numeric::ublas::axpy_lu_factorize(m2, pm2);
		//boost::numeric::ublas::lu_substitute(m2, pm2, vs2);
	}
}

void ublas_inv()
{
	boost::numeric::ublas::matrix<double> A(3,3);
	A(0,0) = 1;  A(0,1) = 1;  A(0,2) = 0;
	A(1,0) = 0;  A(1,1) = 1;  A(1,2) = 0;
	A(2,0) = 1;  A(2,1) = 0;  A(2,2) = 1; 

	std::cout << "A = " << A << std::endl;

	bool is_singular;
	const boost::numeric::ublas::matrix<double> &invA1 = ublas_inv_by_gj(A, is_singular);
	if (is_singular)
		std::cout << "A is singular" << std::endl;
	else
		std::cout << "inverse using G-J: inv(A) = " << invA1 << std::endl;

	boost::numeric::ublas::matrix<double> invA2(A.size1(), A.size2());
	if (ublas_inv_by_lu(A, invA2))
		std::cout << "inverse using LU: inv(A) = " << invA2 << std::endl;
	else
		std::cout << "A is singular" << std::endl;
}

void ublas_qr()
{
	boost::numeric::ublas::matrix<double> A(3,3);
	A(0,0) = 1;  A(0,1) = 1;  A(0,2) = 0;
	A(1,0) = 0;  A(1,1) = 1;  A(1,2) = 0;
	A(2,0) = 1;  A(2,1) = 0;  A(2,2) = 1; 

	std::cout << "A = " << A << std::endl;
	std::cout << "QR decomposition using Householder" << std::endl;
	boost::numeric::ublas::matrix<double> Q(3,3), R(3,3);
	if (ublas_qr(A, Q, R))
	{
		boost::numeric::ublas::matrix<double> Z = boost::numeric::ublas::prod(Q, R) - A;
		const double f = boost::numeric::ublas::norm_1(Z);
		std::cout << "Q = " << Q << std::endl;
		std::cout << "R = " << R << std::endl;
		std::cout << "|Q*R - A| = " << f << std::endl;
	}
	else
		std::cout << "error: cann't compute inverse" << std::endl;
}

}  // namespace local
}  // unnamed namespace

void ublas()
{
	local::ublas_basic();
	local::ublas_vector_operation();
	local::ublas_matrix_operation();
	local::ublas_matrix_vector_operation();
	local::ublas_matrix_matrix_operation();
	local::ublas_lu();

	local::ublas_inv();
	local::ublas_qr();
}
