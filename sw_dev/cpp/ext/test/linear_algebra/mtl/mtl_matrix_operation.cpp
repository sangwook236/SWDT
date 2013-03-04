#include <boost/numeric/mtl/mtl.hpp>
#include <iostream>
#include <complex>


namespace {
namespace local {

template <typename Matrix>
void fill_element(Matrix &m)
{
	// Matrices are not initialized by default
	m = 0.0;

	// Type of m's elements
	typedef typename mtl::Collection<Matrix>::value_type value_type;

	// Create inserter for matrix m
	// Existing values are not overwritten but inserted
	mtl::matrix::inserter<Matrix, mtl::update_plus<value_type> > ins(m, 3);

	// Define element matrix (array)
	double m1[2][2] = {{1.0, -.4}, {-0.5, 2.0}};

	// Corresponding indices of the elements
	std::vector<int> v1(2);
	v1[0] = 1; v1[1] = 3;

	// Insert element matrix
	ins << mtl::element_array(m1, v1);

	// Insert same array with different indices
	v1[0] = 0; v1[1] = 2;
	ins << mtl::element_array(m1, v1);

	// Use element matrix type with dynamic size
	mtl::dense2D<double> m2(2, 3);
	m2[0][0] = 1; m2[0][1] = 0.2; m2[0][2] = 0.1;
	m2[1][0] = 2; m2[1][1] = 1.2; m2[1][2] = 1.1;

	// Vector for column indices
	mtl::dense_vector<int> v2(3);
	// Indices can be out of order
	v2[0] = 4; v2[1] = 1; v2[2] = 3;

	// Use element_matrix and separate vectors for row and column indices
	ins << mtl::matrix::element_matrix(m2, v1, v2);
}

void matrix_element()
{
	const int width = 5, precision = 2;

	// Matrices of different types
	mtl::compressed2D<double> A(5, 5);
	mtl::dense2D<double> B(5, 5);
	mtl::morton_dense<float, mtl::morton_mask> C(5, 5);

	// Fill the matrices generically
	fill_element(A); fill_element(B); fill_element(C);
	std::cout << "A is \n" << mtl::matrix::with_format(A, width, precision)
		  << "\nB is \n" << mtl::matrix::with_format(B, width, precision)
		  << "\nC is \n" << mtl::matrix::with_format(C, width, precision);
}


template <typename Matrix>
void fill_insert(Matrix &m)
{
	// Matrices are not initialized by default
	m = 0.0;

	// Create inserter for matrix m
	mtl::matrix::inserter<Matrix> ins(m);

	// Insert value in m[0][0]
	ins(0, 0) << 2.0;
	ins(1, 2) << 0.5;
	ins(2, 1) << 3.0;

	// Destructor of ins sets final state of m
}

template <typename Matrix>
void modify_insert(Matrix &m)
{
	// Type of m's elements
	typedef typename mtl::Collection<Matrix>::value_type value_type;

	// Create inserter for matrix m
	// Existing values are not overwritten but inserted
	mtl::matrix::inserter<Matrix, mtl::update_plus<value_type> > ins(m, 3);
	//mtl::matrix::inserter<Matrix, mtl::update_store<value_type> > ins(m, 3);
	//mtl::matrix::inserter<Matrix, mtl::update_minus<value_type> > ins(m, 3);
	//mtl::matrix::inserter<Matrix, mtl::update_times<value_type> > ins(m, 3);
	//mtl::matrix::inserter<Matrix, mtl::update_adapter<value_type, MonoidOp> > ins(m, 3);

	// Increment value in m[0][0]
	ins(0, 0) << 1.0;

	// Elements that doesn't exist (in sparse matrices) are inserted
	ins(1, 1) << 2.5;
	ins(2, 1) << 1.0;
	ins(2, 2) << 4.0;

	// Destructor of ins sets final state of m
}

void insert()
{
	// Matrices of different types
	mtl::compressed2D<double> A(3, 3);
	mtl::dense2D<double> B(3, 3);
	mtl::morton_dense<float, mtl::morton_mask> C(3, 3);

	// Fill the matrices generically
	fill_insert(A); fill_insert(B); fill_insert(C);
	std::cout << "A is \n" << A << "\nB is \n" << B << "\nC is \n" << C;

	// Modify the matrices generically
	modify_insert(A); modify_insert(B); modify_insert(C);
	std::cout << "\n\nAfter modification:\nA is \n" << A
		  << "\nB is \n" << B << "\nC is \n" << C;
}

void matrix_operation()
{
	{
		const unsigned n = 3;

		mtl::dense2D<int, mtl::matrix::parameters<mtl::col_major> > B(n, n);
		mtl::morton_dense<double, 0x55555555> C(n, n);
		mtl::morton_dense<double, 0x555555f0> D(n, n);

		B = 0;
		D = 0;

		D(2, 2) = 3;
		B[1][2] = 4;

		C = 2.0;
		D *= 7.0;

		std::cout << "The matrices are: B=\n" << B << "C=\n" << C << "D=\n" << D;
		std::cout << "C * (3i)=\n" << std::complex<double>(0, 3) * C;
	}

	{
		const unsigned n = 100;
		mtl::dense2D<double> A(n, n), B(n, n);
		mtl::morton_dense<double, mtl::doppled_64_row_mask> C(n, n);

		mtl::hessian_setup(A, 3.0);
		mtl::hessian_setup(B, 1.0);
		mtl::hessian_setup(C, 2.0);

		// A = B * B;
		mtl::matrix::mult(B, B, A);

		A = B * B;  // use BLAS
		A = B * C;  // use recursion + tiling from MTL4

		A += B * C;  // Increment A by the product of B and C
		A -= B * C;  // Likewise with decrement
	}

	{
		const unsigned n = 10;

		mtl::compressed2D<double> A(n, n);
		mtl::dense2D<int, mtl::matrix::parameters<mtl::col_major> > B(n, n);
		mtl::morton_dense<double, 0x555555f0> C(n, n), D(n, n);

		mtl::matrix::laplacian_setup(A, 2, 5);
		mtl::matrix::hessian_setup(B, 1);
		mtl::matrix::hessian_setup(C, 2.0);
		mtl::matrix::hessian_setup(D, 3.0);

		D += A - 2 * B + C;

		//std::cout << "The matrices are: A=\n" << A << "B=\n" << B << "C=\n" << C << "D=\n" << D;
		std::cout << "The matrices are: D=\n" << D;
	}
}

void matrix_function()
{
	{
		typedef std::complex<double> complex_type;
		const unsigned row = 2, col = 5, n = row * col;
		mtl::compressed2D<complex_type> A(n, n);

		mtl::matrix::laplacian_setup(A, row, col);

		// Fill imaginary part of the matrix
		A *= complex_type(1, -1);
		std::cout << "A is\n" << mtl::matrix::with_format(A, 7, 1) << std::endl;

		std::cout << "trace(A) is " << mtl::matrix::trace(A) << std::endl;
		std::cout << "conj(A) is\n" << mtl::matrix::with_format(mtl::conj(A), 7, 1) << std::endl;
		std::cout << "trans(A) is\n" << mtl::matrix::with_format(mtl::matrix::trans(A), 7, 1) << std::endl;
	}

	{
		const unsigned n = 10;
		mtl::compressed2D<double> A(n, n);
		mtl::dense2D<float, mtl::matrix::parameters<mtl::col_major> > B(n, n);
		mtl::morton_dense<double, 0x55555555> C(n, n);
		mtl::morton_dense<double, 0x555555f0> D(n, n);

		mtl::matrix::hessian_setup(B, 1.0);
		mtl::matrix::hessian_setup(C, 2.0);
		mtl::matrix::hessian_setup(D, 3.0);

		std::cout << "one_norm(B) is " << mtl::one_norm(B) << std::endl;
		std::cout << "infinity_norm(B) is " << mtl::infinity_norm(B) << std::endl;
		std::cout << "frobenius_norm(B) is " << mtl::matrix::frobenius_norm(B) << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_mtl {

void matrix_operation()
{
	local::matrix_element();
	local::insert();
	local::matrix_operation();
	local::matrix_function();
}

void matrix_vector_operation()
{
	const unsigned xd = 2, yd = 5, n = xd * yd;
	mtl::dense2D<double> A(n, n);
	mtl::compressed2D<double> B(n, n);

	mtl::matrix::hessian_setup(A, 3.0);
	mtl::matrix::laplacian_setup(B, xd, yd);

	typedef std::complex<double> complex_type;
	mtl::dense_vector<complex_type> v(n), w(n);
	for (size_t i= 0; i < mtl::vector::size(v); ++i)
		v[i] = complex_type(i+1, n-i), w[i] = complex_type(i+n);

	v += A * w;
	w = B * v;

	std::cout << "v is " << v << std::endl;
	std::cout << "w is " << w << std::endl;
}

}  // namespace my_mtl
