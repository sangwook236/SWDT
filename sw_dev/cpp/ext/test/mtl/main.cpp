#include <mtl/mtl2lapack.h>
#include <iostream>


typedef double value_t;
typedef mtl::matrix<value_t, mtl::rectangle<>, mtl::dense<>, mtl::column_major>::type matrix_t;
typedef mtl::dense1D<mtl::complex<value_t> > vector_t;

void print_matrix(const matrix_t &mat)
{
	for (matrix_t::const_iterator i = mat.begin(); i != mat.end(); ++i)
	{
		std::cout << '\t';
		for (matrix_t::OneD::const_iterator j = (*i).begin(); j != (*i).end(); ++j)
			std::cout << *j << ", ";
		std::cout << std::endl;
	}
/*
	const matrix_t::size_type nrow = mat.nrows();
	for (int i = 0; i < nrow; ++i)
	{
		std::cout << '\t';
		const mtl::rows_type<matrix_t>::type row = rows(mat);
		double ab = rows(mat)[0];
		//for (mtl::rows_type<matrix_t>::type::const_iterator it = row.begin(); it != row.end(); ++it)
		//	std::cout << *it << ", ";
		//mtl::rows_type<matrix_t>::type::size_type aa = mtl::max_index(row);
		//for (int j = 0; j < mtl::max_index(row); ++j)
		//	std::cout << row[j] << ", ";
		std::cout << std::endl;
	}
*/
}

int wmain(int argc, wchar_t* argv[])
{
	const int dim = 2;
	matrix_t mat(dim, dim);
	mat(0, 0) = 1;
	mat(0, 1) = 4;
	mat(1, 0) = 9;
	mat(1, 1) = 1;

	std::cout << "mat:" << std::endl;
	print_matrix(mat);

	matrix_t eigvec_l(dim, dim), eigvec_r(dim, dim);
	vector_t eigval(dim);
	const int ret = mtl2lapack::geev(mtl2lapack::GEEV_CALC_BOTH, mat, eigval, eigvec_l, eigvec_r);

	std::cout << "right eigenvector:" << std::endl;
	print_matrix(eigvec_r);
	std::cout << "left eigenvector:" << std::endl;
	print_matrix(eigvec_l);

	return 0;
}
