#define VIENNACL_HAVE_OPENCL 1  // VIENNACL_WITH_OPENCL. (?)
#define VIENNACL_HAVE_EIGEN 1  // VIENNACL_WITH_EIGEN.
#define VIENNACL_HAVE_UBLAS 1  // VIENNACL_WITH_UBLAS.
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/backend.hpp>
#include <viennacl/linalg/iterative_operations.hpp>
#include <viennacl/linalg/gmres.hpp>
#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/lu.hpp>
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include "viennacl/io/matrix_market.hpp"
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/matrix_proxy.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/vector_proxy.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/scalar.hpp>
#include <Eigen/Dense>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>


namespace {
namespace local {

// REF [file] >> ${VIENNACL_HOME}/examples/tutorial/vector-io.hpp
template<typename VectorType>
void resize_vector(VectorType & vec, unsigned int size)
{
	vec.resize(size);
}

// REF [file] >> ${VIENNACL_HOME}/examples/tutorial/vector-io.hpp
template<typename VectorType>
bool readVectorFromFile(const std::string & filename, VectorType & vec)
{
	typedef typename viennacl::result_of::value_type<VectorType>::type scalar_type;

	std::ifstream file(filename.c_str());

	if (!file) return false;

	unsigned int size;
	file >> size;

	resize_vector(vec, size);

	for (unsigned int i = 0; i < size; ++i)
	{
		scalar_type element;
		file >> element;
		vec[i] = element;
	}

	return true;
}

// REF [file] >> ${VIENNACL_HOME}/examples/tutorial/vector-io.hpp
template<class MatrixType>
bool readMatrixFromFile(const std::string & filename, MatrixType & matrix)
{
	typedef typename viennacl::result_of::value_type<MatrixType>::type scalar_type;

	std::cout << "Reading matrix..." << std::endl;

	std::ifstream file(filename.c_str());

	if (!file) return false;

	std::string id;
	file >> id;
	if (id != "Matrix") return false;

	unsigned int num_rows, num_columns;
	file >> num_rows >> num_columns;
	if (num_rows != num_columns) return false;

	viennacl::traits::resize(matrix, num_rows, num_rows);

	my_inserter<MatrixType> ins(matrix);
	for (unsigned int row = 0; row < num_rows; ++row)
	{
		int num_entries;
		file >> num_entries;
		for (int j = 0; j < num_entries; ++j)
		{
			unsigned int column;
			scalar_type element;
			file >> column >> element;

			ins.apply(row, column, element);
			//insert(matrix, row, column, element);
			//note: the obvious 'matrix(row, column) = element;' does not work with Eigen, hence another level of indirection
		}
		//std::cout << "reading of row finished" << std::endl;
	}

	return true;
}

// REF [site] >> viennacl.sourceforge.net/viennacl-examples-scalar.html
void scalar_example()
{
#ifdef VIENNACL_HAVE_OPENCL
	if (viennacl::ocl::current_device().double_support())  // Link error.
	{
	}
#endif

	typedef float scalar_type;
	//typedef double scalar_type;  // Use this if your GPU supports double precision.

	// Define a few CPU scalars.
	scalar_type s1 = scalar_type(3.1415926);
	scalar_type s2 = scalar_type(2.71763);
	scalar_type s3 = scalar_type(42.0);

	// ViennaCL scalars are defined in the same way.
	viennacl::scalar<scalar_type> vcl_s1;
	viennacl::scalar<scalar_type> vcl_s2 = 1.0;
	viennacl::scalar<scalar_type> vcl_s3 = 1.0;

	// CPU scalars can be transparently assigned to GPU scalars and vice versa.
	vcl_s1 = s1;
	s2 = vcl_s2;
	vcl_s3 = s3;

	// Operations between GPU scalars work just as for CPU scalars (but are much slower!).
	s1 += s2;
	vcl_s1 += vcl_s2;

	s1 = s2 + s3;
	vcl_s1 = vcl_s2 + vcl_s3;

	s1 = s2 + s3 * s2 - s3 / s1;
	vcl_s1 = vcl_s2 + vcl_s3 * vcl_s2 - vcl_s3 / vcl_s1;

	// Operations can also be mixed.
	vcl_s1 = s1 * vcl_s2 + s3 - vcl_s3;

	// Output stream is overloaded as well.
	std::cout << "CPU scalar s2: " << s2 << std::endl;
	std::cout << "GPU scalar vcl_s2: " << vcl_s2 << std::endl;
}

// REF [site] >> http://viennacl.sourceforge.net/viennacl-examples-vector.html
void vector_example()
{
	typedef float scalar_type;
	//typedef double scalar_type;  // Use this if your GPU supports double precision.

	// Define a few CPU vectors using the STL.
	std::vector<scalar_type> std_vec1(10);
	std::vector<scalar_type> std_vec2(10);
	std::vector<scalar_type> std_vec3(10);

	// Define a few GPU vectors using ViennaCL.
	viennacl::vector<scalar_type> vcl_vec1(10);
	viennacl::vector<scalar_type> vcl_vec2(10);
	viennacl::vector<scalar_type> vcl_vec3(10);

	// Fill the CPU vectors with random values.
	// REF [file] >> ${VIENNACL_HOME}/examples/tutorial/Random.hp
	for (unsigned int i = 0; i < 10; ++i)
	{
		std_vec1[i] = static_cast<scalar_type>(std::rand()) / static_cast<scalar_type>(RAND_MAX);;
		vcl_vec2[i] = static_cast<scalar_type>(std::rand()) / static_cast<scalar_type>(RAND_MAX);;  // Also works for GPU vectors, but is slow!
		std_vec3[i] = static_cast<scalar_type>(std::rand()) / static_cast<scalar_type>(RAND_MAX);;
	}

	// Copy the CPU vectors to the GPU vectors and vice versa.
	copy(std_vec1.begin(), std_vec1.end(), vcl_vec1.begin());  // Either the STL way.
	copy(vcl_vec2.begin(), vcl_vec2.end(), std_vec2.begin());
	copy(std_vec3, vcl_vec3);  // Or using the short hand notation
	copy(vcl_vec2, std_vec2);

	// Compute the inner product of two GPU vectors and write the result to either CPU or GPU.
	viennacl::scalar<scalar_type> vcl_s1 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);
	scalar_type s1 = viennacl::linalg::inner_prod(vcl_vec1, vcl_vec2);

	// Compute norms.
	scalar_type s2 = viennacl::linalg::norm_1(vcl_vec1);
	viennacl::scalar<scalar_type> vcl_s2 = viennacl::linalg::norm_2(vcl_vec2);
	scalar_type s3 = viennacl::linalg::norm_inf(vcl_vec3);
	viennacl::scalar<scalar_type> vcl_s3 = viennacl::linalg::norm_inf(vcl_vec2);

	// Use viennacl::vector via the overloaded operators just as you would write it on paper.
	vcl_vec1 = vcl_s1 * vcl_vec2 / vcl_s3;
	vcl_vec1 = vcl_vec2 / vcl_s1 + vcl_s2 * (vcl_vec1 - vcl_s2 * vcl_vec2);
}

// REF [site] >> http://viennacl.sourceforge.net/viennacl-examples-dense-matrix.html
void dense_matrix_example()
{
	typedef float scalar_type;
	//typedef double scalar_type;  // Use this if your GPU supports double precision.

	// Set up and fill matrix in std_matrix here.
	const size_t dim = 5;
	boost::numeric::ublas::vector<scalar_type> std_rhs(dim);
	for (size_t i = 0; i < std_rhs.size(); ++i)
		std_rhs(i) = scalar_type(i + 1); 
	boost::numeric::ublas::matrix<scalar_type> std_matrix(dim, dim);
	for (size_t i = 0; i < std_matrix.size1(); ++i)
		for (size_t j = 0; j < std_matrix.size2(); ++j)
			std_matrix(i, j) = static_cast<scalar_type>((i + 1) + (j + 1)*(i + 1));

	// Copy data to GPU.
	viennacl::vector<scalar_type> vcl_rhs(std_rhs.size());
	viennacl::copy(std_rhs.begin(), std_rhs.end(), vcl_rhs.begin());
	viennacl::matrix<scalar_type> vcl_matrix(std_matrix.size1(), std_matrix.size2());
	viennacl::copy(std_matrix, vcl_matrix);

	// Compute matrix-vector products.
	const viennacl::vector<scalar_type> &vcl_result1 = viennacl::linalg::prod(vcl_matrix, vcl_rhs);  // The ViennaCL way.
	std::cout << "vcl_result1 = " << vcl_result1 << std::endl;

	// Compute transposed matrix-vector products.
	viennacl::vector<scalar_type> vcl_rhs_trans = vcl_rhs;
	const viennacl::vector<scalar_type> &vcl_result_trans = viennacl::linalg::prod(viennacl::trans(vcl_matrix), vcl_rhs_trans);
	std::cout << "vcl_result_trans = " << vcl_result_trans << std::endl;

	// Solve an upper triangular system on the GPU.
	const viennacl::vector<scalar_type> &vcl_result2 = viennacl::linalg::solve(vcl_matrix, vcl_rhs, viennacl::linalg::upper_tag());
	std::cout << "vcl_result2 = " << vcl_result2 << std::endl;

	// Inplace solution of a lower triangular system.
	viennacl::linalg::inplace_solve(vcl_matrix, vcl_rhs, viennacl::linalg::lower_tag());

	// LU factorization and substitution using ViennaCL on GPU.
	viennacl::matrix<scalar_type> vcl_square_matrix(dim);
	viennacl::linalg::lu_factorize(vcl_square_matrix);
	viennacl::vector<scalar_type> vcl_lu_rhs(vcl_rhs);
	viennacl::linalg::lu_substitute(vcl_square_matrix, vcl_lu_rhs);
	std::cout << "vcl_lu_rhs = " << vcl_lu_rhs << std::endl;
}

// REF [site] >> http://viennacl.sourceforge.net/viennacl-examples-sparse-matrix.html
void sparse_matrix_example()
{
	typedef float scalar_type;
	//typedef double scalar_type;  // Use this if your GPU supports double precision.

	// Set up right hand side vector vcl_rhs as in the other examples.
	const size_t dim = 5;
	boost::numeric::ublas::vector<scalar_type> rhs = boost::numeric::ublas::scalar_vector<scalar_type>(dim, scalar_type(dim));
	rhs(0) = -1; rhs(1) = 0; rhs(2) = 3; rhs(3) = -3; rhs(4) = 8;
	boost::numeric::ublas::compressed_matrix<scalar_type> cpu_sparse_matrix(dim, dim);
	cpu_sparse_matrix(0, 0) = 2.0f; cpu_sparse_matrix(0, 1) = -1.0f;
	cpu_sparse_matrix(1, 0) = -1.0f; cpu_sparse_matrix(1, 1) = 2.0f; cpu_sparse_matrix(1, 2) = -1.0f;
	cpu_sparse_matrix(2, 1) = -1.0f; cpu_sparse_matrix(2, 2) = 2.0f; cpu_sparse_matrix(2, 3) = -1.0f;
	cpu_sparse_matrix(3, 2) = -1.0f; cpu_sparse_matrix(3, 3) = 2.0f; cpu_sparse_matrix(3, 4) = -1.0f;
	cpu_sparse_matrix(4, 3) = -1.0f; cpu_sparse_matrix(4, 4) = 2.0f;

	// Copy data from a sparse matrix on the CPU to the GPU.
	viennacl::compressed_matrix<scalar_type> vcl_compressed_matrix;
	viennacl::coordinate_matrix<scalar_type> vcl_coordinate_matrix;
	viennacl::copy(cpu_sparse_matrix, vcl_compressed_matrix);
	viennacl::copy(cpu_sparse_matrix, vcl_coordinate_matrix);
	viennacl::vector<scalar_type> vcl_rhs;
	viennacl::copy(rhs, vcl_rhs);

	// ViennaCL only allows to compute matrix-vector products for sparse matrix types.
	const viennacl::vector<scalar_type> &vcl_result1 = viennacl::linalg::prod(vcl_compressed_matrix, vcl_rhs);
	std::cout << "vcl_result1 = " << vcl_result1 << std::endl;
	const viennacl::vector<scalar_type> &vcl_result2 = viennacl::linalg::prod(vcl_coordinate_matrix, vcl_rhs);
	std::cout << "vcl_result2 = " << vcl_result2 << std::endl;
}

// REF [site] >> ${VIENNACL_HOME}/examples/tutorial/iterative-ublas.cpp
// REF [site] >> http://viennacl.sourceforge.net/viennacl-examples-iterative.html
void iterative_solvers_with_ublas_example()
{
	typedef float scalar_type;
	//typedef double scalar_type;  // Use this if your GPU supports double precision.

	// Set up some ublas objects.
	boost::numeric::ublas::vector<scalar_type> rhs;
	boost::numeric::ublas::vector<scalar_type> ref_result;
	boost::numeric::ublas::vector<scalar_type> result;
	boost::numeric::ublas::compressed_matrix<scalar_type> ublas_matrix;

	// Read system from matrix-market file.
	if (!viennacl::io::read_matrix_market_file(ublas_matrix, "./data/linear_algebra/mat65k.mtx"))
	{
		std::cerr << "Error reading Matrix file" << std::endl;
		return;
	}

	// Read associated vectors from files
	if (!readVectorFromFile("./data/linear_algebra/rhs65025.txt", rhs))
	{
		std::cerr << "Error reading RHS file" << std::endl;
		return;
	}
	//std::cout << "Done reading rhs" << std::endl;

	if (!readVectorFromFile("./data/linear_algebra/result65025.txt", ref_result))
	{
		std::cerr << "Error reading Result file" << std::endl;
		return;
	}
	//std::cout << "Done reading result" << std::endl;

	// Set up ILUT preconditioners for ViennaCL and ublas objects. Other preconditioners can also be used, see \ref manual-algorithms-preconditioners "Preconditioners".
	viennacl::linalg::ilut_precond<boost::numeric::ublas::compressed_matrix<scalar_type>> ublas_ilut(ublas_matrix, viennacl::linalg::ilut_tag());
	viennacl::linalg::ilu0_precond<boost::numeric::ublas::compressed_matrix<scalar_type>> ublas_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());
	viennacl::linalg::block_ilu_precond<boost::numeric::ublas::compressed_matrix<scalar_type>, viennacl::linalg::ilu0_tag> ublas_block_ilu0(ublas_matrix, viennacl::linalg::ilu0_tag());

	// First we run an conjugate gradient solver with different preconditioners.
	std::cout << "----- CG Test -----" << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag());  // Without preconditioner.
	std::cout << "Residual norm: " << boost::numeric::ublas::norm_2(boost::numeric::ublas::prod(ublas_matrix, result) - rhs) << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilut);  // With preconditioner.
	std::cout << "Residual norm: " << boost::numeric::ublas::norm_2(boost::numeric::ublas::prod(ublas_matrix, result) - rhs) << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_ilu0);  // With preconditioner.
	std::cout << "Residual norm: " << boost::numeric::ublas::norm_2(boost::numeric::ublas::prod(ublas_matrix, result) - rhs) << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::cg_tag(1e-6, 20), ublas_block_ilu0);  // With preconditioner.
	std::cout << "Residual norm: " << boost::numeric::ublas::norm_2(boost::numeric::ublas::prod(ublas_matrix, result) - rhs) << std::endl;

	// Run the stabilized BiConjugate gradient solver without and with preconditioners (ILUT, ILU0).
	std::cout << "----- BiCGStab Test -----" << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag());  // Without preconditioner.
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilut);  // With preconditioner.
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::bicgstab_tag(1e-6, 20), ublas_ilu0);  // With preconditioner.

	// Run the generalized minimum residual method witout and with preconditioners (ILUT, ILU0).
	std::cout << "----- GMRES Test -----" << std::endl;
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag());  // Without preconditioner.
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilut);  // With preconditioner.
	result = viennacl::linalg::solve(ublas_matrix, rhs, viennacl::linalg::gmres_tag(1e-6, 20), ublas_ilu0);  // With preconditioner.
}

// REF [site] >> ${VIENNACL_HOME}/examples/tutorial/iterative-eigen.cpp
// REF [site] >> http://viennacl.sourceforge.net/viennacl-examples-eigen.html
void iterative_solvers_with_eigen_example()
{
	typedef float scalar_type;

	Eigen::SparseMatrix<scalar_type, Eigen::RowMajor> eigen_matrix(65025, 65025);
	Eigen::VectorXf eigen_rhs;
	Eigen::VectorXf eigen_result;
	Eigen::VectorXf ref_result;
	Eigen::VectorXf residual;

	// Read system from file.
	std::cout << "Reading matrix (this might take some time)..." << std::endl;
	eigen_matrix.reserve(65025 * 7);
	if (!viennacl::io::read_matrix_market_file(eigen_matrix, "./data/linear_algebra/mat65k.mtx"))
	{
		std::cerr << "Error reading Matrix file. Make sure you run from the build/-folder." << std::endl;
		return;
	}
	//eigen_matrix.endFill();
	std::cout << "Done: reading matrix" << std::endl;

	if (!readVectorFromFile("./data/linear_algebra/rhs65025.txt", eigen_rhs))
	{
		std::cerr << "Error reading RHS file" << std::endl;
		return;
}

	if (!readVectorFromFile("./data/linear_algebra/result65025.txt", ref_result))
	{
		std::cerr << "Error reading Result file" << std::endl;
		return;
	}

	// Conjugate Gradient (CG) solver.
	std::cout << "----- Running CG -----" << std::endl;
	eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::cg_tag());

	residual = eigen_matrix * eigen_result - eigen_rhs;
	std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

	// Stabilized Bi-Conjugate Gradient (BiCGStab) solver.
	std::cout << "----- Running BiCGStab -----" << std::endl;
	eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::bicgstab_tag());

	residual = eigen_matrix * eigen_result - eigen_rhs;
	std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;

	// Generalized Minimum Residual (GMRES) solver.
	std::cout << "----- Running GMRES -----" << std::endl;
	eigen_result = viennacl::linalg::solve(eigen_matrix, eigen_rhs, viennacl::linalg::gmres_tag());

	residual = eigen_matrix * eigen_result - eigen_rhs;
	std::cout << "Relative residual: " << viennacl::linalg::norm_2(residual) / viennacl::linalg::norm_2(eigen_rhs) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_viennacl {

}  // namespace my_viennacl

int viennacl_main(int argc, char *argv[])
{
	std::srand((unsigned int)std::time(NULL));

	local::scalar_example();
	local::vector_example();
	local::dense_matrix_example();
	local::sparse_matrix_example();

	local::iterative_solvers_with_ublas_example();
	local::iterative_solvers_with_eigen_example();

	return 0;
}
