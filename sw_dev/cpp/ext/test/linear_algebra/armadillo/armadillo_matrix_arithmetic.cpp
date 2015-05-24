#include <armadillo>
#include <iostream>


namespace {
namespace local {

void matrix_arithmetic_1()
{
	arma::mat A1(5, 5);
	A1 = 123.0;

	arma::mat A2(5, 5);
	A2.fill(123.0);

	arma::mat A3 = 123.0 * arma::ones<arma::mat>(5, 5);
}

void matrix_arithmetic_2()
{
	arma::mat A = arma::randu<arma::mat>(5, 5);
	double x = A(1, 2);

	arma::mat B = A + A;
	arma::mat C = A * B;
	arma::mat D = A % B;

	arma::cx_mat X(A, B);

	B.zeros();
	B.set_size(10, 10);
	B.zeros(5, 6);

	// fixed size matrices
	arma::mat::fixed<5, 6> F;
	F.ones();

	arma::mat44 G;
	G.randn();

	std::cout << arma::mat22().randu() << std::endl;

	// constructing matrices from auxiliary (external) memory
	double aux_mem[24];
	arma::mat H(aux_mem, 4, 6, false);
}

}  // namespace local
}  // unnamed namespace

namespace my_armadillo {

void matrix_arithmetic()
{
	local::matrix_arithmetic_1();
	local::matrix_arithmetic_2();
}

}  // namespace my_armadillo
