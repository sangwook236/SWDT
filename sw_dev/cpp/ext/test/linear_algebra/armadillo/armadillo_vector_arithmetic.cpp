#if defined(WIN32)
#include <armadillo/armadillo>
#else
#include <armadillo>
#endif
#include <iostream>


namespace {
namespace local {

void vector_arithmetic_1()
{
	arma::vec q1(5);
	q1 = 123.0;
	arma::vec q2(5);
	q2.fill(123.0);
	arma::vec q3 = 123.0 * arma::ones<arma::vec>(5, 1);

	//
	arma::vec x(10);
	arma::vec y = arma::zeros<arma::vec>(10, 1);

	arma::mat A = arma::randu<arma::mat>(10, 10);
	arma::vec z = A.col(5);  // extract a column vector

	std::cout << z << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_armadillo {

void vector_arithmetic()
{
	local::vector_arithmetic_1();
}

}  // namespace my_armadillo
