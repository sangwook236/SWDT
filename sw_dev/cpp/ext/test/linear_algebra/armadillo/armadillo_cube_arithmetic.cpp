#include <armadillo>
#include <iostream>


namespace {
namespace local {

void cube_arithmetic_1()
{
	arma::cube c1(5, 6, 7);
	//c1.slice(0) = arma::randu<arma::mat>(10, 20);  // compile-time error: wrong size

	arma::cube c2(5, 6, 7);
	c2 = 123.0;

	arma::cube c3(5, 6, 7);
	c3.fill(123.0);

	arma::cube c4 = 123.0 * arma::ones<arma::cube>(5, 6, 7);
}

void cube_arithmetic_2()
{
	arma::cube x(1, 2, 3);
	arma::cube y = arma::randu<arma::cube>(4, 5, 6);

	arma::mat A = y.slice(1);  // extract a slice from the cube (each slice is a matrix)

	arma::mat B = arma::randu<arma::mat>(4, 5);
	y.slice(2) = B;  // set a slice in the cube

	std::cout << y << std::endl;

	arma::cube q = y + y;  // cube addition
	arma::cube r = y % y;  // element-wise cube multiplication

	arma::cube::fixed<4, 5, 6> f;
	f.ones();
}

}  // namespace local
}  // unnamed namespace

namespace my_armadillo {

void cube_arithmetic()
{
	local::cube_arithmetic_1();
	local::cube_arithmetic_2();
}

}  // namespace my_armadillo
