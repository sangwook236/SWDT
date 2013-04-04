//#include "stdafx.h"
#include "../ltensor_lib/LTensor.h"
#include <iostream>


namespace {
namespace local {

void basic_operation()
{
	Marray<double, 1> p(2);
	Marray<double, 2> A(2, 2);

	p(0) = 1;
	p(1) = 2;
	A(0, 0) = 1;  A(0, 1) = 2;
	A(1, 0) = 3;  A(1, 1) = 4;

	//
	Index<'i'> i;
	Index<'j'> j;

	Marray<double, 1> q(2);
	q(i) = A(i, j) * p(j);
	std::cout << q;
	
	p(i) = q(i) + p(i);
	std::cout << p;

	//
	p(0) = 1;
	p(1) = 2;
	q(0) = 3;
	q(1) = 4;

	Marray<double, 2> B(2, 2);
	B(i, j) = p(i) * q(j);  // [ 1 2 ]^T * [ 3 4 ]
	std::cout << B;
}

}  // namespace local
}  // unnamed namespace

namespace my_ltensor {

}  // namespace my_ltensor

int ltensor_main(int argc, char* argv[])
{
	local::basic_operation();

	return 0;
}
