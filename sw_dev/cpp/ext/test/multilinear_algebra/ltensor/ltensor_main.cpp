//#include "stdafx.h"
#include "../ltensor_lib/LTensor.h"
#include <iostream>


namespace {
namespace local {

void basic_operation()
{
	Index<'i'> i;
	Index<'j'> j;
	Marray<double, 1> q(2, 1);
	Marray<double, 1> p(2, 2);
	Marray<double, 2> A(2, 2, 3);

	q(i) = A(i, j) * p(j);
	std::cout << q;
	
	p(i) = q(i) + p(i);
	std::cout << p;
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
