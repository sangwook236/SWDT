#include "staticlib/Arithmetic.h"
#include <iostream>


void static_lib_linking()
{
	//
	std::cout << "2 = " << TWO << std::endl;
	std::cout << "3 * 2 = " << DOUBLE(3) << std::endl;

	//
	std::cout << "2 + 3 = " << Arithmetic::add(2, 3) << std::endl;
	std::cout << "5 - 3 = " << Arithmetic::subtract(5, 3) << std::endl;

	std::cout << "1 = " << Arithmetic::ONE << std::endl;

	//
	Arithmetic ari(3);
	std::cout << "3 + 1 = " << ari.add(1) << std::endl;
	std::cout << "3 - 1 = " << ari.subtract(1) << std::endl;

	ari.multiply(2);
	std::cout << "3 * 2 = " << ari.getValue() << std::endl;
}
