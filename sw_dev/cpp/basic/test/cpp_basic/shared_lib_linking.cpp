#include "sharedlib/Trigonometric.h"
#include <iostream>


void shared_lib_linking()
{
	//
	std::cout << "E = " << E << std::endl;
	std::cout << "SQRT(4) = " << SQRT(4) << std::endl;

	//
	std::cout << "PI = " << Trigonometric::PI << std::endl;
	std::cout << "sin(pi) = " << Trigonometric::sin(Trigonometric::PI) << std::endl;

	//
	Trigonometric tri(Trigonometric::PI);
	std::cout << "cos(pi) = " << tri.cos() << std::endl;
	tri.tan();
	std::cout << "tan(pi) = " << tri.getValue() << std::endl;
}
