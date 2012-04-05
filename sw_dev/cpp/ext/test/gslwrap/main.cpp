//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	extern void Histogram();
	extern void Vector();
	extern void VectorFloat();
	extern void VectorDiagonalView();
	extern void VectorView();
	extern void VectorView2();
	extern void VectorView3();
	extern void GSLFunctionCall();
	extern void RandomNumberGenerator();
	extern void LUInvertAndDecomp();
	extern void Histogram();
	extern void Histogram();
	extern void Histogram();
	extern void OneDimMinimiserTest();
	extern void MultDimMinimiserTest();

	try
	{
		//Histogram();
		//Vector();
		//VectorFloat();
		VectorView();
		//VectorDiagonalView();
		//VectorView2();
		//VectorView3();
		//GSLFunctionCall();
		//RandomNumberGenerator();
		//LUInvertAndDecomp();
		//OneDimMinimiserTest();
		//MultDimMinimiserTest();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}
