#include "stdafx.h"
#include <vector>
#include <string>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

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

int main()
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

	std::cout << "press any key to terminate" << std::flush;
	std::cin.get();

    return 0;
}
