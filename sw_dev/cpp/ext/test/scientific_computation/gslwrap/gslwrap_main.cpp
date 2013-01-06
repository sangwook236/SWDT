#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace gslwrap {

void Histogram();
void Vector();
void VectorFloat();
void VectorDiagonalView();
void VectorView();
void VectorView2();
void VectorView3();
void GSLFunctionCall();
void RandomNumberGenerator();
void LUInvertAndDecomp();
void Histogram();
void Histogram();
void Histogram();
void OneDimMinimiserTest();
void MultDimMinimiserTest();

}  // namespace gslwrap

int gslwrap_main(int argc, char *argv[])
{
	//gslwrap::Histogram();
	//gslwrap::Vector();
	//gslwrap::VectorFloat();
	gslwrap::VectorView();
	//gslwrap::();
	//gslwrap::VectorView2();
	//gslwrap::VectorView3();
	//gslwrap::GSLFunctionCall();
	//gslwrap::RandomNumberGenerator();
	//gslwrap::LUInvertAndDecomp();
	//gslwrap::OneDimMinimiserTest();
	//gslwrap::MultDimMinimiserTest();

    return 0;
}
