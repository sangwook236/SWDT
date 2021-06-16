#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gslwrap {

void Vector();
void VectorFloat();
void VectorDiagonalView();
void VectorView();
void VectorView2();
void VectorView3();

void Histogram();
void GSLFunctionCall();
void RandomNumberGenerator();
void LUInvertAndDecomp();
void OneDimMinimiserTest();
void MultDimMinimiserTest();

}  // namespace my_gslwrap

int gslwrap_main(int argc, char *argv[])
{
	//my_gslwrap::Vector();
	//my_gslwrap::VectorFloat();
	my_gslwrap::VectorDiagonalView();
	//my_gslwrap::VectorView();
	//my_gslwrap::VectorView2();
	//my_gslwrap::VectorView3();

	//my_gslwrap::Histogram();
	//my_gslwrap::GSLFunctionCall();
	//my_gslwrap::RandomNumberGenerator();
	//my_gslwrap::LUInvertAndDecomp();
	//my_gslwrap::OneDimMinimiserTest();
	//my_gslwrap::MultDimMinimiserTest();

    return 0;
}
