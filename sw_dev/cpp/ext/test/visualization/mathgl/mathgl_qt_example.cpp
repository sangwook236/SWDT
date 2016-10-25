//#include "stdafx.h"
#include <mgl2/qt.h>
#include <mgl2/mgl.h>


namespace {
namespace local {

}  // local
}  // unnamed namespace

namespace my_mathgl {

int test_wnd(mglGraph *gr);
int sample(mglGraph *gr);
int sample_m(mglGraph *gr);
int sample_1(mglGraph *gr);
int sample_2(mglGraph *gr);
int sample_3(mglGraph *gr);
int sample_d(mglGraph *gr);

// REF [file] >> ${MATHGL_HOME}/examples/glut_example.cpp.
void qt_example()
{
	// NOTICE [caution] >> Run-time error in DEBUG mode.

	mglQT gr(sample_1, "1D plots");
	//mglQT gr(sample_2, "2D plots");
	//mglQT gr(sample_3, "3D plots");
	//mglQT gr(sample_d, "Dual plots");
	//mglQT gr(test_wnd, "Testing");
	//mglQT gr(sample, "Example of molecules");
}

}  // namespace my_mathgl
