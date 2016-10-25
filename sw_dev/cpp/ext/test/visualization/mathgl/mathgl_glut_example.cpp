//#include "stdafx.h"
#include <mgl2/glut.h>
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
void glut_example()
{
	mglGLUT gr(sample_1, "1D plots");
	//mglGLUT gr(sample_2, "2D plots");
	//mglGLUT gr(sample_3, "3D plots");
	//mglGLUT gr(sample_d, "Dual plots");
	//mglGLUT gr(test_wnd, "Testing");
	//mglGLUT gr(sample, "Example of molecules");
}

}  // namespace my_mathgl
