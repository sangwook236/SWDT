#include <tinysplinecpp.h>
#include <tinyspline.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>


namespace {
namespace local {

// REF [file] >> ${TINYSPLINE_HOME}/examples/cpp/quickstart.cpp
void quick_start_example()
{
	const size_t degree = 3;  // Degree of spline.
	const size_t dim = 2;  // Dimension of each point.
	const size_t numControlPoints = 7;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.

	// Create a clamped spline of degree 3 in 2D consisting of 7 control points.
	ts::BSpline spline(degree, dim, numControlPoints, splineType);
	{
		// Set up the control points.
		std::vector<ts::rational> ctrlp = spline.ctrlp();
		ctrlp[0] = -1.75f; // x0.
		ctrlp[1] = -1.0f;  // y0.
		ctrlp[2] = -1.5f;  // x1.
		ctrlp[3] = -0.5f;  // y1.
		ctrlp[4] = -1.5f;  // x2.
		ctrlp[5] = 0.0f;  // y2.
		ctrlp[6] = -1.25f;  // x3.
		ctrlp[7] = 0.5f;  // y3.
		ctrlp[8] = -0.75f;  // x4.
		ctrlp[9] = 0.75f;  // y4.
		ctrlp[10] = 0.0f;  // x5.
		ctrlp[11] = 0.5f;  // y5.
		ctrlp[12] = 0.5f;  // x6.
		ctrlp[13] = 0.0f;  // y6.
		spline.setCtrlp(ctrlp);
	}

	//
	{
		// Evaluate 'spline' at u = (x - xmin) / (xmax - xmin).
		const ts::rational u = 0.4f;
#if 1
		const ts::DeBoorNet &net = spline.evaluate(u);
		//const ts::DeBoorNet &net = spline(u);
		const std::vector<ts::rational> &result = net.result();
#else
		const std::vector<ts::rational> &result = spline.evaluate(u).result();
		//const std::vector<ts::rational> &result = spline(u).result();
#endif
		std::cout << "x = " << result[0] << ", y = " << result[1] << std::endl;
	}

	{
		// Derive 'spline' and subdivide it into a sequence of Bezier curves.
		ts::BSpline beziers = spline.derive().toBeziers();

		// Evaluate 'beziers' at u = (x - xmin) / (xmax - xmin).
		const ts::rational u = 0.3f;
		//const std::vector<ts::rational> &result = beziers.evaluate(u).result();
		const std::vector<ts::rational> &result = beziers(u).result();
		std::cout << "x = " << result[0] << ", y = " << result[1] << std::endl;
	}
}

// REF [file] >> ${TINYSPLINE_HOME}/examples/c/bspline.c
void bspline_example()
{
	const size_t degree = 3;  // Degree of spline.
	const size_t dim = 3;  // Dimension of each point.
	const size_t numControlPoints = 7;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.
	tsError retval;

	// Create a clamped spline.
	tsBSpline spline;
	retval = ts_bspline_new(degree, dim, numControlPoints, splineType, &spline);
	assert(TS_SUCCESS == retval);
	{
		// Set up the control points.
		spline.ctrlp[0] = -1.75;
		spline.ctrlp[1] = -1.0;
		spline.ctrlp[2] = 0.0;

		spline.ctrlp[3] = -1.5;
		spline.ctrlp[4] = -0.5;
		spline.ctrlp[5] = 0.0;

		spline.ctrlp[6] = -1.5;
		spline.ctrlp[7] = 0.0;
		spline.ctrlp[8] = 0.0;

		spline.ctrlp[9] = -1.25;
		spline.ctrlp[10] = 0.5;
		spline.ctrlp[11] = 0.0;

		spline.ctrlp[12] = -0.75;
		spline.ctrlp[13] = 0.75;
		spline.ctrlp[14] = 0.0;

		spline.ctrlp[15] = 0.0;
		spline.ctrlp[16] = 0.5;
		spline.ctrlp[17] = 0.0;

		spline.ctrlp[18] = 0.5;
		spline.ctrlp[19] = 0.0;
		spline.ctrlp[20] = 0.0;
	}

	//
	{
		tsDeBoorNet net;
		const tsRational u = 0.5f;  // u = (x - xmin) / (xmax - xmin).
		retval = ts_bspline_evaluate(&spline, u, &net);
		assert(TS_SUCCESS == retval);

		// Do something.
		std::cout << "x = " << net.result[0] << ", y = " << net.result[1] << std::endl;

		ts_deboornet_free(&net);
	}

	// Tear down.
	ts_bspline_free(&spline);
}

// REF [file] >> ${TINYSPLINE_HOME}/examples/c/nurbs.c
void nurbs_example()
{
	const size_t degree = 2;  // Degree of spline.
	const size_t dim = 4;  // Dimension of each point.
	const size_t numControlPoints = 9;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.
	tsError retval;

	// Create a clamped spline.
	tsBSpline spline;
	retval = ts_bspline_new(degree, dim, numControlPoints, splineType, &spline);
	assert(TS_SUCCESS == retval);
	{
		// Set up the control points.
		const tsRational w = (tsRational)(std::sqrt(2.0f) / 2.0f);

		spline.ctrlp[0] = 1.f;
		spline.ctrlp[1] = 0.f;
		spline.ctrlp[2] = 0.f;
		spline.ctrlp[3] = 1.f;

		spline.ctrlp[4] = w;
		spline.ctrlp[5] = w;
		spline.ctrlp[6] = 0.f;
		spline.ctrlp[7] = w;

		spline.ctrlp[8] = 0.f;
		spline.ctrlp[9] = 1.f;
		spline.ctrlp[10] = 0.f;
		spline.ctrlp[11] = 1.f;

		spline.ctrlp[12] = -w;
		spline.ctrlp[13] = w;
		spline.ctrlp[14] = 0.f;
		spline.ctrlp[15] = w;

		spline.ctrlp[16] = -1.f;
		spline.ctrlp[17] = 0.f;
		spline.ctrlp[18] = 0.f;
		spline.ctrlp[19] = 1.f;

		spline.ctrlp[20] = -w;
		spline.ctrlp[21] = -w;
		spline.ctrlp[22] = 0.f;
		spline.ctrlp[23] = w;

		spline.ctrlp[24] = 0.f;
		spline.ctrlp[25] = -1.f;
		spline.ctrlp[26] = 0.f;
		spline.ctrlp[27] = 1.f;

		spline.ctrlp[28] = w;
		spline.ctrlp[29] = -w;
		spline.ctrlp[30] = 0.f;
		spline.ctrlp[31] = w;

		spline.ctrlp[32] = 1.f;
		spline.ctrlp[33] = 0.f;
		spline.ctrlp[34] = 0.f;
		spline.ctrlp[35] = 1.f;

		spline.knots[0] = 0.f;
		spline.knots[1] = 0.f;
		spline.knots[2] = 0.f;
		spline.knots[3] = 1.f / 4.f;
		spline.knots[4] = 1.f / 4.f;
		spline.knots[5] = 2.f / 4.f;
		spline.knots[6] = 2.f / 4.f;
		spline.knots[7] = 3.f / 4.f;
		spline.knots[8] = 3.f / 4.f;
		spline.knots[9] = 1.f;
		spline.knots[10] = 1.f;
		spline.knots[11] = 1.f;
	}

	//
	{
		//spline.order
		//spline.dim
		//spline.n_ctrlp
		//spline.ctrlp
		//spline.n_knots
		//spline.knots

		//for (int i = 0; i < spline.n_ctrlp; ++i)
		//	spline.ctrlp[i * spline.dim];

		tsDeBoorNet net;
		const tsRational u = 0.0f;  // u = (x - xmin) / (xmax - xmin).
		retval = ts_bspline_evaluate(&spline, u, &net);
		assert(TS_SUCCESS == retval);

		// Do something.
		std::cout << "x = " << net.result[0] << ", y = " << net.result[1] << std::endl;

		ts_deboornet_free(&net);
	}

	// Tear down.
	ts_bspline_free(&spline);
}

// REF [file] >> ${TINYSPLINE_HOME}/examples/c/beziers.c
void bezier_example()
{
	const size_t degree = 3;  // Degree of spline.
	const size_t dim = 3;  // Dimension of each point.
	const size_t numControlPoints = 7;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.
	const bool drawBeziers = false;  // false - bspline, true - beziers.
	tsError retval;

	// Create a clamped spline.
	tsBSpline spline;
	retval = ts_bspline_new(degree, dim, numControlPoints, splineType, &spline);
	assert(TS_SUCCESS == retval);
	{
		// Set up the control points.
		spline.ctrlp[0] = -1.75;
		spline.ctrlp[1] = -1.0;
		spline.ctrlp[2] = 0.0;

		spline.ctrlp[3] = -1.5;
		spline.ctrlp[4] = -0.5;
		spline.ctrlp[5] = 0.0;

		spline.ctrlp[6] = -1.5;
		spline.ctrlp[7] = 0.0;
		spline.ctrlp[8] = 0.0;

		spline.ctrlp[9] = -1.25;
		spline.ctrlp[10] = 0.5;
		spline.ctrlp[11] = 0.0;

		spline.ctrlp[12] = -0.75;
		spline.ctrlp[13] = 0.75;
		spline.ctrlp[14] = 0.0;

		spline.ctrlp[15] = 0.0;
		spline.ctrlp[16] = 0.5;
		spline.ctrlp[17] = 0.0;

		spline.ctrlp[18] = 0.5;
		spline.ctrlp[19] = 0.0;
		spline.ctrlp[20] = 0.0;
	}

	//
	{
		//spline.order
		//spline.dim
		//spline.n_ctrlp
		//spline.ctrlp
		//spline.n_knots
		//spline.knots

		//for (int i = 0; i < spline.n_ctrlp; ++i)
		//	spline.ctrlp[i * spline.dim];

		tsBSpline draw;
		if (drawBeziers)
			retval = ts_bspline_to_beziers(&spline, &draw);
		else
			retval = ts_bspline_copy(&spline, &draw);
		assert(TS_SUCCESS == retval);

		// Do something.
#if 0
		{
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glColor3f(1.0, 1.0, 1.0);
			glLineWidth(3);
			gluBeginCurve(theNurb);
			gluNurbsCurve(
				theNurb,
				(GLint)draw.n_knots,
				draw.knots,
				(GLint)draw.dim,
				draw.ctrlp,
				(GLint)draw.order,
				GL_MAP1_VERTEX_3
			);
			gluEndCurve(theNurb);

			// Draw control points.
			glColor3f(1.0, 0.0, 0.0);
			glPointSize(5.0);
			glBegin(GL_POINTS);
			for (size_t i = 0; i < spline.n_ctrlp; ++i)
				glVertex3fv(&spline.ctrlp[i * spline.dim]);
			glEnd();

			glutSwapBuffers();
			glutPostRedisplay();
		}
#endif

		ts_bspline_free(&draw);
	}

	// Tear down.
	ts_bspline_free(&spline);
}

// REF [file] >> ${TINYSPLINE_HOME}/examples/c/derivative.c
void derivative_example()
{
	const size_t degree = 3;  // Degree of spline.
	const size_t dim = 3;  // Dimension of each point.
	const size_t numControlPoints = 7;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.
	tsError retval;

	// Create a clamped spline.
	tsBSpline spline;
	retval = ts_bspline_new(degree, dim, numControlPoints, splineType, &spline);
	assert(TS_SUCCESS == retval);
	{
		// Set up the control points.
		spline.ctrlp[0] = -1.75;
		spline.ctrlp[1] = -1.0;
		spline.ctrlp[2] = 0.0;
		spline.ctrlp[3] = -1.5;
		spline.ctrlp[4] = -0.5;
		spline.ctrlp[5] = 0.0;
		spline.ctrlp[6] = -1.5;
		spline.ctrlp[7] = 0.0;
		spline.ctrlp[8] = 0.0;
		spline.ctrlp[9] = -1.25;
		spline.ctrlp[10] = 0.5;
		spline.ctrlp[11] = 0.0;
		spline.ctrlp[12] = -0.75;
		spline.ctrlp[13] = 0.75;
		spline.ctrlp[14] = 0.0;
		spline.ctrlp[15] = 0.0;
		spline.ctrlp[16] = 0.5;
		spline.ctrlp[17] = 0.0;
		spline.ctrlp[18] = 0.5;
		spline.ctrlp[19] = 0.0;
		spline.ctrlp[20] = 0.0;
	}

	tsBSpline derivative;
	retval = ts_bspline_derive(&spline, &derivative);
	assert(TS_SUCCESS == retval);

	//
	{
		const tsRational u = 0.0f;  // u = (x - xmin) / (xmax - xmin).

		tsDeBoorNet net1, net2, net3;
		retval = ts_bspline_evaluate(&spline, u, &net1);
		assert(TS_SUCCESS == retval);
		retval = ts_bspline_evaluate(&derivative, u, &net2);
		assert(TS_SUCCESS == retval);
		retval = ts_bspline_evaluate(&derivative, u, &net3);
		assert(TS_SUCCESS == retval);

		std::cout << "x = " << net1.result[0] << ", y = " << net1.result[1] << std::endl;
		std::cout << "x = " << net2.result[0] << ", y = " << net2.result[1] << std::endl;
		std::cout << "x = " << net3.result[0] << ", y = " << net3.result[1] << std::endl;

		ts_deboornet_free(&net1);
		ts_deboornet_free(&net2);
		ts_deboornet_free(&net3);
	}

	// Tear down.
	ts_bspline_free(&spline);
	ts_bspline_free(&derivative);
}

void cubic_spline_interpolation_and_derivative()
{
	//const size_t degree = 3;  // Degree of spline.
	const size_t dim = 2;  // Dimension of each point.
	tsError retval;

	// Cubic spline interpolation using Thomas algorithm.
	// Create a spline which is a sequence of bezier curves connecting each point in points.
	// Each bezier curve is of degree 3 with dimension dim.
	// The total number of control points is (n - 1)*4.
	// n is the number of points in points and not the length of points.
	tsBSpline spline;
	{
		// Set up the points.
#if 1
		const size_t n = 5;  // Number of points.
		tsRational points[10] = { 0.0, };

		points[0] = -1.0;
		points[1] = 1.0;

		points[2] = -0.5;
		points[3] = 0.25;

		points[4] = 0.0;
		points[5] = 0.0;

		points[6] = 0.5;
		points[7] = 0.25;

		points[8] = 1.0;
		points[9] = 1.0;
#else
		const size_t n = 7;  // Number of points.
		tsRational points[14] = { 0.0, };

		points[0] = -1.0;
		points[1] = 1.0;

		points[2] = -0.75;
		points[3] = 0.5625;

		points[4] = -0.5;
		points[5] = 0.25;

		points[6] = 0.0;
		points[7] = 0.0;

		points[8] = 0.5;
		points[9] = 0.25;

		points[10] = 0.5625;
		points[11] = 0.25;

		points[12] = 1.0;
		points[13] = 1.0;
#endif
		retval = ts_bspline_interpolate(points, n, dim, &spline);
		assert(TS_SUCCESS == retval);
	}

	//
	{
		//const tsRational x = 0.25;
		const tsRational u = 0.625;  // u = (x - xmin) / (xmax - xmin) = (0.25 - -1.0) / (1.0 - -1.0) = 0.625.
		tsDeBoorNet net;

		// Evaluate spline.
		retval = ts_bspline_evaluate(&spline, u, &net);
		assert(TS_SUCCESS == retval);
		std::cout << "Spline: x = " << net.result[0] << ", y = " << net.result[1] << std::endl;  // Expected: 0.0580357.
		ts_deboornet_free(&net);
/*
		// Evaluate the 1st derivative of spline.
		tsBSpline deriv1;
		retval = ts_bspline_derive(&spline, &deriv1);  // NOTICE [error] >> Run-time error: underivable.
		assert(TS_SUCCESS == retval);

		retval = ts_bspline_evaluate(&deriv1, u, &net);
		assert(TS_SUCCESS == retval);
		std::cout << "The 1st derivative of spline: x = " << net.result[0] << ", y = " << net.result[1] << std::endl;
		ts_deboornet_free(&net);

		// Evaluate the 2nd derivative of spline.
		tsBSpline deriv2;
		retval = ts_bspline_derive(&deriv1, &deriv2);  // NOTICE [error] >> Run-time error: underivable.
		assert(TS_SUCCESS == retval);

		retval = ts_bspline_evaluate(&deriv2, u, &net);
		assert(TS_SUCCESS == retval);
		std::cout << "The 2nd derivative of spline: x = " << net.result[0] << ", y = " << net.result[1] << std::endl;
		ts_deboornet_free(&net);

		ts_bspline_free(&deriv2);
		ts_bspline_free(&deriv1);
*/
	}

	// Tear down.
	ts_bspline_free(&spline);
}

}  // namespace local
}  // unnamed namespace

namespace my_tinyspline {

}  // namespace my_tinyspline

int tinyspline_main(int argc, char *argv[])
{
	// C example ------------------------------------------
	//local::bspline_example();
	//local::nurbs_example();
	//local::bezier_example();
	//local::buckle_example();  // Not yet implemented.
	//local::derivative_example();

	// C++ example ----------------------------------------
	//local::quick_start_example();

	// Interpolation & derivative -------------------------
	//	REF [site] >> http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
	local::cubic_spline_interpolation_and_derivative();  // NOTICE [info] >> It has some error on differentiation.

	return 0;
}
