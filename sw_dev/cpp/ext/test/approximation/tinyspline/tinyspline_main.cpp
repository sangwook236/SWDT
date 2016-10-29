#include <tinysplinecpp.h>
#include <tinyspline.h>
#include <iostream>
#include <vector>


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

	//
	{
		// Evaluate 'spline' at u = 0.4.
		//const ts::DeBoorNet net = spline.evaluate(0.4f);
		//const std::vector<ts::rational> result = net.result();
		const std::vector<ts::rational> result = spline.evaluate(0.4f).result();  // You can use '()' instead of 'evaluate'.
		std::cout << "x = " << result[0] << ", y = " << result[1] << std::endl;
	}

	{
		// Derive 'spline' and subdivide it into a sequence of Bezier curves.
		ts::BSpline beziers = spline.derive().toBeziers();

		// Evaluate 'beziers' at u = 0.3.
		const std::vector<ts::rational> result = beziers(0.3f).result();  // You can use '()' instead of 'evaluate'.
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

	//
	{
		tsDeBoorNet net;
		retval = ts_bspline_evaluate(&spline, 0.5f, &net);
		//net.result

		// Do something.

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
		tsRational u = 0.0f;
		retval = ts_bspline_evaluate(&spline, u, &net);
		//net.result

		// Do something.

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

		// Do something.

		ts_bspline_free(&draw);
	}

	// Tear down.
	ts_bspline_free(&spline);
}

void bspline_derivative_example()
{
	const size_t degree = 3;  // Degree of spline.
	const size_t dim = 3;  // Dimension of each point.
	const size_t numControlPoints = 7;  // Number of control points.
	const tsBSplineType splineType = TS_CLAMPED;  // Used to hit first and last control point.
	tsError retval;

	// Create a clamped spline.
	tsBSpline spline;
	retval = ts_bspline_new(degree, dim, numControlPoints, splineType, &spline);

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

	//
	{
		tsBSpline deriv;
		retval = ts_bspline_derive(&spline, &deriv);
		//net.result

		// Do something.

		ts_bspline_free(&deriv);
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
	// C++ example ----------------------------------------
	local::quick_start_example();

	// C example ------------------------------------------
	local::bspline_example();
	local::nurbs_example();
	local::bezier_example();

	// Derivative -----------------------------------------
	// REF [site] >> http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
	local::bspline_derivative_example();

	return 0;
}
