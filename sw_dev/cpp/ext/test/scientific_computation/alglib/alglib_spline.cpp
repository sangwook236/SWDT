//#include "stdafx.h"
#include <interpolation.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

// REF [site] >> http://www.alglib.net/translator/man/manual.cpp.html#example_spline1d_d_linear
void linear_spline_example()
{
	// Use piecewise linear spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// Build spline.
	alglib::spline1dinterpolant spline;
	alglib::spline1dbuildlinear(x, y, spline);
	//alglib::spline1dbuildlinear(x, y, 5, spline);

	// Calculate S(0.25) - it is quite different from 0.25^2 = 0.0625.
	const double t = 0.25;
	const double v = alglib::spline1dcalc(spline, t);
	std::cout << "Spline at t = " << v << std::endl;  // EXPECTED: 0.125.
}

// REF [site] >> http://www.alglib.net/translator/man/manual.cpp.html#example_spline1d_d_cubic
void cubic_spline_example()
{
	// Use cubic spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// First, use default boundary conditions ("parabolically terminated spline") because cubic spline built with such boundary conditions will exactly reproduce any quadratic f(x).
	// Calculate S(0.25) (almost same as original function).
	{
		alglib::spline1dinterpolant spline;
		alglib::spline1dbuildcubic(x, y, spline);

		double t = 0.25;
		const double v = alglib::spline1dcalc(spline, t);
		std::cout << "Spline at t = " << v << std::endl;  // EXPECTED: 0.0625.
	}

	// Then try to use natural boundary conditions
	//     d2S(-1)/dx^2 = 0.0
	//     d2S(+1)/dx^2 = 0.0
	// and see that such spline interpolated f(x) with small error.
	// Calculate S(0.25) (small interpolation error).
	{
		const alglib::ae_int_t natural_bound_type = 2;
		alglib::spline1dinterpolant spline;
		alglib::spline1dbuildcubic(x, y, 5, natural_bound_type, 0.0, natural_bound_type, 0.0, spline);

		double t = 0.25;
		const double v = alglib::spline1dcalc(spline, t);
		std::cout << "Spline at t = " << v << std::endl;  // EXPECTED: 0.0580.
	}
}

// REF [site] >> http://www.alglib.net/translator/man/manual.cpp.html#example_spline1d_d_griddiff
void differentiation_on_grid_example()
{
	// Use cubic spline to do grid differentiation, i.e. having values of f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1]
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// Use default boundary conditions ("parabolically terminated spline") because cubic spline built with such boundary conditions will exactly reproduce any quadratic f(x).
	// Actually, we could use natural conditions, but we feel that spline which exactly reproduces f() will show us more understandable results.

	// Calculate first derivatives: they must be equal to 2*x.
	// Calculate derivatives of cubic spline at nodes WITHOUT CONSTRUCTION OF SPLINE OBJECT.
	//	Efficient functions: spline1dgriddiffcubic() and spline1dgriddiff2cubic().
	{
		alglib::real_1d_array d1;
		alglib::spline1dgriddiffcubic(x, y, d1);
		std::cout << "The first deriative of spline = " << d1.tostring(3) << std::endl;  // EXPECTED: [-2.0, -1.0, 0.0, +1.0, +2.0].
	}

	// Calculate first and second derivatives.
	// First derivative is 2*x, second is equal to 2.0.
	{
		alglib::real_1d_array d1, d2;
		alglib::spline1dgriddiff2cubic(x, y, d1, d2);
		std::cout << "The first deriative of spline = " << d1.tostring(3) << std::endl;  // EXPECTED: [-2.0, -1.0, 0.0, +1.0, +2.0].
		std::cout << "The second deriative of spline = " << d2.tostring(3) << std::endl;  // EXPECTED: [ 2.0,  2.0, 2.0,  2.0,  2.0].
	}
}

// REF [site] >> http://www.alglib.net/translator/man/manual.cpp.html#example_spline1d_d_convdiff
void conversion_from_one_grid_to_another_example()
{
	// Use cubic spline to do resampling, i.e. having values of f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x_old = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y_old = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.
	const alglib::real_1d_array x_new = "[-1.00, -0.75, -0.50, -0.25, 0.00, +0.25, +0.50, +0.75, +1.00]";

	// Calculate values/derivatives of cubic spline on another grid (equidistant with 9 nodes on [-1, +1]) WITHOUT CONSTRUCTION OF SPLINE OBJECT.
	//	Efficient functions: spline1dconvcubic(), spline1dconvdiffcubic() and spline1dconvdiff2cubic().

	// Use default boundary conditions ("parabolically terminated spline") because cubic spline built with such boundary conditions will exactly reproduce any quadratic f(x).
	// Actually, we could use natural conditions, but we feel that spline which exactly reproduces f() will show us more understandable results.

	// First, conversion without differentiation.
	{
		alglib::real_1d_array y_new;
		alglib::spline1dconvcubic(x_old, y_old, x_new, y_new);

		std::cout << "Conversion without differentiation" << std::endl;
		std::cout << '\t' << y_new.tostring(3) << std::endl;  // EXPECTED: [1.0000, 0.5625, 0.2500, 0.0625, 0.0000, 0.0625, 0.2500, 0.5625, 1.0000].
	}

	// Then, conversion with differentiation (first derivatives only).
	{
		alglib::real_1d_array y_new;
		alglib::real_1d_array d1_new;
		alglib::spline1dconvdiffcubic(x_old, y_old, x_new, y_new, d1_new);

		std::cout << "Conversion with the first derivative only" << std::endl;
		std::cout << '\t' << y_new.tostring(3) << std::endl;  // EXPECTED: [1.0000, 0.5625, 0.2500, 0.0625, 0.0000, 0.0625, 0.2500, 0.5625, 1.0000].
		std::cout << '\t' << d1_new.tostring(3) << std::endl;  // EXPECTED: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0].
	}

	// Finally, conversion with first and second derivatives.
	{
		alglib::real_1d_array y_new;
		alglib::real_1d_array d1_new, d2_new;
		alglib::spline1dconvdiff2cubic(x_old, y_old, x_new, y_new, d1_new, d2_new);

		std::cout << "Conversion with the first and second derivatives" << std::endl;
		std::cout << '\t' << y_new.tostring(3) << std::endl;  // EXPECTED: [1.0000, 0.5625, 0.2500, 0.0625, 0.0000, 0.0625, 0.2500, 0.5625, 1.0000].
		std::cout << '\t' << d1_new.tostring(3) << std::endl;  // EXPECTED: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0].
		std::cout << '\t' << d2_new.tostring(3) << std::endl;  // EXPECTED: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0].
	}
}

void hermite_spline()
{
	// Use piecewise linear spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.
	const alglib::real_1d_array d = "[+2.0, +1.0, 0.0, +1.0, +2.0]";  // Derivatives.

	// Build spline.
	alglib::spline1dinterpolant spline;
	alglib::spline1dbuildhermite(x, y, d, spline);
	//alglib::spline1dbuildhermite(x, y, d, 5, spline);

	// Calculate S(0.25) - it is quite different from 0.25^2 = 0.0625.
	const double t = 0.25;
	const double v = alglib::spline1dcalc(spline, t);
	std::cout << "Hermite spline at t = " << v << std::endl;  // EXPECTED: 0.0625.
}

void akima_spline()
{
	// Use piecewise linear spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// Build spline.
	alglib::spline1dinterpolant spline;
	alglib::spline1dbuildakima(x, y, spline);

	// Calculate S(0.25) - it is quite different from 0.25^2 = 0.0625.
	const double t = 0.25;
	const double v = alglib::spline1dcalc(spline, t);
	std::cout << "Akima spline at t = " << v << std::endl;  // EXPECTED: 0.0625.
}

void spline_differentiation()
{
	// Use piecewise linear spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// Build spline.
	alglib::spline1dinterpolant spline;
	//alglib::spline1dbuildcubic(x, y, spline);
	//const alglib::ae_int_t natural_bound_type = 2;
	//alglib::spline1dbuildcubic(x, y, 5, natural_bound_type, 0.0, natural_bound_type, 0.0, spline);
	alglib::spline1dbuildakima(x, y, spline);

	// Differentiate.
	const double t = 0.25;
	double s = 0.0, ds = 0.0, d2s = 0.0;
	alglib::spline1ddiff(spline, t, s, ds, d2s);
	std::cout << "Derivative of spline:" << std::endl;
	std::cout << "\tt = " << t << ", s = " << s << ", ds = " << ds << ", d2s = " << d2s << std::endl;
}

void spline_integration()
{
	// Use piecewise linear spline to interpolate f(x)=x^2 sampled at 5 equidistant nodes on [-1, +1].
	const alglib::real_1d_array x = "[-1.0, -0.5, 0.0, +0.5, +1.0]";
	const alglib::real_1d_array y = "[+1.0, 0.25, 0.0, 0.25, +1.0]";  // Function values.

	// Build spline.
	alglib::spline1dinterpolant spline;
	//alglib::spline1dbuildcubic(x, y, spline);
	//const alglib::ae_int_t natural_bound_type = 2;
	//alglib::spline1dbuildcubic(x, y, 5, natural_bound_type, 0.0, natural_bound_type, 0.0, spline);
	alglib::spline1dbuildakima(x, y, spline);

	// Integrate.
	const double t = 0.25;
	const double sdt = alglib::spline1dintegrate(spline, t);
	std::cout << "Integration of spline:" << std::endl;
	std::cout << "\tt = (-1," << t << "), sdt = " << sdt << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_alglib {

void spline()
{
	//local::linear_spline_example();
	//local::cubic_spline_example();
	//local::differentiation_on_grid_example();
	//local::conversion_from_one_grid_to_another_example();

	//
	//local::hermite_spline();
	//local::akima_spline();

	// Differentiation and integration of spline.
	local::spline_differentiation();
	local::spline_integration();
}

}  // namespace my_alglib
