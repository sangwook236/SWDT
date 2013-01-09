//#include "stdafx.h"
#include <cairo/cairo.h>
#include <cairo/cairo-svg.h>
#include <boost/math/constants/constants.hpp>
#include <iomanip>
#include <cmath>


namespace {
namespace local {

void draw_link(cairo_t *cr, const int X, const int Y, const double L, const int W, const double theta)
{
	const int X2 = X + (int)std::floor(L * std::cos(theta) + 0.5), Y2 = Y + (int)std::floor(L * std::sin(theta) + 0.5);
	const int X_off = (int)std::floor(W * 0.5 * std::sin(theta) + 0.5);
	const int Y_off = (int)std::floor(W * 0.5 * std::cos(theta) + 0.5);

	//cairo_set_line_width(cr, 1.0);

	cairo_move_to(cr, X + X_off, Y - Y_off);
	cairo_line_to(cr, X2 + X_off, Y2 - Y_off);
	cairo_line_to(cr, X2 - X_off, Y2 + Y_off);
	cairo_line_to(cr, X - X_off, Y + Y_off);
	//cairo_line_to(cr, X + X_off, Y - Y_off);
	cairo_fill(cr);

	cairo_arc(cr, X, Y, W / 2, 0, 2 * boost::math::constants::pi<double>());
	cairo_arc(cr, X2, Y2, W / 2, 0, 2 * boost::math::constants::pi<double>());
	cairo_fill(cr);
}

void draw_arm(const double L1, const double W1, const double L2, const double W2, const double theta1, const double theta2, const int index)
{
	const int WIDTH = 96, HEIGHT = 128;

	cairo_surface_t *surface = cairo_svg_surface_create(".\\graphics_2d_data\\cairo\\two_link_arm.svg", (double)WIDTH, (double)HEIGHT);
    cairo_t *cr = cairo_create(surface);

	//---------------------------------------------------------------
	//

	const int X1 = WIDTH / 3, Y1 = HEIGHT / 2;
	const int X2 = X1 + (int)std::floor(L1 * std::cos(theta1) + 0.5), Y2 = Y1 + (int)std::floor(L1 * std::sin(theta1) + 0.5);

	// background color
	cairo_rectangle(cr, 0.0, 0.0, WIDTH, HEIGHT);
	cairo_set_source_rgb(cr, 0.0, 0.0, 0.0);
	cairo_fill(cr);

	// draw arm
	cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
	draw_link(cr, X1, Y1, L1, W1, theta1);  // upper-arm
	draw_link(cr, X2, Y2, L2, W2, theta1 + theta2);  // forearm

	// save to a png file
	std::ostringstream sstream;
	sstream << ".\\graphics_2d_data\\cairo\\two_link_arm_" << std::setfill('0') << std::setw(2) << index << ".png";
	cairo_surface_write_to_png(surface, sstream.str().c_str());

	//
    cairo_destroy(cr);
	cairo_surface_destroy(surface);
}

}  // namespace local
}  // unnamed namespace

namespace my_cairo {

void two_link_arm()
{
	const int L1 = 30, W1 = 6;  // upper-arm
	const int L2 = 20, W2 = 6;  // forearm

	const double theta1 = boost::math::constants::pi<double>() / 4.0;
	const double theta2 = boost::math::constants::pi<double>() / 4.0;

	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			local::draw_arm(L1, W1, L2, W2, theta1 * (i - 2), theta2 * (j - 2), i * 5 + j + 1);
}

}  // namespace my_cairo
