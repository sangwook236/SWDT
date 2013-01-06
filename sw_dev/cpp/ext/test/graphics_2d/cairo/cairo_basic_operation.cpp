//#include "stdafx.h"
#include <cairo/cairo.h>
#include <cairo/cairo-svg.h>
#include <cmath>


#if defined(_MSC_VER)
const double M_PI = std::atan(1.0) * 4.0;
#endif

namespace {
namespace local {

void draw_ellipse()
{
	const int width = 500, height = 500;

#if defined(CAIRO_HAS_SVG_SURFACE)
	cairo_surface_t *surface = cairo_svg_surface_create(".\\graphics_2d_data\\cairo\\basic_drawing_1.svg", (double)width, (double)height);
#elif defined(CAIRO_HAS_IMAGE_SURFACE)
	cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
#elif defined(CAIRO_HAS_PDF_SURFACE)
	cairo_surface_t *surface = cairo_pdf_surface_create(".\\graphics_2d_data\\cairo\\basic_drawing_1.pdf", (double)width, (double)height);
#elif defined(CAIRO_HAS_WIN32_SURFACE)
	cairo_surface_t *surface = cairo_win32_surface_create(hdc);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_dib(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_ddb(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_printing_surface_create(hdc);
#endif

    cairo_t *cr = cairo_create(surface);

	{
		const double xc = 128.0;
		const double yc = 128.0;
		const double radius = 100.0;
		const double angle1 = 45.0  * (M_PI / 180.0);  // angles are specified in radians
		const double angle2 = 180.0 * (M_PI / 180.0);

		cairo_set_line_width(cr, 10.0);
		cairo_arc(cr, xc, yc, radius, angle1, angle2);
		cairo_stroke(cr);

		// draw helping lines
		cairo_set_source_rgba(cr, 1, 0.2, 0.2, 0.6);
		cairo_set_line_width(cr, 6.0);

		cairo_arc(cr, xc, yc, 10.0, 0, 2 * M_PI);
		cairo_fill(cr);

		cairo_arc(cr, xc, yc, radius, angle1, angle1);
		cairo_line_to(cr, xc, yc);
		cairo_arc(cr, xc, yc, radius, angle2, angle2);
		cairo_line_to(cr, xc, yc);
		cairo_stroke(cr);
	}

	// save to a png file
	cairo_surface_write_to_png(surface, ".\\graphics_2d_data\\cairo\\basic_drawing_1_out.png");

	//
    cairo_destroy(cr);
	cairo_surface_destroy(surface);
}

}  // namespace local
}  // unnamed namespace

namespace cairo {

void basic_drawing()
{
	local::draw_ellipse();
}

}  // namespace cairo
