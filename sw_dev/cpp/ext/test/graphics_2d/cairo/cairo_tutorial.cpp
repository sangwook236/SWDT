//#include "stdafx.h"
#include <cairo/cairo.h>
#include <cairo/cairo-svg.h>
#include <cmath>


#if defined(_MSC_VER)
const double M_PI = std::atan(1.0) * 4.0;
#endif

namespace {
namespace local {

void tutorial_1()
{
	const int width = 100, height = 100;

#if defined(CAIRO_HAS_SVG_SURFACE)
	cairo_surface_t *surface = cairo_svg_surface_create(".\\graphics_2d_data\\cairo\\tutorial_1.svg", (double)width, (double)height);
#elif defined(CAIRO_HAS_IMAGE_SURFACE)
	cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
#elif defined(CAIRO_HAS_PDF_SURFACE)
	cairo_surface_t *surface = cairo_pdf_surface_create(".\\graphics_2d_data\\cairo\\tutorial_1.pdf", (double)width, (double)height);
#elif defined(CAIRO_HAS_WIN32_SURFACE)
	cairo_surface_t *surface = cairo_win32_surface_create(hdc);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_dib(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_ddb(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_printing_surface_create(hdc);
#endif

    cairo_t *cr = cairo_create(surface);

	cairo_scale(cr, (double)width, (double)height);

	// use color as source
	{
		cairo_set_source_rgb(cr, 0, 0, 0);
		cairo_move_to(cr, 0, 0);
		cairo_line_to(cr, 1, 1);
		cairo_move_to(cr, 1, 0);
		cairo_line_to(cr, 0, 1);
		cairo_set_line_width(cr, 0.2);
		cairo_stroke(cr);

		cairo_rectangle(cr, 0, 0, 0.5, 0.5);
		cairo_set_source_rgba(cr, 1, 0, 0, 0.80);
		cairo_fill(cr);

		cairo_rectangle(cr, 0, 0.5, 0.5, 0.5);
		cairo_set_source_rgba(cr, 0, 1, 0, 0.60);
		cairo_fill(cr);

		cairo_rectangle(cr, 0.5, 0, 0.5, 0.5);
		cairo_set_source_rgba(cr, 0, 0, 1, 0.40);
		cairo_fill(cr);
	}

	// save to a png file
	cairo_surface_write_to_png(surface, ".\\graphics_2d_data\\cairo\\tutorial_1_out.png");

	//
    cairo_destroy(cr);
	cairo_surface_destroy(surface);
}

void tutorial_2()
{
	const int width = 100, height = 100;

#if defined(CAIRO_HAS_SVG_SURFACE)
	cairo_surface_t *surface = cairo_svg_surface_create(".\\graphics_2d_data\\cairo\\tutorial_2.svg", (double)width, (double)height);
#elif defined(CAIRO_HAS_IMAGE_SURFACE)
	cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
#elif defined(CAIRO_HAS_PDF_SURFACE)
	cairo_surface_t *surface = cairo_pdf_surface_create(".\\graphics_2d_data\\cairo\\tutorial_2.pdf", (double)width, (double)height);
#elif defined(CAIRO_HAS_WIN32_SURFACE)
	cairo_surface_t *surface = cairo_win32_surface_create(hdc);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_dib(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_ddb(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_printing_surface_create(hdc);
#endif

    cairo_t *cr = cairo_create(surface);

	cairo_scale(cr, (double)width, (double)height);

	// use image as source
	cairo_save(cr);
	{
		// read from a png file
		cairo_surface_t *img = cairo_image_surface_create_from_png(".\\graphics_2d_data\\cairo\\pattern_1.png");

		const int w = cairo_image_surface_get_width(img);
		const int h = cairo_image_surface_get_height(img);
		//const unsigned char *data = cairo_image_surface_get_data(img);
		//const cairo_format_t format = cairo_image_surface_get_format(img);
		//const int stride = cairo_image_surface_get_stride(img);

		//cairo_push_group(cr);
			cairo_scale(cr, 1.0 / (double)w, 1.0 / (double)h);

#if 1
			cairo_set_source_surface(cr, img, 0.0, 0.0);
#else
			cairo_pattern_t *pat = cairo_pattern_create_for_surface(img);
			cairo_set_source(cr, pat);
			cairo_pattern_destroy(pat);
#endif

			cairo_surface_destroy(img);
		//	cairo_paint(cr);
		//cairo_pop_group_to_source(cr);

		//cairo_paint(cr);
		cairo_paint_with_alpha(cr, 0.5);
	}
	cairo_restore(cr);

	// use gradient as source
	{
		//cairo_pattern_t *radpat = cairo_pattern_create_radial(0.25, 0.25, 0.1,  0.5, 0.5, 0.5);
		cairo_pattern_t *radpat = cairo_pattern_create_radial(0.25, 0.25, 0.25,  0.5, 0.5, 0.25);
		cairo_pattern_add_color_stop_rgb(radpat, 0,  1.0, 0.0, 0.0);
		cairo_pattern_add_color_stop_rgb(radpat, 1,  0.0, 0.0, 1.0);

		for (int i = 1; i < 10; ++i)
			for (int j = 1; j < 10; ++j)
				cairo_rectangle(cr, i/10.0 - 0.04, j/10.0 - 0.04, 0.08, 0.08);
		cairo_set_source(cr, radpat);
		//cairo_pattern_destroy(radpat);
		cairo_fill(cr);

		//
		cairo_pattern_t *linpat = cairo_pattern_create_linear(0.25, 0.35, 0.75, 0.65);
		//cairo_pattern_t *linpat = cairo_pattern_create_linear(0.25, 0.65, 0.75, 0.65);
		//cairo_pattern_t *linpat = cairo_pattern_create_linear(0.0, 0.65, 1.0, 0.65);
		cairo_pattern_add_color_stop_rgba(linpat, 0.00,  1, 1, 1, 0);
		cairo_pattern_add_color_stop_rgba(linpat, 0.25,  0, 1, 0, 0.5);
		cairo_pattern_add_color_stop_rgba(linpat, 0.50,  1, 1, 1, 0);
		cairo_pattern_add_color_stop_rgba(linpat, 0.75,  0, 0, 1, 0.5);
		cairo_pattern_add_color_stop_rgba(linpat, 1.00,  1, 1, 1, 0);

		cairo_rectangle(cr, 0.0, 0.0, 1, 1);
		cairo_set_source(cr, linpat);
		//cairo_pattern_destroy(linpat);
		cairo_fill(cr);
	}

	// save to a png file
	cairo_surface_write_to_png(surface, ".\\graphics_2d_data\\cairo\\tutorial_2_out.png");

	//
    cairo_destroy(cr);
	cairo_surface_destroy(surface);
}

void tutorial_3()
{
	const int width = 100, height = 100;

#if defined(CAIRO_HAS_SVG_SURFACE)
	cairo_surface_t *surface = cairo_svg_surface_create(".\\graphics_2d_data\\cairo\\tutorial_3.svg", (double)width, (double)height);
#elif defined(CAIRO_HAS_IMAGE_SURFACE)
	cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
#elif defined(CAIRO_HAS_PDF_SURFACE)
	cairo_surface_t *surface = cairo_pdf_surface_create(".\\graphics_2d_data\\cairo\\tutorial_3.pdf", (double)width, (double)height);
#elif defined(CAIRO_HAS_WIN32_SURFACE)
	cairo_surface_t *surface = cairo_win32_surface_create(hdc);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_dib(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_surface_create_with_ddb(CAIRO_FORMAT_ARGB32, width, height);
	//cairo_surface_t *surface = cairo_win32_printing_surface_create(hdc);
#endif

    cairo_t *cr = cairo_create(surface);

	cairo_scale(cr, (double)width, (double)height);

	{
		cairo_set_source_rgb(cr, 0, 0.5, 0);
		cairo_set_line_width(cr, 0.02);

		cairo_move_to(cr, 0.25, 0.25);
		cairo_line_to(cr, 0.5, 0.375);
		cairo_rel_line_to(cr, 0.25, -0.125);

		cairo_arc(cr, 0.5, 0.5, 0.25 * std::sqrt(2.0), -0.25 * M_PI, 0.25 * M_PI);  // CW

		cairo_rel_curve_to(cr, -0.25, -0.125, -0.25, 0.125, -0.5, 0);

		cairo_close_path(cr);

		cairo_stroke(cr);
	}

	{
		cairo_select_font_face(cr, "Georgia", CAIRO_FONT_SLANT_ITALIC, CAIRO_FONT_WEIGHT_BOLD);
		cairo_set_font_size(cr, 0.1);

		const char *text = "Hello";


		cairo_font_options_t *options = cairo_font_options_create();
		cairo_get_font_options(cr, options);
		cairo_set_font_options(cr, options);
		cairo_font_options_destroy(options);

		cairo_font_extents_t fe;
		cairo_text_extents_t te;
		cairo_font_extents(cr, &fe);
		cairo_text_extents(cr, text, &te);

		//
		cairo_set_source_rgb(cr, 0, 0, 1);

		cairo_move_to(cr, 0.5, 0.5);
		cairo_show_text(cr, text);
	}

	// save to a png file
	cairo_surface_write_to_png(surface, ".\\graphics_2d_data\\cairo\\tutorial_3_out.png");

	//
    cairo_destroy(cr);
	cairo_surface_destroy(surface);
}

}  // namespace local
}  // unnamed namespace

namespace cairo {

void tutorial()
{
	local::tutorial_1();
	local::tutorial_2();
	local::tutorial_3();
}

}  // namespace cairo
