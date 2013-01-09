/*
* draw.c draws the images in the "Drawing with cairo" section as part
* of the the cairo tutorial
* Copyright (C) 2007  Nis Martensen
* Derived from draw.py
* <http://www.tortall.net/mu/wiki/CairoTutorial/draw.py?raw>
* Copyright (C) 2006-2007 Michael Urman
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
//#include "stdafx.h"
#include <cairo/cairo.h>
#include <cairo/cairo-svg.h>
#include <string>
#include <cstring>
#include <cmath>


#if defined(_MSC_VER)
const double M_PI = std::atan(1.0) * 4.0;
#endif

namespace {
namespace local {

void diagram (char *name);
void draw_diagram (char *name, cairo_t *cr);
void draw_setsourcergba (cairo_t *cr);
void draw_setsourcegradient (cairo_t *cr);
void path_diagram (cairo_t *cr);
void draw_path_curveto_hints (cairo_t *cr);
void draw_path_moveto (cairo_t *cr);
void draw_path_lineto (cairo_t *cr);
void draw_path_arcto (cairo_t *cr);
void draw_path_curveto (cairo_t *cr);
void draw_path_close (cairo_t *cr);
void draw_textextents (cairo_t *cr);

void diagram (char *name)
{
	cairo_surface_t *surf;
	cairo_t *cr;

	double width=120, height=120;
	double ux=2, uy=2;

	const std::string svg_filename(std::string("./graphics_2d_data/cairo/") + std::string(name) + std::string(".svg"));
	const std::string png_filename(std::string("./graphics_2d_data/cairo/") + std::string(name) + std::string(".png"));

	surf = cairo_svg_surface_create (svg_filename.c_str(), width, height);
	cr = cairo_create (surf);

	cairo_scale (cr, width, height);
	cairo_set_line_width (cr, 0.01);

	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_set_source_rgb (cr, 1, 1, 1);
	cairo_fill (cr);

	draw_diagram (name, cr);

	cairo_device_to_user_distance (cr, &ux, &uy);
	if (ux < uy)
		ux = uy;
	cairo_set_line_width (cr, ux);
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_stroke (cr);

	/* write output and clean up */
	cairo_surface_write_to_png (surf, png_filename.c_str());
	cairo_destroy (cr);
	cairo_surface_destroy (surf);
}

void draw_diagram (char *name, cairo_t *cr)
{
	if (strcmp(name, "setsourcergba") == 0) {
		draw_setsourcergba (cr);
	} else if (strcmp(name, "setsourcegradient") == 0) {
		draw_setsourcegradient (cr);
	} else if (strcmp(name, "path-moveto") == 0) {
		draw_path_moveto (cr);
	} else if (strcmp(name, "path-lineto") == 0) {
		draw_path_lineto (cr);
	} else if (strcmp(name, "path-arcto") == 0) {
		draw_path_arcto (cr);
	} else if (strcmp(name, "path-curveto") == 0) {
		draw_path_curveto_hints (cr);
		draw_path_curveto (cr);
	} else if (strcmp(name, "path-close") == 0) {
		draw_path_close (cr);
	} else if (strcmp(name, "textextents") == 0) {
		draw_textextents (cr);
	}

	if (strncmp(name, "path-", 5) == 0)
		path_diagram (cr);
}

void draw_setsourcergba (cairo_t *cr)
{
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_move_to (cr, 0, 0);
	cairo_line_to (cr, 1, 1);
	cairo_move_to (cr, 1, 0);
	cairo_line_to (cr, 0, 1);
	cairo_set_line_width (cr, 0.2);
	cairo_stroke (cr);

	cairo_rectangle (cr, 0, 0, 0.5, 0.5);
	cairo_set_source_rgba (cr, 1, 0, 0, 0.80);
	cairo_fill (cr);

	cairo_rectangle (cr, 0, 0.5, 0.5, 0.5);
	cairo_set_source_rgba (cr, 0, 1, 0, 0.60);
	cairo_fill (cr);

	cairo_rectangle (cr, 0.5, 0, 0.5, 0.5);
	cairo_set_source_rgba (cr, 0, 0, 1, 0.40);
	cairo_fill (cr);
}

void draw_setsourcegradient (cairo_t *cr)
{
	int i, j;
	cairo_pattern_t *radpat, *linpat;

	radpat = cairo_pattern_create_radial (0.25, 0.25, 0.1, 0.5, 0.5, 0.5);
	cairo_pattern_add_color_stop_rgb (radpat, 0, 1.0, 0.8, 0.8);
	cairo_pattern_add_color_stop_rgb (radpat, 1, 0.9, 0.0, 0.0);

	for (i = 1; i < 10; i++)
		for (j = 1; j < 10; j++)
			cairo_rectangle (cr, i / 10.0 - 0.04, j / 10.0 - 0.04,
			0.08, 0.08);
	cairo_set_source (cr, radpat);
	cairo_fill (cr);

	linpat = cairo_pattern_create_linear (0.25, 0.35, 0.75, 0.65);
	cairo_pattern_add_color_stop_rgba (linpat, 0.00, 1, 1, 1, 0);
	cairo_pattern_add_color_stop_rgba (linpat, 0.25, 0, 1, 0, 0.5);
	cairo_pattern_add_color_stop_rgba (linpat, 0.50, 1, 1, 1, 0);
	cairo_pattern_add_color_stop_rgba (linpat, 0.75, 0, 0, 1, 0.5);
	cairo_pattern_add_color_stop_rgba (linpat, 1.00, 1, 1, 1, 0);

	cairo_rectangle (cr, 0.0, 0.0, 1, 1);
	cairo_set_source (cr, linpat);
	cairo_fill (cr);
}

void path_diagram (cairo_t *cr)
{
	cairo_path_t *path;
	cairo_path_data_t *data;
	double x, y, px = 3, py = 3;

	path = cairo_copy_path_flat (cr);

	cairo_device_to_user_distance (cr, &px, &py);
	if (px < py)
		px = py;
	cairo_set_line_width (cr, px);
	cairo_set_source_rgb (cr, 0, 0.6, 0);
	cairo_stroke (cr);

	if (path->num_data > 1) {
		/*
		* Draw markers at the first and the last point of the
		* path, but only if the path is not closed.
		*
		* If the last path manipulation was a cairo_close(),
		* then we can detect this at the end of the path->data
		* array. The CLOSE_PATH element will be followed by a
		* MOVE_TO element (since cairo 1.2.4), so we need to
		* check position path->num_data - 3.
		*
		* More details can be found here:
		* <http://cairographics.org/manual/cairo-Paths.html#cairo-close-path>
		* <http://cairographics.org/manual/cairo-Paths.html#cairo-path-data-t>
		*/
		if (path->data[path->num_data-3].header.type != CAIRO_PATH_CLOSE_PATH) {
			/* Get the first point in the path */
			data = &path->data[0];
			x = data[1].point.x;
			y = data[1].point.y;

			px = 5; py = 5;
			cairo_device_to_user_distance (cr, &px, &py);
			if (px < py)
				px = py;

			cairo_arc (cr, x, y, px, 0, 2*M_PI);
			cairo_set_source_rgba (cr, 0.0, 0.6, 0.0, 0.5);
			cairo_fill(cr);

			/*
			* Because cairo_copy_path_flat() was used to
			* retrieve this path, there is no CURVE_TO
			* element, so the elements all have a length of
			* 2. The index of the last element must be
			* path->num_data - 2.
			*/
			data = &path->data[path->num_data-2];
			x = data[1].point.x;
			y = data[1].point.y;
			cairo_arc (cr, x, y, px, 0, 2*M_PI);
			cairo_set_source_rgba (cr, 0.0, 0.0, 0.75, 0.5);
			cairo_fill (cr);
		}
	}

	cairo_path_destroy (path);
}

void draw_path_curveto_hints (cairo_t *cr)
{
	double px = 3, py = 3;
	cairo_save (cr);
	cairo_device_to_user_distance (cr, &px, &py);
	if (px < py)
		px = py;
	cairo_set_source_rgba (cr, 0.5, 0, 0, 0.5);

	cairo_new_sub_path (cr);
	cairo_arc (cr, 0.5, 0.625, px, 0, 2*M_PI);
	cairo_fill (cr);
	cairo_arc (cr, 0.5, 0.875, px, 0, 2*M_PI);
	cairo_fill (cr);

	px = 2; py = 2;
	cairo_device_to_user_distance (cr, &px, &py);
	if (px < py)
		px = py;
	cairo_set_line_width (cr, px);
	cairo_set_source_rgba (cr, 0.5, 0, 0, 0.25);

	cairo_move_to (cr, 0.25, 0.75);
	cairo_rel_line_to (cr, 0.25, 0.125);
	cairo_stroke (cr);

	cairo_move_to (cr, 0.75, 0.75);
	cairo_rel_line_to (cr, -0.25, -0.125);
	cairo_stroke (cr);

	cairo_restore (cr);
}

void draw_path_moveto (cairo_t *cr)
{
	cairo_set_line_width (cr, 0.1);
	cairo_set_source_rgb (cr, 0, 0, 0);

	cairo_move_to (cr, 0.25, 0.25);
}

void draw_path_lineto (cairo_t *cr)
{
	draw_path_moveto (cr);

	cairo_line_to (cr, 0.5, 0.375);
	cairo_rel_line_to (cr, 0.25, -0.125);
}

void draw_path_arcto (cairo_t *cr)
{
	draw_path_lineto (cr);

	cairo_arc (cr, 0.5, 0.5, 0.25 * std::sqrt(2.0), -0.25 * M_PI, 0.25 * M_PI);
}

void draw_path_curveto (cairo_t *cr)
{
	draw_path_arcto (cr);

	cairo_rel_curve_to (cr, -0.25, -0.125, -0.25, 0.125, -0.5, 0);
}

void draw_path_close (cairo_t *cr)
{
	draw_path_curveto (cr);

	cairo_close_path (cr);
}

void draw_textextents (cairo_t *cr)
{
	double x, y, px, ux=1, uy=1, dashlength;
	char text[]="joy";
	cairo_font_extents_t fe;
	cairo_text_extents_t te;

	cairo_set_font_size (cr, 0.5);

	/* Drawing code goes here */
	cairo_set_source_rgb (cr, 0.0, 0.0, 0.0);
	cairo_select_font_face (cr, "Georgia",
		CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
	cairo_font_extents (cr, &fe);

	cairo_device_to_user_distance (cr, &ux, &uy);
	if (ux > uy)
		px = ux;
	else
		px = uy;
	cairo_font_extents (cr, &fe);
	cairo_text_extents (cr, text, &te);
	x = 0.5 - te.x_bearing - te.width / 2;
	y = 0.5 - fe.descent + fe.height / 2;

	/* baseline, descent, ascent, height */
	cairo_set_line_width (cr, 4*px);
	dashlength = 9*px;
	cairo_set_dash (cr, &dashlength, 1, 0);
	cairo_set_source_rgba (cr, 0, 0.6, 0, 0.5);
	cairo_move_to (cr, x + te.x_bearing, y);
	cairo_rel_line_to (cr, te.width, 0);
	cairo_move_to (cr, x + te.x_bearing, y + fe.descent);
	cairo_rel_line_to (cr, te.width, 0);
	cairo_move_to (cr, x + te.x_bearing, y - fe.ascent);
	cairo_rel_line_to (cr, te.width, 0);
	cairo_move_to (cr, x + te.x_bearing, y - fe.height);
	cairo_rel_line_to (cr, te.width, 0);
	cairo_stroke (cr);

	/* extents: width & height */
	cairo_set_source_rgba (cr, 0, 0, 0.75, 0.5);
	cairo_set_line_width (cr, px);
	dashlength = 3*px;
	cairo_set_dash (cr, &dashlength, 1, 0);
	cairo_rectangle (cr, x + te.x_bearing, y + te.y_bearing, te.width, te.height);
	cairo_stroke (cr);

	/* text */
	cairo_move_to (cr, x, y);
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_show_text (cr, text);

	/* bearing */
	cairo_set_dash (cr, NULL, 0, 0);
	cairo_set_line_width (cr, 2 * px);
	cairo_set_source_rgba (cr, 0, 0, 0.75, 0.5);
	cairo_move_to (cr, x, y);
	cairo_rel_line_to (cr, te.x_bearing, te.y_bearing);
	cairo_stroke (cr);

	/* text's advance */
	cairo_set_source_rgba (cr, 0, 0, 0.75, 0.5);
	cairo_arc (cr, x + te.x_advance, y + te.y_advance, 5 * px, 0, 2 * M_PI);
	cairo_fill (cr);

	/* reference point */
	cairo_arc (cr, x, y, 5 * px, 0, 2 * M_PI);
	cairo_set_source_rgba (cr, 0.75, 0, 0, 0.5);
	cairo_fill (cr);
}

}  // namespace local
}  // unnamed namespace

namespace my_cairo {

void drawing_illustration()
{
	local::diagram("setsourcergba");
	local::diagram("setsourcegradient");
	local::diagram("path-moveto");
	local::diagram("path-lineto");
	local::diagram("path-arcto");
	local::diagram("path-curveto");
	local::diagram("path-close");
	local::diagram("textextents");
}

}  // namespace my_cairo
