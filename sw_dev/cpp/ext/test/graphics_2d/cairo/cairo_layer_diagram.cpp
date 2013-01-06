/*
* diagram.c draws the layer diagrams as part of the the cairo tutorial
* Copyright (C) 2007  Nis Martensen
* Derived from diagram.py
* <http://www.tortall.net/mu/wiki/CairoTutorial/diagram.py?raw>
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


namespace {
namespace local {

void diagram (char *name, double *alpha);
void draw_source (char *name, cairo_t *cr);
void draw_mask (char *name, cairo_t *cr);
void draw_dest (char *name, cairo_t *cr);
void diagram_draw_source (cairo_t *cr);
void diagram_draw_mask (cairo_t *cr);
void diagram_draw_mask_pattern (cairo_t *cr, cairo_pattern_t *pat);
void diagram_draw_dest (cairo_t *cr);
void stroke_draw_mask (cairo_t *cr);
void stroke_draw_dest (cairo_t *cr);
void fill_draw_mask (cairo_t *cr);
void fill_draw_dest (cairo_t *cr);
void showtext_draw_mask (cairo_t *cr);
void showtext_draw_dest (cairo_t *cr);
void paint_draw_source (cairo_t *cr);
void paint_draw_dest (cairo_t *cr);
void mask_draw_source (cairo_t *cr);
void mask_draw_mask (cairo_t *cr);
void mask_draw_dest (cairo_t *cr);

void diagram (char *name, double *alpha)
{
	cairo_surface_t *surf;
	cairo_t *cr;
	cairo_matrix_t mat;

	double width=160, height=120;
	double ux=2, uy=2;

	const std::string svg_filename(std::string("./graphics_2d_data/cairo/") + std::string(name) + std::string(".svg"));
	const std::string png_filename(std::string("./graphics_2d_data/cairo/") + std::string(name) + std::string(".png"));

	surf = cairo_svg_surface_create (svg_filename.c_str(), width, height);
	cr = cairo_create (surf);

	/*
	* show layers separately on the right
	*/
	cairo_save (cr);
	cairo_scale (cr, height/3, height/3);
	cairo_translate (cr, 3*width/height-1, 0);
	/* source */
	cairo_save (cr);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_source (name, cr);
	cairo_pop_group_to_source (cr);
	cairo_paint (cr);
	cairo_restore (cr);
	/* mask */
	cairo_translate (cr, 0, 1);
	cairo_save (cr);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_mask (name, cr);
	cairo_pop_group_to_source (cr);
	cairo_paint (cr);
	cairo_restore (cr);
	/* destination */
	cairo_translate (cr, 0, 1);
	cairo_save (cr);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_dest (name, cr);
	cairo_pop_group_to_source (cr);
	cairo_paint (cr);
	cairo_restore (cr);
	cairo_restore (cr);

	/* draw a border around the layers */
	cairo_save (cr);
	cairo_scale (cr, height/3, height/3);
	cairo_translate (cr, 3*width/height-1, 0);
	cairo_device_to_user_distance (cr, &ux, &uy);
	if (ux < uy)
		ux = uy;
	cairo_set_line_width (cr, ux);
	cairo_rectangle (cr, 0, 0, 1, 3);
	cairo_clip_preserve (cr);
	cairo_stroke (cr);
	cairo_rectangle (cr, 0, 1, 1, 1);
	cairo_stroke (cr);
	cairo_restore (cr);

	/*
	* layer diagram on the left
	*/

	/* destination layer */
	cairo_save (cr);
	cairo_scale (cr, width-height/3, height);
	cairo_matrix_init (&mat, 0.6, 0, 1.0/3, 0.5, 0.02, 0.45);
	cairo_transform (cr, &mat);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_dest (name, cr);
	/* this layer gets a black border */
	cairo_set_source_rgb (cr, 0, 0, 0);
	ux = 2; uy = 2;
	cairo_device_to_user_distance (cr, &ux, &uy);
	if (ux < uy)
		ux = uy;
	cairo_set_line_width (cr, ux);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_stroke (cr);
	cairo_pop_group_to_source (cr);
	cairo_paint_with_alpha (cr, alpha[0]);
	cairo_restore (cr);

	/* mask layer */
	cairo_save (cr);
	cairo_scale (cr, width-height/3, height);
	cairo_matrix_init (&mat, 0.6, 0, 1.0/3, 0.5, 0.04, 0.25);
	cairo_transform (cr, &mat);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_mask (name, cr);
	cairo_pop_group_to_source (cr);
	cairo_paint_with_alpha (cr, alpha[1]);
	cairo_restore (cr);

	/* source layer */
	cairo_save (cr);
	cairo_scale (cr, width-height/3, height);
	cairo_matrix_init (&mat, 0.6, 0, 1.0/3, 0.5, 0.06, 0.05);
	cairo_transform (cr, &mat);
	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	draw_source (name, cr);
	cairo_pop_group_to_source (cr);
	cairo_paint_with_alpha (cr, alpha[2]);
	cairo_restore (cr);

	/* write output and clean up */
	cairo_surface_write_to_png (surf, png_filename.c_str());
	cairo_destroy (cr);
	cairo_surface_destroy (surf);
}

void draw_source (char *name, cairo_t *cr)
{
	if (strcmp (name, "paint") == 0) {
		paint_draw_source (cr);
	} else if (strcmp (name, "mask") == 0) {
		mask_draw_source (cr);
	} else {
		diagram_draw_source (cr);
	}
}

void draw_mask (char *name, cairo_t *cr)
{
	if (strcmp (name, "stroke") == 0) {
		stroke_draw_mask (cr);
	} else if (strcmp (name, "fill") == 0) {
		fill_draw_mask (cr);
	} else if (strcmp (name, "showtext") == 0) {
		showtext_draw_mask (cr);
	} else if (strcmp (name, "paint") == 0) {
	} else if (strcmp (name, "mask") == 0) {
		mask_draw_mask (cr);
	} else {
		diagram_draw_mask (cr);
	}
}
void draw_dest (char *name, cairo_t *cr)
{
	if (strcmp(name, "stroke") == 0) {
		stroke_draw_dest (cr);
	} else if (strcmp(name, "fill") == 0) {
		fill_draw_dest (cr);
	} else if (strcmp(name, "showtext") == 0) {
		showtext_draw_dest (cr);
	} else if (strcmp(name, "paint") == 0) {
		paint_draw_dest (cr);
	} else if (strcmp(name, "mask") == 0) {
		mask_draw_dest (cr);
	} else {
		diagram_draw_dest (cr);
	}
}

void diagram_draw_source (cairo_t *cr)
{
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);
}

void diagram_draw_mask (cairo_t *cr)
{
	cairo_set_source_rgb (cr, 1, 0.9, 0.6);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);
}

void diagram_draw_mask_pattern (cairo_t *cr, cairo_pattern_t *pat)
{
	cairo_set_source_rgb (cr, 1, 0.9, 0.6);
	cairo_mask (cr, pat);
}

void diagram_draw_dest (cairo_t *cr)
{
	cairo_set_source_rgb (cr, 1, 1, 1);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);
}

void stroke_draw_mask (cairo_t *cr)
{
	double px=1, py=1;
	cairo_pattern_t *pat;

	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_rectangle (cr, 0.20, 0.20, 0.6, 0.6);
	cairo_rectangle (cr, 0.30, 0.30, 0.4, 0.4);
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_EVEN_ODD);
	cairo_fill (cr);
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_WINDING);

	pat = cairo_pop_group (cr);

	diagram_draw_mask_pattern (cr, pat);

	cairo_rectangle (cr, 0.25, 0.25, 0.5, 0.5);
	cairo_set_source_rgb (cr, 0, 0.6, 0);

	cairo_device_to_user_distance (cr, &px, &py);
	if (px < py)
		px = py;
	cairo_set_line_width (cr, px);
	cairo_stroke (cr);
}

void stroke_draw_dest (cairo_t *cr)
{
	diagram_draw_dest (cr);

	cairo_set_line_width (cr, 0.1);
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_rectangle (cr, 0.25, 0.25, 0.5, 0.5);
	cairo_stroke (cr);
}

void fill_draw_mask (cairo_t *cr)
{
	double px=1, py=1;
	cairo_pattern_t *pat;

	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_rectangle (cr, 0.25, 0.25, 0.5, 0.5);
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_EVEN_ODD);
	cairo_fill (cr);
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_WINDING);
	pat = cairo_pop_group (cr);

	diagram_draw_mask_pattern (cr, pat);

	cairo_rectangle (cr, 0.25, 0.25, 0.5, 0.5);
	cairo_set_source_rgb (cr, 0, 0.6, 0);
	cairo_device_to_user_distance (cr, &px, &py);
	if (px < py)
		px = py;
	cairo_set_line_width (cr, px);
	cairo_stroke (cr);
}

void fill_draw_dest (cairo_t *cr)
{
	diagram_draw_dest (cr);

	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_rectangle (cr, 0.25, 0.25, 0.5, 0.5);
	cairo_fill (cr);
}

void showtext_draw_mask (cairo_t *cr)
{
	cairo_text_extents_t te;
	double ux=1, uy=1;

	/* yellow mask color */
	cairo_set_source_rgb (cr, 1, 0.9, 0.6);

	/* rectangle with an "a"-shaped hole */
	cairo_select_font_face (cr, "Georgia",
		CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
	cairo_set_font_size (cr, 1.2);
	cairo_text_extents (cr, "a", &te);

	cairo_push_group (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_move_to (cr, 0.5 - te.width / 2 - te.x_bearing,
		0.5 - te.height / 2 - te.y_bearing);
	cairo_text_path (cr, "a");
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_EVEN_ODD);
	cairo_fill (cr);
	cairo_set_fill_rule (cr, CAIRO_FILL_RULE_WINDING);
	cairo_pop_group_to_source (cr);
	cairo_paint (cr);

	/* show the outline of the glyph with a green line */
	cairo_move_to (cr, 0.5 - te.width / 2 - te.x_bearing,
		0.5 - te.height / 2 - te.y_bearing);
	cairo_set_source_rgb (cr, 0, 0.6, 0);

	cairo_device_to_user_distance (cr, &ux, &uy);
	if (ux < uy)
		ux = uy;
	cairo_set_line_width (cr, ux);

	cairo_text_path (cr, "a");
	cairo_stroke (cr);
}

void showtext_draw_dest (cairo_t *cr)
{
	cairo_text_extents_t te;

	/* white background */
	cairo_set_source_rgb (cr, 1, 1, 1);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);

	/* black letter "a" */
	cairo_set_source_rgb (cr, 0.0, 0.0, 0.0);
	cairo_select_font_face (cr, "Georgia",
		CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
	cairo_set_font_size (cr, 1.2);
	cairo_text_extents (cr, "a", &te);
	cairo_move_to (cr, 0.5 - te.width / 2 - te.x_bearing,
		0.5 - te.height / 2 - te.y_bearing);
	cairo_show_text (cr, "a");
}

void paint_draw_source (cairo_t *cr)
{
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);
}

void paint_draw_dest (cairo_t *cr)
{
	diagram_draw_dest (cr);
	cairo_set_source_rgb (cr, 0, 0, 0);
	cairo_paint_with_alpha (cr, 0.5);
}

void mask_draw_source (cairo_t *cr)
{
	cairo_pattern_t *linpat;

	linpat = cairo_pattern_create_linear (0, 0, 1, 1);
	cairo_pattern_add_color_stop_rgb (linpat, 0, 0, 0.3, 0.8);
	cairo_pattern_add_color_stop_rgb (linpat, 1, 0, 0.8, 0.3);

	cairo_set_source (cr, linpat);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_fill (cr);
}

void mask_draw_mask (cairo_t *cr)
{
	cairo_pattern_t *radialinv;

	radialinv = cairo_pattern_create_radial (0.5, 0.5, 0.25, 0.5, 0.5, 0.75);
	cairo_pattern_add_color_stop_rgba (radialinv, 0, 0, 0, 0, 0);
	cairo_pattern_add_color_stop_rgba (radialinv, 0.5, 0, 0, 0, 1);

	cairo_save (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	diagram_draw_mask_pattern (cr, radialinv);
	cairo_restore (cr);
}

void mask_draw_dest (cairo_t *cr)
{
	cairo_pattern_t *linpat, *radpat;
	linpat = cairo_pattern_create_linear (0, 0, 1, 1);
	cairo_pattern_add_color_stop_rgb (linpat, 0, 0, 0.3, 0.8);
	cairo_pattern_add_color_stop_rgb (linpat, 1, 0, 0.8, 0.3);

	radpat = cairo_pattern_create_radial (0.5, 0.5, 0.25, 0.5, 0.5, 0.75);
	cairo_pattern_add_color_stop_rgba (radpat, 0, 0, 0, 0, 1);
	cairo_pattern_add_color_stop_rgba (radpat, 0.5, 0, 0, 0, 0);

	diagram_draw_dest (cr);
	cairo_save (cr);
	cairo_rectangle (cr, 0, 0, 1, 1);
	cairo_clip (cr);
	cairo_set_source (cr, linpat);
	cairo_mask (cr, radpat);
	cairo_restore (cr);
}

}  // namespace local
}  // unnamed namespace

namespace cairo {

void layer_diagram()
{
	double alpha[3];

	alpha[0]=1.0;  alpha[1]=0.15;  alpha[2]=0.15;
	local::diagram("destination", alpha);

	alpha[0]=0.15;  alpha[1]=1.0;  alpha[2]=0.15;
	local::diagram("the-mask", alpha);

	alpha[0]=0.15;  alpha[1]=0.15;  alpha[2]=1.0;
	local::diagram("source", alpha);

	alpha[0]=1.0;  alpha[1]=0.8;  alpha[2]=0.4;
	local::diagram("stroke", alpha);

	alpha[0]=1.0;  alpha[1]=0.8;  alpha[2]=0.4;
	local::diagram("fill", alpha);

	alpha[0]=1.0;  alpha[1]=0.8;  alpha[2]=0.4;
	local::diagram("showtext", alpha);

	alpha[0]=1.0;  alpha[1]=0.8;  alpha[2]=0.4;
	local::diagram("paint", alpha);

	alpha[0]=1.0;  alpha[1]=0.8;  alpha[2]=0.4;
	local::diagram("mask", alpha);
}

}  // namespace cairo
