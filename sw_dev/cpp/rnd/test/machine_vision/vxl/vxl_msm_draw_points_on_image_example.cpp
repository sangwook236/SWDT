//include "stdafx.h"
#include <mbl/mbl_eps_writer.h>
#include <msm/msm_points.h>
#include <msm/utils/msm_draw_shape_to_eps.h>
#include <vil/vil_resample_bilin.h>
#include <vil/algo/vil_gauss_filter.h>
#include <msm/msm_curve.h>
#include <vgl/vgl_point_2d.h>
#include <vul/vul_arg.h>
#include <vil/vil_load.h>


namespace {
namespace local {

// Note: Currently doesn't quite deal with images correctly - need a
// half pixel offset so (0,0) is the centre of the pixel.

void print_usage()
{
	vcl_cout << "msm_draw_points_on_image  -p points.pts -c curves.crvs -i image.jpg -o image+pts.eps\n"
		<< "Load in points and curves.\n"
		<< "Writes out eps file displaying the curves.\n"
		<< "If image supplied, then includes that too."
		<< vcl_endl;

	vul_arg_display_usage_and_exit();
}

}  // namespace local
}  // unnamed namespace

namespace my_vxl {

// [ref] ${VXL_HOME}/contrib/mul/msm/tools/msm_draw_points_on_image.cxx
int msm_draw_points_on_image_example(int argc, char *argv[])
{
	vul_arg<vcl_string> curves_path("-c", "File containing curves");
	vul_arg<vcl_string> pts_path("-p", "File containing points");
	vul_arg<vcl_string> image_path("-i", "Image");
	vul_arg<vcl_string> out_path("-o", "Output path","image+pts.eps");
	vul_arg<vcl_string> line_colour("-lc", "Line colour","yellow");
	vul_arg<vcl_string> pt_colour("-pc", "Point colour","none");
	vul_arg<vcl_string> pt_edge_colour("-pbc", "Point border colour","none");
	vul_arg<double> pt_radius("-pr", "Point radius",2.0);
	vul_arg<double> scale("-s", "Scaling to apply",1.0);

	vul_arg_parse(argc, argv);

	if (pts_path() == "")
	{
		local::print_usage();
		return 0;
	}

	msm_curves curves;
	if (curves_path() != "" && !curves.read_text_file(curves_path()))
		vcl_cerr << "Failed to read in curves from " << curves_path() << vcl_endl;

	msm_points points;
	if (!points.read_text_file(pts_path()))
	{
		vcl_cerr << "Failed to read points from " << pts_path() << vcl_endl;
		return 2;
	}
	vcl_vector<vgl_point_2d<double> > pts;
	points.get_points(pts);

	//================ Attempt to load image ========
	vil_image_view<vxl_byte> image;
	if (image_path() != "")
	{
		image = vil_load(image_path().c_str());
		if (image.size() == 0)
		{
			vcl_cout << "Failed to load image from " << image_path() << vcl_endl;
			return 1;
		}
		vcl_cout << "Image is " << image << vcl_endl;
	}

	if (scale() > 1.001 || scale() < 0.999)
	{
		// Scale image and points
		vil_image_view<vxl_byte> image2;
		image2.deep_copy(image);
		if (scale() < 0.51)
			vil_gauss_filter_2d(image, image2, 1.0, 3);
		vil_resample_bilin(image2, image, int(0.5 + scale() * image.ni()), int(0.5 + scale() * image.nj()));

		points.scale_by(scale());
	}


	// Compute bounding box of points
	vgl_box_2d<double> bbox = points.bounds();
	bbox.scale_about_centroid(1.05);  // Add a border

	// If an image is supplied, use that to define bounding box
	if (image.size() > 0)
	{
		bbox = vgl_box_2d<double>(0, image.ni(), 0,image.nj());
	}

	vgl_point_2d<double> blo = bbox.min_point();

	// Translate all points to allow for shift of origin
	points.translate_by(-blo.x(), -blo.y());

	mbl_eps_writer writer(out_path().c_str(), bbox.width(), bbox.height());

	if (image.size() > 0)
		writer.draw_image(image, 0, 0, 1, 1);

	if (pt_colour() != "none")
	{
		// Draw all the points
		writer.set_colour(pt_colour());
		msm_draw_points_to_eps(writer, points, pt_radius());
	}

	if (pt_edge_colour() != "none")
	{
		// Draw disks around all the points
		writer.set_colour(pt_edge_colour());
		msm_draw_points_to_eps(writer, points, pt_radius(), false);
	}

	if (curves.size() > 0 && line_colour() != "none")
	{
		// Draw all the lines
		writer.set_colour(line_colour());
		msm_draw_shape_to_eps(writer, points, curves);
	}
	writer.close();

	vcl_cout << "Graphics saved to " << out_path() << vcl_endl;

	return 0;
}

}  // namespace my_vxl

