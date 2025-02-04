//include "stdafx.h"
#include <mbl/mbl_read_props.h>
#include <mbl/mbl_exception.h>
#include <mbl/mbl_parse_colon_pairs_list.h>
#include <vul/vul_arg.h>
#include <vul/vul_string.h>
#include <vcl_sstream.h>
#include <vcl_fstream.h>
#include <vcl_string.h>
#include <vsl/vsl_quick_file.h>
#include <msm/msm_aligner.h>
#include <msm/msm_add_all_loaders.h>
#include <msm/utils/msm_draw_shape_to_eps.h>


namespace {
namespace local {

/*
Parameter file format:
<START FILE>
//: Aligner for shape model
aligner: msm_similarity_aligner

//: Radius of points to display (if <0, then don't draw points)
point_radius: 0.5

point_colour: black
mean_colour: red

// Approximate width of region to display shape
display_width: 256

// When supplied, draw curves for mean shape
curves_path: face_front.crvs
line_colour: green

output_path: sim_aligned.eps

image_dir: /home/images/
points_dir: /home/points/
images: {
image1.pts : image1.jpg
image2.pts : image2.jpg
}

<END FILE>
*/

void print_usage()
{
	vcl_cout << "msm_plot_aligned_shapes -p param_file\n"
		<< "Generate EPS file containing aligned shape points.\n"
		<< vcl_endl;

	vul_arg_display_usage_and_exit();
}

//: Structure to hold parameters
struct tool_params
{
	//: Aligner for shape model
	vcl_auto_ptr<msm_aligner> aligner;

	vcl_string curves_path;

	//: Colour to draw curves
	vcl_string line_colour;

	//: Directory containing images
	vcl_string image_dir;

	//: Directory containing points
	vcl_string points_dir;

	//: Approximate width of region to display shape
	int display_width;

	//: Point colour
	vcl_string point_colour;
	vcl_string mean_colour;

	//: Radius of points to display (if <0, then don't draw points)
	double point_radius;

	//: File to save EPS file to
	vcl_string output_path;

	//: List of image names
	vcl_vector<vcl_string> image_names;

	//: List of points file names
	vcl_vector<vcl_string> points_names;

	//: Parse named text file to read in data
	//  Throws a mbl_exception_parse_error if fails
	void read_from_file(const vcl_string& path);
};

//: Parse named text file to read in data
//  Throws a mbl_exception_parse_error if fails
void tool_params::read_from_file(const vcl_string &path)
{
	vcl_ifstream ifs(path.c_str());
	if (!ifs)
	{
		vcl_string error_msg = "Failed to open file: " + path;
		throw (mbl_exception_parse_error(error_msg));
	}

	mbl_read_props_type props = mbl_read_props_ws(ifs);

	curves_path = props["curves_path"];

	point_radius = vul_string_atof(props.get_optional_property("point_radius", "1.5"));
	display_width = vul_string_atoi(props.get_optional_property("display_width", "100"));
	point_colour = props.get_optional_property("point_colour", "black");
	mean_colour = props.get_optional_property("mean_colour", "red");
	line_colour = props.get_optional_property("line_colour", "green");

	image_dir = props.get_optional_property("image_dir", "./");
	points_dir = props.get_optional_property("points_dir", "./");
	output_path = props.get_optional_property("output_path", "aligned.eps");

	{
		vcl_string aligner_str = props.get_required_property("aligner");
		vcl_stringstream ss(aligner_str);
		aligner = msm_aligner::create_from_stream(ss);
	}

	mbl_parse_colon_pairs_list(props.get_required_property("images"), points_names, image_names);

	// Don't look for unused props so can use a single common parameter file.
}

void load_shapes(const vcl_string &points_dir, const vcl_vector<vcl_string> &filenames, vcl_vector<msm_points> &shapes)
{
	unsigned n = filenames.size();

	shapes.resize(n);
	for (unsigned i = 0; i < n; ++i)
	{
		vcl_string path = points_dir + "/" + filenames[i];
		if (!shapes[i].read_text_file(path))
		{
			mbl_exception_parse_error x("Failed to load points from " + path);
			mbl_exception_error(x);
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_vxl {

// [ref] ${VXL_HOME}/contrib/mul/msm/tools/msm_plot_aligned_shapes.cxx
int msm_plot_aligned_shapes_example(int argc, char *argv[])
{
	vul_arg<vcl_string> param_path("-p","Parameter filename");
	vul_arg_parse(argc,argv);

	msm_add_all_loaders();

	if (param_path() == "")
	{
		local::print_usage();
		return 0;
	}

	local::tool_params params;
	try
	{
		params.read_from_file(param_path());
	}
	catch (const mbl_exception_parse_error &e)
	{
		vcl_cerr << "Error: " << e.what() << vcl_endl;
		return 1;
	}

	msm_curves curves;
	if (params.curves_path != "")
	{
		if (!curves.read_text_file(params.curves_path))
			vcl_cerr << "Failed to read in curves from " << params.curves_path << vcl_endl;
	}

	// Load in all the shapes
	vcl_vector<msm_points> shapes;
	local::load_shapes(params.points_dir, params.points_names, shapes);

	msm_points mean_shape;
	vcl_vector<vnl_vector<double> > pose_to_ref;
	vnl_vector<double> average_pose;

	params.aligner->align_set(shapes, mean_shape, pose_to_ref, average_pose);

	// Apply pose_to_ref to map each to the reference frame
	vgl_box_2d<double> bounds;
	for (unsigned i = 0; i < shapes.size(); ++i)
	{
		params.aligner->apply_transform(shapes[i], pose_to_ref[i], shapes[i]);
		bounds.add(shapes[i].bounds());
	}

	int d_width = params.display_width;
	double b = 0.05;
	double s = (1.0 - 2 * b) * d_width / bounds.width();
	int d_height = int(s * bounds.height() / (1.0 - 2 * b));

	double tx = b*bounds.width() - bounds.min_x();
	double ty = b*bounds.height() - bounds.min_y();

	mbl_eps_writer writer(params.output_path.c_str(), d_width, d_height);

	writer.set_colour(params.point_colour);
	for (unsigned i = 0; i < shapes.size(); ++i)
	{
		msm_points points = shapes[i];
		points.translate_by(tx, ty);
		points.scale_by(s);
		msm_draw_points_to_eps(writer,points, params.point_radius);
	}

	// Draw mean points
	msm_points points = mean_shape;
	points.translate_by(tx, ty);
	points.scale_by(s);
	writer.set_colour(params.mean_colour);
	msm_draw_points_to_eps(writer, points, 3.0 * params.point_radius);

	// Draw curves if available
	writer.set_colour(params.line_colour);
	msm_draw_shape_to_eps(writer, points, curves);

	writer.close();
	vcl_cout << "Plotted shapes to " << params.output_path << vcl_endl;

	return 0;
}

}  // namespace my_vxl

