//include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vxl {

int fhs_find_matches_example(int argc, char *argv[]);
int fhs_mr_find_matches_example(int argc, char *argv[]);
int fhs_match_tree_model_example(int argc, char *argv[]);

int msm_apply_tps_warp_example(int argc, char *argv[]);
int msm_build_shape_model_example(int argc, char *argv[]);
int msm_draw_points_on_image_example(int argc, char *argv[]);
int msm_draw_shape_modes_example(int argc, char *argv[]);
int msm_get_shape_params_example(int argc, char *argv[]);
int msm_plot_aligned_shapes_example(int argc, char *argv[]);

}  // namespace my_vxl

int vxl_main(int argc, char *argv[])
{
	int retval = 0;

	// fhs: Feature matching using Felzenszwalb and Huttenlocher's method.
	// [ref] http://paine.wiau.man.ac.uk/pub/doc_vxl/contrib/mul/fhs/html/index.html
	//	"Efficient Matching of Pictorial Structures", CVPR, 2000.
	retval = my_vxl::fhs_find_matches_example(argc, argv);
	retval = my_vxl::fhs_mr_find_matches_example(argc, argv);
	retval = my_vxl::fhs_match_tree_model_example(argc, argv);

	// msm: Manchester Shape Model library.
	// [ref] http://public.kitware.com/vxl/doc/release/contrib/mul/msm/html/index.html
	retval = my_vxl::msm_apply_tps_warp_example(argc, argv);
	retval = my_vxl::msm_build_shape_model_example(argc, argv);
	retval = my_vxl::msm_draw_points_on_image_example(argc, argv);
	retval = my_vxl::msm_draw_shape_modes_example(argc, argv);
	retval = my_vxl::msm_get_shape_params_example(argc, argv);
	retval = my_vxl::msm_plot_aligned_shapes_example(argc, argv);

	return retval;
}
