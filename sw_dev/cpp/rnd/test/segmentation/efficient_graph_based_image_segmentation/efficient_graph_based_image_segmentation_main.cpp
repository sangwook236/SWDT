#include "../efficient_graph_based_image_segmentation_lib/segment-image.h"
#include "../efficient_graph_based_image_segmentation_lib/image.h"
#include "../efficient_graph_based_image_segmentation_lib/misc.h"
#include "../efficient_graph_based_image_segmentation_lib/pnmfile.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

/*
	The program takes a color image (PPM format) and produces a segmentation
	with a random color assigned to each region.

	run "segment sigma k min input output".

	The parameters are: (see the paper for details)

	sigma: Used to smooth the input image before segmenting it.
	k: Value for the threshold function.
	min: Minimum component size enforced by post-processing.
	input: Input image.
	output: Output image.
*/

// [ref] ${Efficient_Graph_Based_Image_Segmentation_HOME}/segment.cpp
void sample()
{
#if 1
	const std::string input_filename("./segmentation_data/beach.ppm");
	const std::string output_filename("./segmentation_data/beach_segmented.ppm");
	const float sigma = 0.5f;
	const float k = 500.0f;
	const int min_size = 50;
#elif 
	const std::string input_filename("./segmentation_data/grain.ppm");
	const std::string output_filename("./segmentation_data/grain_segmented.ppm");
	const float sigma = 0.5f;
	const float k = 1000.0f;
	const int min_size = 100;
#endif

	std::cout << "loading input image." << std::endl;
	image<rgb> *input = loadPPM(input_filename.c_str());  // color

	std::cout << "processing" << std::endl;

	int num_ccs;

	image<rgb> *seg = NULL;
	{
		boost::timer::auto_cpu_timer timer;

		seg = segment_image(input, sigma, k, min_size, &num_ccs);
	}
	
	savePPM(seg, output_filename.c_str());

	std::cout << "got " << num_ccs << " components" << std::endl;

	// show results
	cv::Mat img(seg->height(), seg->width(), CV_8UC3, (void *)seg->data);

#if 1
	cv::imshow("segmented image", img);
#else
	double minVal, maxVal;
	cv::minMaxLoc(img, &minVal, &maxVal);
	cv::Mat tmp;
	img.convertTo(tmp, CV_8UC3, 255.0 / maxVal, 0.0);

	cv::imshow("segmented image", tmp);
#endif

	cv::waitKey(0);

	cv::destroyAllWindows();

	delete seg;
	seg = NULL;
}

}  // namespace local
}  // unnamed namespace

namespace my_efficient_graph_based_image_segmentation {

}  // namespace my_efficient_graph_based_image_segmentation

/*
[ref]
	"Efficient Graph-Based Image Segmentation", Pedro F. Felzenszwalb and Daniel P. Huttenlocher, IJCV, 2004.
	http://cs.brown.edu/~pff/segment/
*/

int efficient_graph_based_image_segmentation_main(int argc, char *argv[])
{
	local::sample();

	return 0;
}
